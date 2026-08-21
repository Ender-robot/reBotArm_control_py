"""DM 机械臂单关节 PD 扫描重力辨识。

主动关节使用 MIT 模式的 PD 跟踪绝对目标，且前馈力矩保持为零。
准静态时，PD 跟踪误差对应的估算力矩可用于拟合当前 URDF 重力模型。
joint2 到 joint5 中的其余关节保持启动或预定位姿，并实时加入重力前馈。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from tools.gravity_calibration.dm_calibration import DMCalibrationArm
except ModuleNotFoundError:
    from dm_calibration import DMCalibrationArm


LOAD_JOINTS = ("joint2", "joint3", "joint4", "joint5")
RATE = 40.0
VELOCITY_ABORT = 1.2
FREEZE_SECONDS = 5.0
RAMP_SECONDS = 2.0
RETURN_SPEED = 0.12
DATA_DIR = Path(__file__).parent / "data" / "dm"


class SafetyAbort(RuntimeError):
    pass


def build_sweep_path(start, end, speed, period):
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if period <= 0.0:
        raise ValueError("period must be positive")
    if start == end:
        raise ValueError("end must differ from the current position")

    steps = max(1, int(np.ceil(abs(end - start) / (speed * period))))
    outward = np.linspace(start, end, steps + 1)
    returning = np.linspace(end, start, steps + 1)[1:]
    direction = 1.0 if end > start else -1.0
    positions = np.concatenate((outward, returning))
    directions = np.concatenate((
        np.full(outward.shape, direction),
        np.full(returning.shape, -direction),
    ))
    return positions, directions


def parse_prepositions(specifications, active_joint, calibration):
    targets = {}
    for specification in specifications:
        parts = specification.split("=", 1)
        if len(parts) != 2:
            raise ValueError(
                f"invalid --pre {specification!r}; expected joint=value"
            )
        name = parts[0]
        if name not in LOAD_JOINTS:
            raise ValueError(f"--pre only supports {', '.join(LOAD_JOINTS)}")
        if name == active_joint:
            raise ValueError("the active joint cannot also be pre-positioned")
        target = float(parts[1])
        calibration.validate_target(name, target)
        targets[name] = target
    return targets


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run an absolute out-and-back PD gravity sweep on a DM joint.",
    )
    parser.add_argument("--joint", default="joint3", choices=LOAD_JOINTS)
    parser.add_argument(
        "--end",
        required=True,
        type=float,
        help="absolute active-joint endpoint in radians",
    )
    parser.add_argument("--speed", type=float, default=0.12, help="rad/s")
    parser.add_argument(
        "--pre",
        action="append",
        default=[],
        help="pre-position a hold joint, for example joint2=-0.7",
    )
    parser.add_argument("--tag", default="", help="output filename suffix")
    parser.add_argument(
        "--torque-fraction",
        type=float,
        default=0.5,
        help="fraction of each URDF effort limit used by the safety checks",
    )
    return parser


class PDSweepRunner:

    def __init__(self, calibration, args):
        self.calibration = calibration
        self.args = args
        self.period = 1.0 / RATE
        self.joint_names = calibration.joint_names
        self.active_index = self.joint_names.index(args.joint)
        self.prepositions = parse_prepositions(
            args.pre,
            args.joint,
            calibration,
        )
        self.start_positions = None
        self.hold_targets = None
        self.samples = []
        self.start_time = None

    def read_positions(self):
        positions = self.calibration.read_positions()
        if len(positions) != len(self.joint_names):
            raise RuntimeError(
                f"expected {len(self.joint_names)} arm positions, "
                f"received {len(positions)}"
            )
        return positions

    def gravity(self, positions):
        return self.calibration.gravity(positions)

    def send_hold_commands(self, positions, targets=None, scale=1.0):
        if targets is None:
            targets = self.hold_targets
        gravity = self.gravity(positions)
        for name in LOAD_JOINTS:
            if name == self.args.joint:
                continue
            index = self.joint_names.index(name)
            kp, kd = self.calibration.gains(name)
            torque = self.calibration.clamp_torque(
                name,
                scale * gravity[index],
            )
            self.calibration.send_mit(
                name,
                targets[index],
                0.0,
                scale * kp,
                kd,
                torque,
            )

    def send_all_to_targets(self, positions, targets, scale=1.0):
        gravity = self.gravity(positions)
        for name in LOAD_JOINTS:
            index = self.joint_names.index(name)
            kp, kd = self.calibration.gains(name)
            torque = self.calibration.clamp_torque(
                name,
                scale * gravity[index],
            )
            self.calibration.send_mit(
                name,
                targets[index],
                0.0,
                scale * kp,
                kd,
                torque,
            )

    def move_to_targets(self, starts, targets, speed):
        distance = float(np.max(np.abs(targets - starts)))
        steps = max(1, int(np.ceil(distance / (speed * self.period))))
        for fraction in np.linspace(0.0, 1.0, steps + 1):
            positions = self.read_positions()
            command = starts + fraction * (targets - starts)
            self.send_all_to_targets(positions, command)
            time.sleep(self.period)

    def preposition(self):
        if not self.prepositions:
            return
        targets = self.start_positions.copy()
        for name, target in self.prepositions.items():
            targets[self.joint_names.index(name)] = target
        print(
            "pre-positioning:",
            {
                name: round(target, 4)
                for name, target in self.prepositions.items()
            },
            flush=True,
        )
        self.move_to_targets(
            self.start_positions,
            targets,
            RETURN_SPEED,
        )
        self.hold_targets = targets

    def active_velocity(self, position, previous_position, now, previous_time):
        elapsed = max(now - previous_time, 1e-4)
        return (position - previous_position) / elapsed

    def sweep(self):
        active = self.args.joint
        kp, kd = self.calibration.gains(active)
        positions = self.read_positions()
        start = float(positions[self.active_index])
        path, directions = build_sweep_path(
            start,
            self.args.end,
            self.args.speed,
            self.period,
        )
        previous_position = start
        previous_time = time.monotonic()

        for sample_index, (target, direction) in enumerate(
            zip(path, directions)
        ):
            positions = self.read_positions()
            position = float(positions[self.active_index])
            now = time.monotonic()
            velocity = self.active_velocity(
                position,
                previous_position,
                now,
                previous_time,
            )
            if sample_index > 10 and abs(velocity) > VELOCITY_ABORT:
                raise SafetyAbort(
                    f"{active} velocity {velocity:+.3f} rad/s exceeds "
                    f"{VELOCITY_ABORT:.3f} rad/s"
                )

            torque_estimate = kp * (target - position) - kd * velocity
            torque_limit = self.calibration.torque_limit(active)
            if abs(torque_estimate) > torque_limit:
                raise SafetyAbort(
                    f"{active} estimated PD torque "
                    f"{torque_estimate:+.3f} N·m exceeds "
                    f"{torque_limit:.3f} N·m"
                )

            self.calibration.send_mit(
                active,
                target,
                0.0,
                kp,
                kd,
                0.0,
            )
            self.send_hold_commands(positions)
            self.samples.append({
                "t": round(time.monotonic() - self.start_time, 6),
                "target": round(float(target), 6),
                "q": round(position, 6),
                "velocity": round(float(velocity), 6),
                "tau_est": round(float(torque_estimate), 6),
                "q6": [round(float(value), 6) for value in positions],
                "direction": float(direction),
            })
            previous_position = position
            previous_time = now
            time.sleep(self.period)

    def freeze(self):
        positions = self.read_positions()
        targets = positions.copy()
        print(
            f"freezing for {FREEZE_SECONDS:.1f} s before returning",
            flush=True,
        )
        for _ in range(int(FREEZE_SECONDS * RATE)):
            positions = self.read_positions()
            self.send_all_to_targets(positions, targets)
            time.sleep(self.period)

    def return_and_fade(self):
        positions = self.read_positions()
        self.move_to_targets(
            positions,
            self.start_positions,
            RETURN_SPEED,
        )
        steps = max(1, int(RAMP_SECONDS * RATE))
        for scale in np.linspace(1.0, 0.0, steps):
            positions = self.read_positions()
            self.send_all_to_targets(
                positions,
                self.start_positions,
                scale=scale,
            )
            time.sleep(self.period)

    def dump(self, aborted=False, reason=""):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"_{self.args.tag}" if self.args.tag else ""
        path = DATA_DIR / (
            f"pdsweep_{self.args.joint}{tag}_{int(time.time())}.json"
        )
        document = {
            "hardware": self.calibration.rebotarm.hardware_yaml,
            "joint": self.args.joint,
            "kp": self.calibration.gains(self.args.joint)[0],
            "kd": self.calibration.gains(self.args.joint)[1],
            "end": self.args.end,
            "speed": self.args.speed,
            "prepositions": self.prepositions,
            "torque_fraction": self.calibration.torque_fraction,
            "aborted": aborted,
            "reason": reason,
            "samples": self.samples,
        }
        with open(path, "w") as output:
            json.dump(document, output, indent=2)
        print("WROTE", path, flush=True)
        return path

    def run(self):
        self.calibration.validate_target(self.args.joint, self.args.end)
        self.start_positions = self.read_positions()
        self.hold_targets = self.start_positions.copy()
        self.start_time = time.monotonic()
        print(
            f"PD SWEEP {self.args.joint}: "
            f"{self.start_positions[self.active_index]:+.4f} -> "
            f"{self.args.end:+.4f} -> start @ {self.args.speed:.3f} rad/s",
            flush=True,
        )
        print(
            "start pose:",
            np.round(self.start_positions, 4).tolist(),
            flush=True,
        )
        self.calibration.enable(LOAD_JOINTS)

        aborted = False
        reason = ""
        try:
            self.preposition()
            self.sweep()
        except (KeyboardInterrupt, SafetyAbort) as error:
            aborted = True
            reason = str(error) or "SIGINT"
            print(f"ABORT: {reason}", flush=True)
            self.freeze()
        finally:
            self.return_and_fade()
            self.dump(aborted=aborted, reason=reason)

        if aborted:
            raise SafetyAbort(reason)


def main():
    args = build_parser().parse_args()
    try:
        with DMCalibrationArm(args.torque_fraction) as calibration:
            PDSweepRunner(calibration, args).run()
    except SafetyAbort as error:
        print(f"scan stopped safely: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
