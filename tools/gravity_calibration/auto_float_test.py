"""DM 机械臂单关节重力补偿悬浮验证。

脚本先用 PD 将主动关节移动到绝对目标位姿，再逐步把 kp 降为零。
悬浮阶段保留 kd，并发送经过比例、零偏和常量偏置修正的 URDF 重力
前馈；慢速积分项用于观察模型剩余误差。结束或安全中止后均返回启动
姿态并逐步卸载。
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
VELOCITY_ABORT = 1.0
POSITION_WINDOW = 0.5
FADE_SECONDS = 3.0
FREEZE_SECONDS = 5.0
RAMP_SECONDS = 2.0
INTEGRAL_GAIN = 0.8
INTEGRAL_LIMIT = 2.5
DATA_DIR = Path(__file__).parent / "data" / "dm"


class SafetyAbort(RuntimeError):
    pass


def update_integral(integral, velocity, dt, ki, limit):
    return float(np.clip(
        integral - ki * velocity * dt,
        -limit,
        limit,
    ))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate DM gravity compensation in a single-joint float.",
    )
    parser.add_argument("--joint", default="joint3", choices=LOAD_JOINTS)
    parser.add_argument(
        "--pose",
        required=True,
        type=float,
        help="absolute float pose in radians",
    )
    parser.add_argument("--float-s", type=float, default=12.0)
    parser.add_argument("--speed", type=float, default=0.15, help="rad/s")
    parser.add_argument(
        "--k",
        type=float,
        default=1.0,
        help="active-joint gravity scale from fit_sweeps.py",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=0.0,
        help="active-joint constant torque bias in N·m",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="active-joint zero offset added before gravity calculation",
    )
    parser.add_argument(
        "--torque-fraction",
        type=float,
        default=0.5,
        help="fraction of each URDF effort limit used by safety checks",
    )
    return parser


class AutoFloatRunner:

    def __init__(self, calibration, args):
        self.calibration = calibration
        self.args = args
        self.period = 1.0 / RATE
        self.joint_names = calibration.joint_names
        self.active_index = self.joint_names.index(args.joint)
        self.start_positions = None
        self.samples = []
        self.start_time = None
        self.previous_position = None
        self.previous_time = None

    def read_positions(self):
        positions = self.calibration.read_positions()
        if len(positions) != len(self.joint_names):
            raise RuntimeError(
                f"expected {len(self.joint_names)} arm positions, "
                f"received {len(positions)}"
            )
        return positions

    def gravity_terms(self, positions):
        gravity = self.calibration.gravity(positions)
        adjusted = positions.copy()
        adjusted[self.active_index] += self.args.offset
        adjusted_gravity = self.calibration.gravity(adjusted)
        active_torque = (
            self.args.k * adjusted_gravity[self.active_index]
            + self.args.c
        )
        return gravity, float(active_torque)

    def require_safe_torque(self, name, torque):
        limit = self.calibration.torque_limit(name)
        if abs(torque) > limit:
            raise SafetyAbort(
                f"{name} estimated torque {torque:+.3f} N·m exceeds "
                f"{limit:.3f} N·m"
            )

    def send_pd(
        self,
        name,
        current,
        target,
        torque,
        kp_scale=1.0,
        velocity=0.0,
    ):
        kp, kd = self.calibration.gains(name)
        estimated = (
            kp_scale * kp * (target - current)
            - kd * velocity
            + torque
        )
        self.require_safe_torque(name, estimated)
        self.calibration.send_mit(
            name,
            target,
            0.0,
            kp_scale * kp,
            kd,
            torque,
        )

    def send_holds(self, positions, targets):
        gravity, _ = self.gravity_terms(positions)
        for name in LOAD_JOINTS:
            if name == self.args.joint:
                continue
            index = self.joint_names.index(name)
            torque = float(gravity[index])
            self.send_pd(
                name,
                float(positions[index]),
                float(targets[index]),
                torque,
            )

    def active_velocity(self, position):
        now = time.monotonic()
        if self.previous_time is None:
            velocity = 0.0
        else:
            elapsed = max(now - self.previous_time, 1e-4)
            velocity = (position - self.previous_position) / elapsed
        self.previous_position = position
        self.previous_time = now
        return velocity

    def move_active_to_pose(self):
        active = self.args.joint
        start = float(self.start_positions[self.active_index])
        distance = abs(self.args.pose - start)
        steps = max(
            1,
            int(np.ceil(distance / (self.args.speed * self.period))),
        )
        for sample_index, target in enumerate(
            np.linspace(start, self.args.pose, steps + 1)
        ):
            positions = self.read_positions()
            position = float(positions[self.active_index])
            velocity = self.active_velocity(position)
            if sample_index > 10 and abs(velocity) > VELOCITY_ABORT:
                raise SafetyAbort(
                    f"{active} velocity {velocity:+.3f} rad/s exceeds "
                    f"{VELOCITY_ABORT:.3f} rad/s"
                )
            _, active_torque = self.gravity_terms(positions)
            self.send_pd(
                active,
                position,
                float(target),
                active_torque,
                velocity=velocity,
            )
            self.send_holds(positions, self.start_positions)
            self.samples.append({
                "phase": "move",
                "t": round(time.monotonic() - self.start_time, 6),
                "target": round(float(target), 6),
                "q": round(position, 6),
                "velocity": round(float(velocity), 6),
                "tau_model": round(active_torque, 6),
                "q6": [round(float(value), 6) for value in positions],
            })
            time.sleep(self.period)

    def fade_to_float(self):
        steps = max(1, int(FADE_SECONDS * RATE))
        for kp_scale in np.linspace(1.0, 0.0, steps):
            positions = self.read_positions()
            position = float(positions[self.active_index])
            velocity = self.active_velocity(position)
            if abs(velocity) > VELOCITY_ABORT:
                raise SafetyAbort(
                    f"{self.args.joint} fade velocity "
                    f"{velocity:+.3f} rad/s exceeds "
                    f"{VELOCITY_ABORT:.3f} rad/s"
                )
            _, active_torque = self.gravity_terms(positions)
            self.send_pd(
                self.args.joint,
                position,
                self.args.pose,
                active_torque,
                kp_scale=kp_scale,
                velocity=velocity,
            )
            self.send_holds(positions, self.start_positions)
            self.samples.append({
                "phase": "fade",
                "t": round(time.monotonic() - self.start_time, 6),
                "q": round(position, 6),
                "velocity": round(float(velocity), 6),
                "kp_scale": round(float(kp_scale), 6),
                "tau_model": round(active_torque, 6),
            })
            time.sleep(self.period)

    def float_joint(self):
        integral = 0.0
        float_start = time.monotonic()
        positions = self.read_positions()
        reference = float(positions[self.active_index])
        _, kd = self.calibration.gains(self.args.joint)

        while time.monotonic() - float_start < self.args.float_s:
            positions = self.read_positions()
            position = float(positions[self.active_index])
            velocity = self.active_velocity(position)
            if abs(velocity) > VELOCITY_ABORT:
                raise SafetyAbort(
                    f"{self.args.joint} float velocity "
                    f"{velocity:+.3f} rad/s exceeds "
                    f"{VELOCITY_ABORT:.3f} rad/s"
                )
            if abs(position - reference) > POSITION_WINDOW:
                raise SafetyAbort(
                    f"{self.args.joint} left the float window by "
                    f"{position - reference:+.3f} rad"
                )

            _, model_torque = self.gravity_terms(positions)
            integral = update_integral(
                integral,
                velocity,
                self.period,
                INTEGRAL_GAIN,
                INTEGRAL_LIMIT,
            )
            torque = model_torque + integral
            self.require_safe_torque(self.args.joint, torque)
            self.calibration.send_mit(
                self.args.joint,
                position,
                0.0,
                0.0,
                kd,
                torque,
            )
            self.send_holds(positions, self.start_positions)
            self.samples.append({
                "phase": "float",
                "t": round(time.monotonic() - self.start_time, 6),
                "q": round(position, 6),
                "velocity": round(float(velocity), 6),
                "tau_model": round(model_torque, 6),
                "integral": round(integral, 6),
                "tau_sent": round(torque, 6),
                "q6": [round(float(value), 6) for value in positions],
            })
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
            gravity, active_torque = self.gravity_terms(positions)
            for name in LOAD_JOINTS:
                index = self.joint_names.index(name)
                torque = (
                    active_torque
                    if name == self.args.joint
                    else float(gravity[index])
                )
                kp, kd = self.calibration.gains(name)
                self.calibration.send_mit(
                    name,
                    float(targets[index]),
                    0.0,
                    kp,
                    kd,
                    self.calibration.clamp_torque(name, torque),
                )
            time.sleep(self.period)

    def return_to_start(self):
        positions = self.read_positions()
        return_start = positions.copy()
        distance = float(np.max(np.abs(self.start_positions - return_start)))
        steps = max(
            1,
            int(np.ceil(distance / (self.args.speed * self.period))),
        )
        for fraction in np.linspace(0.0, 1.0, steps + 1):
            positions = self.read_positions()
            targets = (
                return_start
                + fraction * (self.start_positions - return_start)
            )
            gravity = self.calibration.gravity(positions)
            for name in LOAD_JOINTS:
                index = self.joint_names.index(name)
                self.send_pd(
                    name,
                    float(positions[index]),
                    float(targets[index]),
                    float(gravity[index]),
                )
            time.sleep(self.period)

    def fade_at_start(self):
        steps = max(1, int(RAMP_SECONDS * RATE))
        for scale in np.linspace(1.0, 0.0, steps):
            positions = self.read_positions()
            gravity = self.calibration.gravity(positions)
            for name in LOAD_JOINTS:
                index = self.joint_names.index(name)
                kp, kd = self.calibration.gains(name)
                torque = self.calibration.clamp_torque(
                    name,
                    scale * gravity[index],
                )
                self.calibration.send_mit(
                    name,
                    float(self.start_positions[index]),
                    0.0,
                    scale * kp,
                    kd,
                    torque,
                )
            time.sleep(self.period)

    def dump(self, aborted=False, reason=""):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / (
            f"autofloat_{self.args.joint}_{int(time.time())}.json"
        )
        document = {
            "hardware": self.calibration.rebotarm.hardware_yaml,
            "joint": self.args.joint,
            "pose": self.args.pose,
            "float_seconds": self.args.float_s,
            "speed": self.args.speed,
            "k": self.args.k,
            "c": self.args.c,
            "offset": self.args.offset,
            "torque_fraction": self.calibration.torque_fraction,
            "aborted": aborted,
            "reason": reason,
            "samples": self.samples,
        }
        with open(path, "w") as output:
            json.dump(document, output, indent=2)
        print("WROTE", path, flush=True)
        return path

    def print_result(self):
        samples = [
            sample
            for sample in self.samples
            if sample["phase"] == "float"
        ]
        if not samples:
            return
        tail_count = max(1, min(len(samples), int(3.0 * RATE)))
        drift = samples[-1]["q"] - samples[0]["q"]
        residual = float(np.mean([
            sample["integral"]
            for sample in samples[-tail_count:]
        ]))
        model_torque = float(np.mean([
            sample["tau_model"]
            for sample in samples[-tail_count:]
        ]))
        print(
            f"RESULT {self.args.joint}: q={samples[-1]['q']:+.4f}, "
            f"drift={drift:+.4f} rad, "
            f"tau_model={model_torque:+.3f} N·m, "
            f"residual={residual:+.3f} N·m",
            flush=True,
        )

    def run(self):
        if self.args.float_s <= 0.0:
            raise ValueError("float-s must be positive")
        if self.args.speed <= 0.0:
            raise ValueError("speed must be positive")
        self.calibration.validate_target(self.args.joint, self.args.pose)
        self.start_positions = self.read_positions()
        self.start_time = time.monotonic()
        self.previous_position = float(
            self.start_positions[self.active_index]
        )
        self.previous_time = self.start_time
        print(
            f"AUTO FLOAT {self.args.joint}: "
            f"{self.start_positions[self.active_index]:+.4f} -> "
            f"{self.args.pose:+.4f}",
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
            self.move_active_to_pose()
            self.fade_to_float()
            self.float_joint()
            self.print_result()
        except (KeyboardInterrupt, SafetyAbort) as error:
            aborted = True
            reason = str(error) or "SIGINT"
            print(f"ABORT: {reason}", flush=True)
            self.freeze()
        finally:
            self.return_to_start()
            self.fade_at_start()
            self.dump(aborted=aborted, reason=reason)

        if aborted:
            raise SafetyAbort(reason)


def main():
    args = build_parser().parse_args()
    try:
        with DMCalibrationArm(args.torque_fraction) as calibration:
            AutoFloatRunner(calibration, args).run()
    except SafetyAbort as error:
        print(f"float test stopped safely: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
