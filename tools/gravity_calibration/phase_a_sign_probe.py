"""Phase A — sign calibration by hand (NO torque, motors stay disabled).

Loads the current RebotArm hardware configuration and streams arm joint
positions at ~5 Hz. Move one joint at a time by hand; the delta sign vs the
physical direction fixes the motor-to-URDF sign convention.

Run: python phase_a_sign_probe.py [duration_s]
Output: prints a line whenever any joint moves >0.03 rad from its last
printed value; writes full trace to data/dm/phase_a_trace.jsonl.
"""

import json
import sys
import time
from pathlib import Path

from reBotArm_control_py.actuator import RebotArm


TRACE_PATH = Path(__file__).parent / "data" / "dm" / "phase_a_trace.jsonl"


def read_all(group):
    positions = group.get_positions()
    return {
        name: float(position)
        for name, position in zip(group.joint_names, positions)
    }


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    rebotarm = RebotArm()
    try:
        rebotarm.connect()
        group = rebotarm.arm
        start = read_all(group)
        print(
            "START pose:",
            {name: round(position, 4) for name, position in start.items()},
            flush=True,
        )
        last_printed = dict(start)

        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACE_PATH, "w") as trace:
            start_time = time.time()
            while time.time() - start_time < duration:
                positions = read_all(group)
                trace.write(json.dumps({
                    "t": round(time.time() - start_time, 3),
                    "q": positions,
                }) + "\n")
                trace.flush()
                for name, position in positions.items():
                    if abs(position - last_printed[name]) > 0.03:
                        print(
                            f"MOVE {name}: {position:+.4f} rad "
                            f"(delta vs start {position - start[name]:+.4f})",
                            flush=True,
                        )
                        last_printed[name] = position
                time.sleep(0.2)

        end = read_all(group)
        print(
            "END pose:",
            {name: round(position, 4) for name, position in end.items()},
            flush=True,
        )
    finally:
        rebotarm.disconnect()


if __name__ == "__main__":
    main()
