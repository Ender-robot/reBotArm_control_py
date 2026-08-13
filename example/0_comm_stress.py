"""通信压力测试"""

SERIAL_PORT = "/dev/ttyACM0"

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motorbridge import Controller

from reBotArm_control_py.actuator.rebotarm import load_cfg


TEST_SECONDS = 30.0
RATE = 400.0
BAUD = 921600


def percentile(values, percent):
    if not values:
        return 0.0
    return float(np.percentile(values, percent))


def main():
    cfg = load_cfg()
    joints = [joint for joint in cfg["joints"] if joint.name != "gripper"]
    controller = None
    motors = []

    loop_times = []
    send_errors = 0
    feedback_errors = 0
    empty_states = 0
    completed_loops = 0

    print(f"串口: {SERIAL_PORT}")
    print(f"电机数: {len(joints)}")
    print(f"目标频率: {RATE:.0f} Hz")
    print(f"测试时长: {TEST_SECONDS:.0f} s")
    print("测试不会主动使能电机，但会发送零刚度 MIT 帧")
    print("Ctrl+C 可提前结束")

    try:
        controller = Controller.from_dm_serial(SERIAL_PORT, BAUD)
        for joint in joints:
            motor = controller.add_damiao_motor(
                joint.motor_id,
                joint.feedback_id,
                joint.model,
            )
            motors.append(motor)

        period = 1.0 / RATE
        started = time.perf_counter()
        deadline = started

        while time.perf_counter() - started < TEST_SECONDS:
            loop_start = time.perf_counter()
            loop_times.append(loop_start)

            for motor in motors:
                try:
                    motor.request_feedback()
                except Exception:
                    feedback_errors += 1

            try:
                controller.poll_feedback_once()
            except Exception:
                feedback_errors += 1

            positions = []
            for motor in motors:
                state = motor.get_state()
                if state is None:
                    empty_states += 1
                    continue

                positions.append(state.pos)

            if len(positions) == len(motors):
                for motor, position in zip(motors, positions):
                    try:
                        motor.send_mit(position, 0.0, 0.0, 0.0, 0.0)
                    except Exception:
                        send_errors += 1

            completed_loops += 1
            deadline += period
            sleep_time = deadline - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                deadline = time.perf_counter()

    except KeyboardInterrupt:
        print("\n测试被用户提前停止。")
    finally:
        if controller is not None:
            try:
                controller.shutdown()
            except Exception:
                pass
            controller.close()

    intervals_ms = []
    for index in range(1, len(loop_times)):
        intervals_ms.append((loop_times[index] - loop_times[index - 1]) * 1000.0)

    elapsed = loop_times[-1] - loop_times[0] if len(loop_times) > 1 else 0.0
    actual_rate = (len(loop_times) - 1) / elapsed if elapsed > 0 else 0.0
    over_4_ms = sum(value > 4.0 for value in intervals_ms)
    over_10_ms = sum(value > 10.0 for value in intervals_ms)
    over_20_ms = sum(value > 20.0 for value in intervals_ms)
    print("\n========== 通信压力测试结果 ==========")
    print(f"完成循环: {completed_loops}")
    print(f"实际平均频率: {actual_rate:.1f} Hz")
    print(f"循环周期 p99: {percentile(intervals_ms, 99):.3f} ms")
    print(f"最大控制断流: {max(intervals_ms, default=0.0):.3f} ms")
    print(f"超期次数 (>4/>10/>20 ms): {over_4_ms}/{over_10_ms}/{over_20_ms}")
    print(f"空反馈次数: {empty_states}")
    print(f"反馈通信错误: {feedback_errors}")
    print(f"MIT 发送错误: {send_errors}")


if __name__ == "__main__":
    main()
