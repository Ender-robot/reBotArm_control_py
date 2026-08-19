import json
import logging
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

from reBotArm_control_py.controllers import RebotArmController
from reBotArm_control_py.kinematics import (
    compute_fk,
    load_robot_model,
    pad_q_for_model,
)


# 运动参数：位置单位为 m，关节速度单位为 rad/s，加速度单位为 rad/s²
INTEGRATION_STEP = 0.003
Z_FINAL_INTEGRATION_DISTANCE = 0.20
X_FINAL_INTEGRATION_DISTANCE = 0.20
Z_HOLD_TIME = 1.0
SERVOL_GAIN = 18.0
SERVOL_SPEED = [3.4, 3.4, 3.4, 11.3, 11.3, 11.3]
SERVOL_ACC = [15.0, 15.0, 15.0, 20.0, 20.0, 20.0]
SERVOL_LOOKAHEAD = 0.1
CONTROL_FREQUENCY = 30.0
COMMAND_PERIOD = 1.0 / CONTROL_FREQUENCY


def _send_and_record(controller, model, target, recording):
    """发送统一的 servoL 指令并记录当前状态。"""
    controller.servoL(
        target,
        SERVOL_SPEED,
        SERVOL_ACC,
        SERVOL_GAIN,
        SERVOL_LOOKAHEAD,
    )

    feedback = controller.arm_state.feedback
    servo_status = controller.servo_state.status
    command = controller.arm_state.command.arm
    q_feedback = pad_q_for_model(
        model,
        feedback.arm.position.copy(),
    )
    tcp_position, _, _ = compute_fk(model, q_feedback)
    status_valid = servo_status.q_reference is not None

    recording.append({
        "timestamp": time.monotonic(),
        "target_tcp": target.tolist(),
        "actual_tcp_pos": tcp_position.tolist(),
        "target_joint_pos": command.position.tolist(),
        "actual_joint_pos": feedback.arm.position.tolist(),
        "actual_joint_vel": feedback.arm.velocity.tolist(),
        "clik_error": (
            float(servo_status.error) if status_valid else None
        ),
        "clik_sigma_min": (
            float(servo_status.sigma_min) if status_valid else None
        ),
        "clik_damping": (
            float(servo_status.damping) if status_valid else None
        ),
        "clik_speed_scale": (
            float(servo_status.speed_scale) if status_valid else None
        ),
        "q_reference": (
            servo_status.q_reference.tolist() if status_valid else None
        ),
    })


def _integrate_axis(
    controller,
    model,
    target,
    initial_position,
    axis,
    final_distance,
    recording,
):
    """沿指定 TCP 轴积分路径点。"""
    step = abs(INTEGRATION_STEP)
    integrated_distance = 0.0

    while integrated_distance != final_distance:
        if final_distance >= 0.0:
            integrated_distance = min(
                integrated_distance + step,
                final_distance,
            )
        else:
            integrated_distance = max(
                integrated_distance - step,
                final_distance,
            )

        target[axis] = initial_position[axis] + integrated_distance
        _send_and_record(controller, model, target, recording)
        time.sleep(COMMAND_PERIOD)


def main():
    """连接真机并以 30 Hz 发送 TCP 路径点，同时录制调试信息。"""
    controller = RebotArmController()

    # 录制数据
    recording = []
    model = load_robot_model()

    try:
        controller.connect()

        feedback = controller.arm_state.feedback
        while feedback.timestamp == 0.0:
            time.sleep(COMMAND_PERIOD)

        joint_position = pad_q_for_model(
            model,
            feedback.arm.position.copy(),
        )
        position, rotation, _ = compute_fk(model, joint_position)
        quaternion = pin.Quaternion(rotation).coeffs()
        target = np.concatenate((position, quaternion))
        initial_position = position.copy()
        logging.info("初始 TCP 位置: %s", initial_position)

        _integrate_axis(
            controller,
            model,
            target,
            initial_position,
            2,
            Z_FINAL_INTEGRATION_DISTANCE,
            recording,
        )
        hold_until = time.monotonic() + Z_HOLD_TIME
        while time.monotonic() < hold_until:
            _send_and_record(controller, model, target, recording)
            time.sleep(COMMAND_PERIOD)
        _integrate_axis(
            controller,
            model,
            target,
            initial_position,
            0,
            X_FINAL_INTEGRATION_DISTANCE,
            recording,
        )

        # 保持最终位置，继续录制直到Ctrl+C
        logging.info("到达目标位置，保持不动并继续录制...（按Ctrl+C停止）")
        while True:
            _send_and_record(controller, model, target, recording)
            time.sleep(COMMAND_PERIOD)

    except KeyboardInterrupt:
        logging.info("收到 Ctrl+C，停止笛卡尔伺服演示")
    finally:
        controller.disconnect()

        # 保存录制数据
        output_file = Path(__file__).parent / f"servoL_recording_gain{SERVOL_GAIN}_lookahead{SERVOL_LOOKAHEAD}_acc{SERVOL_ACC}_step{INTEGRATION_STEP}.json"
        with open(output_file, "w") as f:
            json.dump(recording, f, indent=2)
        logging.info("调试数据已保存到: %s", output_file)
        logging.info("共录制 %d 条记录", len(recording))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
