import logging
import time

import numpy as np
import pinocchio as pin

from reBotArm_control_py.controllers import RebotArmController
from reBotArm_control_py.kinematics import (
    compute_fk,
    load_robot_model,
    pad_q_for_model,
)


# 运动参数：位置单位为 m，关节速度单位为 rad/s
MODE = "mit"
INTEGRATION_STEP = 0.003
Z_FINAL_INTEGRATION_DISTANCE = 0.15
X_FINAL_INTEGRATION_DISTANCE = 0.15
Z_HOLD_TIME = 2.0
SERVOL_SPEED = [3.0, 3.0, 6.0, 6.0, 6.0, 3.0]
SERVOL_LOOKAHEAD = 0.033
CONTROL_FREQUENCY = 30.0
COMMAND_PERIOD = 1.0 / CONTROL_FREQUENCY


def main():
    """连接真机并以 30 Hz 发送 TCP 路径点。"""
    controller = RebotArmController(MODE)
    time.sleep(5.0)

    try:
        controller.connect()

        feedback = controller.rebotarm.arm_state.feedback
        while feedback.timestamp == 0.0:
            time.sleep(COMMAND_PERIOD)

        model = load_robot_model()
        joint_position = pad_q_for_model(
            model,
            feedback.arm.position.copy(),
        )
        position, rotation, _ = compute_fk(model, joint_position)
        quaternion = pin.Quaternion(rotation).coeffs()
        target = np.concatenate((position, quaternion))
        initial_position = position.copy()
        logging.info("初始 TCP 位置: %s", initial_position)

        step = abs(INTEGRATION_STEP)
        integrated_distance = 0.0

        while integrated_distance != Z_FINAL_INTEGRATION_DISTANCE:
            if Z_FINAL_INTEGRATION_DISTANCE >= 0.0:
                integrated_distance = min(
                    integrated_distance + step,
                    Z_FINAL_INTEGRATION_DISTANCE,
                )
            else:
                integrated_distance = max(
                    integrated_distance - step,
                    Z_FINAL_INTEGRATION_DISTANCE,
                )

            target[2] = initial_position[2] + integrated_distance
            controller.servoL(
                target,
                SERVOL_SPEED,
                SERVOL_LOOKAHEAD,
            )
            time.sleep(COMMAND_PERIOD)

        hold_until = time.monotonic() + Z_HOLD_TIME
        while time.monotonic() < hold_until:
            controller.servoL(
                target,
                SERVOL_SPEED,
                SERVOL_LOOKAHEAD,
            )
            time.sleep(COMMAND_PERIOD)

        integrated_distance = 0.0
        while integrated_distance != X_FINAL_INTEGRATION_DISTANCE:
            if X_FINAL_INTEGRATION_DISTANCE >= 0.0:
                integrated_distance = min(
                    integrated_distance + step,
                    X_FINAL_INTEGRATION_DISTANCE,
                )
            else:
                integrated_distance = max(
                    integrated_distance - step,
                    X_FINAL_INTEGRATION_DISTANCE,
                )

            target[0] = initial_position[0] + integrated_distance
            controller.servoL(
                target,
                SERVOL_SPEED,
                SERVOL_LOOKAHEAD,
            )
            time.sleep(COMMAND_PERIOD)

        while True:
            controller.servoL(
                target,
                SERVOL_SPEED,
                SERVOL_LOOKAHEAD,
            )
            time.sleep(COMMAND_PERIOD)
    except KeyboardInterrupt:
        logging.info("收到 Ctrl+C，停止笛卡尔伺服演示")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
