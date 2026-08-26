import logging
import time

from reBotArm_control_py.controllers import RebotArmController


# 运动参数：位置单位为 rad，速度单位为 rad/s。
INTEGRATION_STEP = 0.05
JOINT_SPEED = 1.57
FINAL_INTEGRATION_TARGET = 0.785

COMMAND_PERIOD = 1.0 / 30.0


def main():
    """连接真机并以 30 Hz 发送单关节积分指令。"""
    controller = RebotArmController("posvel")

    try:
        controller.connect()

        feedback = controller.rebotarm.arm_state.feedback
        while feedback.timestamp == 0.0:
            time.sleep(COMMAND_PERIOD)

        target = feedback.arm.position.copy()
        initial_joint_position = target[0]
        integrated_position = 0.0
        step = abs(INTEGRATION_STEP)

        while True:
            if FINAL_INTEGRATION_TARGET >= 0.0:
                integrated_position = min(
                    integrated_position + step,
                    FINAL_INTEGRATION_TARGET,
                )
            else:
                integrated_position = max(
                    integrated_position - step,
                    FINAL_INTEGRATION_TARGET,
                )

            target[0] = initial_joint_position + integrated_position
            controller.servoJ(target, JOINT_SPEED)
            time.sleep(COMMAND_PERIOD)
    except KeyboardInterrupt:
        logging.info("收到 Ctrl+C，停止关节伺服演示")
    finally:
        time.sleep(1.0) # 等上一条伺服指令的控制权超时释放
        if controller.home(): # disconnect 会掉力矩, 回零失败就不能断开
            controller.disconnect()
        else:
            logging.error("回零失败，保持通电，请手动扶住机械臂后再断开")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
