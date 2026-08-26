import logging
import time

from reBotArm_control_py.controllers import RebotArmController


MIT_KP = 0.0
MIT_KD = 4.0


def main():
    """以零刚度、速度阻尼模式运行重力补偿。"""
    controller = RebotArmController("mit")
    arm = controller.rebotarm.groups["arm"]
    original_kp = arm._mit_kp.copy() # 回零需要位置环, 先记下原始刚度
    original_kd = arm._mit_kd.copy()

    try:
        controller.connect()
        for name in arm.joint_names:
            controller.set_mit_params(name, MIT_KP, MIT_KD)

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("停止阻尼重力补偿演示")
    finally:
        time.sleep(1.0) # 等上一条指令的控制权超时释放
        # 零刚度下位置指令不产生拉力, 必须先恢复刚度才能回零
        for index, name in enumerate(arm.joint_names):
            controller.set_mit_params(name, original_kp[index], original_kd[index])
        if controller.home(): # disconnect 会掉力矩, 回零失败就不能断开
            controller.disconnect()
        else:
            logging.error("回零失败，保持通电，请手动扶住机械臂后再断开")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
