import logging
import time

from reBotArm_control_py.controllers import RebotArmController


MIT_KP = 0.0
MIT_KD = 4.0


def main():
    """以零刚度、速度阻尼模式运行重力补偿。"""
    controller = RebotArmController("mit")

    try:
        controller.connect()
        for name in controller.rebotarm.groups["arm"].joint_names:
            controller.set_mit_params(name, MIT_KP, MIT_KD)

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("停止阻尼重力补偿演示")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
