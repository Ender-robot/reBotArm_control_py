#!/usr/bin/env python3
"""设置机械臂零点，可选第一个命令行参数指定硬件配置。"""

import sys

from reBotArm_control_py.actuator import RebotArm


def main():
    hardware_yaml = sys.argv[1] if len(sys.argv) > 1 else None
    rebotarm = RebotArm("posvel", hw_yaml=hardware_yaml)

    print(f"使用配置: {rebotarm.hardware_yaml}")
    rebotarm.connect()
    try:
        print("正在设置零点...")
        rebotarm.set_zero()
        print("零点设置完成")
    finally:
        rebotarm.disconnect()


if __name__ == "__main__":
    main()
