"""启动位置速度控制，执行一次直线运动，输入 q 后安全回零并退出。"""

SYNC = True
X = 0.3
Y = 0.0
Z = 0.3
Roll = 0.0
Pitch = 0.0
Yaw = 0.52

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose

rebotarm = RebotArm()
ctrl = RebotArmEndPose(rebotarm, arm_control_mode="posvel")
ctrl.start()

try:
    success = ctrl.move_linear(
        x=X,
        y=Y,
        z=Z,
        roll=Roll,
        pitch=Pitch,
        yaw=Yaw,
        sync=SYNC,
        tolerance=0.001
    )
    print(f"move_linear: {'成功' if success else '失败'}")
finally:
    actual = rebotarm.arm.get_positions()
    target = ctrl._traj[-1] if ctrl._traj else ctrl._q_target
    errors = target - actual
    for index, error in enumerate(errors, start=1):
        print(f"joint{index}: {error:+.6f} rad")

while input("输入 q 并按回车，机械臂将安全回零并退出：").strip().lower() != "q":
    pass

ctrl.end()
