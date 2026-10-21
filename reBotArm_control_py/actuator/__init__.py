"""actuator 模块 — JointGroup 架构（分组控制，同步发送）。

硬件参数由 config/rebotarm.yaml 的 hardware_yaml 指定，软件参数由 ControlParams 管理。

示例::

    rebotarm = RebotArm("posvel")
    rebotarm.connect()
    rebotarm.submit_arm(joint_pos, joint_speed)
    rebotarm.disconnect()
"""

from .rebotarm import RebotArm, JointGroup, JointCfg, load_cfg

__all__ = [
    "RebotArm",
    "JointGroup",
    "JointCfg",
    "load_cfg",
]
