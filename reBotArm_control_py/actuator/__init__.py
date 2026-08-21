"""actuator 模块 — JointGroup 架构（分组控制，同步发送）。

所有参数均在 config/rebotarm.yaml 中定义，hardware_yaml 字段指定硬件配置文件。

示例::

    rebotarm = RebotArm("posvel")
    rebotarm.connect()
    rebotarm.arm_state.command.arm.position[:] = joint_pos
    rebotarm._joint_command_ready.set()
    rebotarm.disconnect()
"""

from .rebotarm import RebotArm, JointGroup, JointCfg, load_cfg

__all__ = [
    "RebotArm",
    "JointGroup",
    "JointCfg",
    "load_cfg",
]
