"""控制软件默认参数；控制器与通讯器共享同一实例。"""

from ..kinematics import IKSolverParams
from ..trajectory.mit_reference import MitReferenceParams


class ControlParams:
    def __init__(self, num_joints):
        self.command_timeout = 0.1
        self.scheduler_gap_min = 0.05
        self.scheduler_gap_periods = 3
        # servoL 专用配置；通用 IK 求解器的默认配置保持独立。
        self.ik = IKSolverParams(max_iter=50, position_tolerance=0.002, step_size=0.8)
        self.reference = MitReferenceParams(num_joints)
