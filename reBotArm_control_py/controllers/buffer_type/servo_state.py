from dataclasses import dataclass

import numpy as np
import pinocchio as pin


class ServoState():

    def __init__(self, mode, num_joints):
        buffer_types = {
            "clik": CLIK,
        }

        self._buffer = buffer_types[mode](num_joints)
        self.command = self._buffer.command
        self.status = self._buffer.status


class CLIK():
    @dataclass
    class Command():
        target: pin.SE3
        speed: np.ndarray  # 六轴关节速度上限
        gain: float        # TCP 位姿误差追踪增益
        timestamp: float   # 最近一次目标更新时间
        lookahead: float   # 前馈积分步长

    @dataclass
    class Status():
        error: float        # 当前 TCP 位姿误差范数
        sigma_min: float    # 雅可比矩阵最小奇异值
        damping: float      # 本周期实际阻尼系数
        speed_scale: float  # 关节速度整体缩放比例
        Vtarget: np.ndarray  # 目标 TCP LOCAL 速度 [vx, vy, vz, wx, wy, wz]
        q_reference: np.ndarray | None  # 当前关节参考位置
        timestamp: float    # 最近一次成功计算时间

    def __init__(self, num_joints):
        self.command = self.Command(
            target=pin.SE3.Identity(),
            speed=np.zeros(num_joints),
            gain=0.0,
            timestamp=0.0,
            lookahead=0.1,
        )
        self.status = self.Status(
            error=0.0,
            sigma_min=0.0,
            damping=0.0,
            speed_scale=0.0,
            Vtarget=np.zeros(6),
            q_reference=None,
            timestamp=0.0,
        )
