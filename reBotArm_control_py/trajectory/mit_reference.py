"""MIT servoL 的关节参考模型；只计算参考，不操作硬件。"""

import math

import numpy as np


class MitReferenceParams:
    def __init__(self, num_joints):
        self.frequency_hz = 40.0
        self.max_acceleration = np.full(num_joints, 20.0)


class MitReference:
    def __init__(self, num_joints, params=None):
        self.params = MitReferenceParams(num_joints) if params is None else params
        self.position = np.zeros(num_joints)
        self.velocity = np.zeros(num_joints)
        self.acceleration = np.zeros(num_joints)

    def reset(self, position):
        self.position[:] = position
        self.velocity[:] = 0.0
        self.acceleration[:] = 0.0

    def validate(self, goal, speed, acceleration=None, frequency_hz=None):
        goal = np.asarray(goal, dtype=np.float64)
        speed = np.broadcast_to(np.asarray(speed, dtype=np.float64), self.position.shape)
        if acceleration is None:
            acceleration = self.params.max_acceleration
        acceleration = np.broadcast_to(np.asarray(acceleration, dtype=np.float64), self.position.shape)
        if frequency_hz is None:
            frequency_hz = self.params.frequency_hz
        frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
        if goal.shape != self.position.shape or not np.all(np.isfinite(goal)):
            raise ValueError("servoL goal must contain one finite position per joint")
        if not np.all(np.isfinite(speed)) or np.any(speed < 0):
            raise ValueError("servoL speed must be finite and non-negative")
        if frequency_hz.ndim != 0 or not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("reference frequency_hz must be finite and positive")
        if not np.all(np.isfinite(acceleration)) or np.any(acceleration <= 0):
            raise ValueError("reference max_acceleration must be finite and positive")
        return goal, speed, acceleration, float(frequency_hz)

    def step(self, goal, speed, dt, *, acceleration=None, frequency_hz=None):
        goal, speed, limit, frequency_hz = self.validate(goal, speed, acceleration, frequency_hz)
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("reference dt must be finite and non-negative")
        if dt == 0:
            return self.position.copy(), self.velocity.copy()
        omega = 2 * math.pi * frequency_hz
        # 子步仅用于数值稳定，不改变后台线程和总线发送频率。
        count = math.ceil(dt / min(0.001, 0.25 / omega))
        h = dt / count
        previous_velocity = self.velocity.copy()
        for _ in range(count):
            error = goal - self.position
            desired_velocity = np.sign(error) * np.minimum(
                np.minimum(omega * np.abs(error) / 2, np.sqrt(limit * np.abs(error))), speed,
            )
            acceleration = np.clip(2 * omega * (desired_velocity - self.velocity), -limit, limit)
            # 降低限速时按加速度约束减速，不瞬间裁掉已有速度。
            reachable_speed = np.maximum(speed, np.abs(self.velocity))
            velocity = np.clip(self.velocity + acceleration * h, -reachable_speed, reachable_speed)
            self.position += (self.velocity + velocity) * (h / 2)
            self.velocity[:] = velocity
        self.acceleration[:] = (self.velocity - previous_velocity) / dt
        return self.position.copy(), self.velocity.copy()
