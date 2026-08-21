import numpy as np
from motorbridge import Mode

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.dynamics import compute_generalized_gravity
from reBotArm_control_py.kinematics import load_robot_model


class DMCalibrationArm:

    def __init__(self, torque_fraction=0.5):
        if torque_fraction <= 0.0 or torque_fraction > 1.0:
            raise ValueError("torque_fraction must be in (0, 1]")

        self.rebotarm = RebotArm()
        self.group = self.rebotarm.groups["arm"]
        self.joint_names = self.group.joint_names
        self.model = load_robot_model()
        self.data = self.model.createData()
        self.torque_fraction = torque_fraction
        self._enabled = []
        self._configs = {
            config.name: config
            for config in self.group._jcfgs
        }

        for config in self._configs.values():
            if config.vendor != "damiao":
                raise ValueError(
                    f"{config.name} uses {config.vendor}, expected damiao"
                )

    def connect(self):
        self.rebotarm.connect()

    def close(self):
        self.disable_enabled()
        self.rebotarm.disconnect()

    def read_positions(self):
        return self.group.get_positions()

    def gains(self, name):
        config = self._configs[name]
        return config.kp, config.kd

    def position_limits(self, name):
        joint_id = self.model.getJointId(name)
        position_index = self.model.joints[joint_id].idx_q
        lower = float(self.model.lowerPositionLimit[position_index])
        upper = float(self.model.upperPositionLimit[position_index])
        return lower, upper

    def effort_limit(self, name):
        joint_id = self.model.getJointId(name)
        velocity_index = self.model.joints[joint_id].idx_v
        return float(self.model.effortLimit[velocity_index])

    def torque_limit(self, name):
        return self.torque_fraction * self.effort_limit(name)

    def clamp_torque(self, name, torque):
        limit = self.torque_limit(name)
        return float(np.clip(torque, -limit, limit))

    def validate_target(self, name, target):
        lower, upper = self.position_limits(name)
        if target < lower or target > upper:
            raise ValueError(
                f"{name} target {target} outside [{lower}, {upper}]"
            )

    def gravity(self, positions):
        return compute_generalized_gravity(
            self.model,
            positions,
            self.data,
        )[:len(self.joint_names)]

    def enable(self, names):
        for name in names:
            motor = self.rebotarm._motor_map[name]
            motor.ensure_mode(Mode.MIT, 1000)
        for name in names:
            motor = self.rebotarm._motor_map[name]
            motor.enable()
            if name not in self._enabled:
                self._enabled.append(name)

    def disable_enabled(self):
        for name in reversed(self._enabled):
            motor = self.rebotarm._motor_map.get(name)
            if motor is None:
                continue
            try:
                motor.disable()
            except Exception:
                pass
        self._enabled.clear()

    def send_mit(self, name, position, velocity, kp, kd, torque):
        self.rebotarm._motor_map[name].send_mit(
            float(position),
            float(velocity),
            float(kp),
            float(kd),
            float(torque),
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
