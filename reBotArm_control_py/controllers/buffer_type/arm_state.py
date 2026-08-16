from dataclasses import dataclass

import numpy as np


class ArmState():

    def __init__(self, mode, num_joints):
        buffer_types = {
            "posvel": PosVel,
        }

        self._buffer = buffer_types[mode](num_joints)
        self.command = self._buffer.command
        self.feedback = self._buffer.feedback


class PosVel():
    @dataclass
    class Command():
        position: np.ndarray
        velocity: np.ndarray

    @dataclass
    class Feedback():
        position: np.ndarray
        velocity: np.ndarray
        torque: np.ndarray
        timestamp: float

    def __init__(self, num_joints):
        self.command = self.Command(
            position=np.zeros(num_joints),
            velocity=np.zeros(num_joints),
        )
        self.feedback = self.Feedback(
            position=np.zeros(num_joints),
            velocity=np.zeros(num_joints),
            torque=np.zeros(num_joints),
            timestamp=0.0,
        )
