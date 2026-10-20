from dataclasses import dataclass

import numpy as np


class ArmState():

    def __init__(self, mode, num_arm_joints, num_gripper_joints):
        buffer_types = {
            "mit": Mit,
            "posvel": PosVel,
        }

        self._buffer = buffer_types[mode](
            num_arm_joints,
            num_gripper_joints,
        )
        self.command = self._buffer.command
        self.feedback = self._buffer.feedback


class PosVel():
    @dataclass
    class CommandGroup():
        position: np.ndarray
        velocity: np.ndarray

    @dataclass
    class FeedbackGroup():
        position: np.ndarray
        velocity: np.ndarray
        torque: np.ndarray

    @dataclass
    class Command():
        arm: "PosVel.CommandGroup"
        gripper: "PosVel.CommandGroup"
        timestamp: float

    @dataclass
    class Feedback():
        arm: "PosVel.FeedbackGroup"
        gripper: "PosVel.FeedbackGroup"
        timestamp: float

    def __init__(self, num_arm_joints, num_gripper_joints):
        self.command = self.Command(
            arm=self.CommandGroup(
                position=np.zeros(num_arm_joints),
                velocity=np.zeros(num_arm_joints),
            ),
            gripper=self.CommandGroup(
                position=np.zeros(num_gripper_joints),
                velocity=np.zeros(num_gripper_joints),
            ),
            timestamp=0.0,
        )
        self.feedback = self.Feedback(
            arm=self.FeedbackGroup(
                position=np.zeros(num_arm_joints),
                velocity=np.zeros(num_arm_joints),
                torque=np.zeros(num_arm_joints),
            ),
            gripper=self.FeedbackGroup(
                position=np.zeros(num_gripper_joints),
                velocity=np.zeros(num_gripper_joints),
                torque=np.zeros(num_gripper_joints),
            ),
            timestamp=0.0,
        )


class Mit():
    @dataclass
    class CommandGroup():
        position: np.ndarray
        velocity: np.ndarray
        kp: np.ndarray
        kd: np.ndarray
        torque: np.ndarray

    @dataclass
    class FeedbackGroup():
        position: np.ndarray
        velocity: np.ndarray
        torque: np.ndarray

    @dataclass
    class Command():
        arm: "Mit.CommandGroup"
        gripper: "Mit.CommandGroup"
        timestamp: float
        source: str
        sequence: int
        published_ns: int
        speed_limit: np.ndarray
        acceleration_limit: np.ndarray
        frequency_hz: float
        reset_reason: str

    @dataclass
    class Feedback():
        arm: "Mit.FeedbackGroup"
        gripper: "Mit.FeedbackGroup"
        gravity_torque: np.ndarray  # 标定后的重力补偿力矩，单位 N·m
        timestamp: float

    def __init__(self, num_arm_joints, num_gripper_joints):
        self.command = self.Command(
            arm=self.CommandGroup(
                position=np.zeros(num_arm_joints),
                velocity=np.zeros(num_arm_joints),
                kp=np.zeros(num_arm_joints),
                kd=np.zeros(num_arm_joints),
                torque=np.zeros(num_arm_joints),
            ),
            gripper=self.CommandGroup(
                position=np.zeros(num_gripper_joints),
                velocity=np.zeros(num_gripper_joints),
                kp=np.zeros(num_gripper_joints),
                kd=np.zeros(num_gripper_joints),
                torque=np.zeros(num_gripper_joints),
            ),
            timestamp=0.0,
            source="direct",
            sequence=0,
            published_ns=0,
            speed_limit=np.zeros(num_arm_joints),
            acceleration_limit=np.zeros(num_arm_joints),
            frequency_hz=0.0,
            reset_reason="initial",
        )
        self.feedback = self.Feedback(
            arm=self.FeedbackGroup(
                position=np.zeros(num_arm_joints),
                velocity=np.zeros(num_arm_joints),
                torque=np.zeros(num_arm_joints),
            ),
            gripper=self.FeedbackGroup(
                position=np.zeros(num_gripper_joints),
                velocity=np.zeros(num_gripper_joints),
                torque=np.zeros(num_gripper_joints),
            ),
            gravity_torque=np.zeros(num_arm_joints),
            timestamp=0.0,
        )
