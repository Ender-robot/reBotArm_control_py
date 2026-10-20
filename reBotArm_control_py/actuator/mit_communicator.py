"""MIT 后台：消费最新目标、推进参考并发送补偿后的电机指令。"""

import time

import numpy as np

from ..dynamics import compute_generalized_gravity, load_dynamics_model
from ..trajectory import MitReference
from .CommunicationBase import CommunicationBase


class MitCommunicator(CommunicationBase):
    def __init__(self, hardware):
        super().__init__(hardware)
        arm = hardware.groups["arm"]
        self.reference = MitReference(arm.num_joints, hardware.params.reference)
        self.model = load_dynamics_model(hardware.config["urdf_path"])
        if list(self.model.names[1:]) != arm.joint_names:
            raise ValueError("dynamics model joints must match the arm group")
        self.data = self.model.createData()
        self.step_id = 0
        self.reset("initial")

    def reset(self, reason):
        self.reference.reset(self.hardware.arm_state.feedback.arm.position)
        self.active = False
        self.initialized = False
        self.command = None
        self.last_cycle = None
        self.consumed_ns = 0
        self.dt = 0.0
        self.reason = reason
        self.position = self.reference.position.copy()
        self.velocity = self.reference.velocity.copy()
        self.target_position = self.position
        self.feedback_position = self.position
        self.sent_command = None

    def _consume(self):
        hardware = self.hardware
        with hardware._command_lock:
            self.consumed_ns = time.monotonic_ns()
            if hardware._joint_command_ready.is_set() or self.command is None:
                state = hardware.arm_state.command
                arm = state.arm
                self.command = {
                    "position": arm.position.copy(), "velocity": arm.velocity.copy(),
                    "kp": arm.kp.copy(), "kd": arm.kd.copy(), "torque": arm.torque.copy(),
                    "source": state.source, "sequence": state.sequence,
                    "published_ns": state.published_ns, "timestamp": state.timestamp,
                    "speed_limit": state.speed_limit.copy(),
                    "acceleration_limit": state.acceleration_limit.copy(),
                    "frequency_hz": state.frequency_hz, "reset_reason": state.reset_reason,
                    "revision": hardware._arm_revision,
                }
                hardware._joint_command_ready.clear()
        return self.command

    def _hold(self, reason):
        hardware = self.hardware
        position = hardware.arm_state.feedback.arm.position.copy()
        self.reference.reset(position)
        self.active = False
        command = self.command
        command.update(position=position, velocity=self.reference.velocity.copy(),
                       source="hold", timestamp=0.0, reset_reason=reason)
        # 计算期间可能已提交新目标；不能用旧周期的保持结果覆盖它。
        with hardware._command_lock:
            if hardware._arm_revision == command["revision"]:
                state = hardware.arm_state.command
                state.source = "hold"
                state.timestamp = 0.0
                state.reset_reason = reason
                state.arm.position[:] = position
                state.arm.velocity[:] = 0.0

    def prepare_command(self, now=None):
        """消费并推进一周期参考；仅返回本周期真正发送的五组参数。"""
        command = self._consume()
        if now is None:
            now = self.consumed_ns / 1e9
        dt = 0.0 if self.last_cycle is None else now - self.last_cycle
        self.last_cycle = now
        self.target_position = command["position"]
        self.feedback_position = self.hardware.arm_state.feedback.arm.position.copy()
        params = self.hardware.params
        if command["source"] == "servol":
            if now - command["timestamp"] >= params.command_timeout:
                self._hold("timeout")
            elif dt > max(params.scheduler_gap_min, params.scheduler_gap_periods * self.hardware._comm_period) or dt < 0:
                self._hold("scheduler_gap")
            else:
                if not self.active:
                    self.reference.reset(self.feedback_position)
                    self.active = True
                    dt = 0.0
                position, velocity = self.reference.step(
                    command["position"], command["speed_limit"], dt,
                    acceleration=command["acceleration_limit"], frequency_hz=command["frequency_hz"],
                )
                if np.any(position < self.model.lowerPositionLimit) or np.any(position > self.model.upperPositionLimit):
                    self._hold("joint_limit")
        else:
            if self.active or command["source"] == "hold":
                self.reference.reset(command["position"])
            self.active = False
            if command["timestamp"] and now - command["timestamp"] >= params.command_timeout:
                command["timestamp"] = 0.0
                command["velocity"][:] = 0.0
                with self.hardware._command_lock:
                    if self.hardware._arm_revision == command["revision"]:
                        self.hardware.arm_state.command.timestamp = 0.0
                        self.hardware.arm_state.command.arm.velocity[:] = 0.0

        if self.active:
            self.position = position
            self.velocity = velocity
            self.reason = "tracking"
        else:
            self.position = command["position"].copy()
            self.velocity = command["velocity"].copy()
            self.reason = command["reset_reason"] if command["source"] == "hold" else "direct"
        self.dt = dt
        self.step_id += 1
        return self.position, self.velocity, command["kp"], command["kd"], command["torque"]

    def snapshot(self):
        """按需生成诊断快照，正常发送路径不构造诊断字典。"""
        command = self.command
        if command is None:
            return {"step_id": self.step_id, "active": False, "reason": self.reason}
        return {
            "step_id": self.step_id, "sequence": command["sequence"], "source": command["source"],
            "published_ns": command["published_ns"], "consumed_ns": self.consumed_ns,
            "target_position": self.target_position.copy(),
            "position": self.position.copy(), "velocity": self.velocity.copy(),
            "acceleration": self.reference.acceleration.copy(),
            "acceleration_limit": command["acceleration_limit"].copy(),
            "frequency_hz": command["frequency_hz"],
            "feedback_position": self.feedback_position.copy(),
            "dt_s": self.dt, "active": self.active, "reason": self.reason,
        }

    def step(self):
        hardware = self.hardware
        feedback = hardware.arm_state.feedback
        if not self.initialized:
            if not hardware._feedback_valid[hardware._arm_indexes].all():
                return
            hardware.submit_arm(feedback.arm.position, 0.0)
            self.initialized = True

        position, velocity, kp, kd, torque = self.prepare_command()
        arm = hardware.groups["arm"]
        gravity = compute_generalized_gravity(self.model, feedback.arm.position + arm._gravity_offset, self.data)
        gravity_torque = arm._gravity_k * gravity
        feedback.gravity_torque[:] = gravity_torque
        compensation = gravity_torque + arm._friction * np.sign(feedback.arm.velocity) + arm._tau_bias
        values = (position, velocity, kp, kd, torque + compensation)
        arm.send_mit(*values)
        self.sent_command = (values, time.monotonic_ns(), self.step_id, self.command["sequence"])

        with hardware._command_lock:
            if not hardware._gripper_command_ready.is_set():
                return
            command = hardware.arm_state.command.gripper
            values = (command.position.copy(), command.velocity.copy(), command.kp.copy(),
                      command.kd.copy(), command.torque.copy())
            hardware._gripper_command_ready.clear()
        hardware.groups["gripper"].send_mit(*values)

    def sent_snapshot(self):
        """最后一次六轴发送函数返回时的参数；不代表硬件执行时刻。"""
        command = self.sent_command
        if command is None:
            return None
        return {"position": command[0][0].copy(), "velocity": command[0][1].copy(),
                "timestamp": command[1] / 1e9, "step_id": command[2], "sequence": command[3]}
