"""POS_VEL 后台：每组消费并发送最新一份命令。"""

from .CommunicationBase import CommunicationBase


class PosVelCommunicator(CommunicationBase):
    def step(self):
        hardware = self.hardware
        for name, ready in (("arm", hardware._joint_command_ready),
                            ("gripper", hardware._gripper_command_ready)):
            with hardware._command_lock:
                if not ready.is_set():
                    continue
                command = getattr(hardware.arm_state.command, name)
                position = command.position.copy()
                velocity = command.velocity.copy()
                ready.clear()
            hardware.groups[name].send_pos_vel(position, velocity)
