import threading
import logging
logger = logging.getLogger("RebotArmController")

import numpy as np

from ..actuator import RebotArm
from .buffer_type import ArmState


class RebotArmController:

    def __init__(self, mode="posvel"):
        self.rebotarm = RebotArm()
        self.arm_state = ArmState(mode, self.rebotarm.num_joints)
        self._group_routes = []

        # >>>>> 后台线程 >>>>>
        self.comunicater = None # 后台通讯线程
        # <<<<< 后台线程 <<<<<

        # >>>>> 线程标志 >>>>>
        self._stop_comunicater = threading.Event()
        self._command_ready = threading.Event()
        # <<<<< 线程标志 <<<<<


    # >>>>> 公共接口 >>>>>
    def connect(self):
        """ 连接真机 """
        if self.comunicater is not None and self.comunicater.is_alive():
            logger.warning("已连接, 或通讯线程已存在")
            return

        self.rebotarm.connect()
        try:
            self._configure_mode()
            self.rebotarm.enable_all()
            self._prepare_group_routes()
            self._stop_comunicater.clear()
            self.comunicater = threading.Thread(
                target=self._communicate,
                name="comunicater",
                daemon=True,
            )
            self.comunicater.start()
            logger.info("连接成功")
        except Exception:
            self.comunicater = None
            self.rebotarm.disconnect()
            raise

    def disconnect(self):
        """ 断开真机 """
        thread = self.comunicater
        if thread is not None:
            self._stop_comunicater.set()
            thread.join()
            self.comunicater = None

        self._command_ready.clear()
        self.rebotarm.disconnect()
        logger.info("断开连接")

    def servoJ(self, target, speed):
        """ 透传关节指令 """
        target = np.asarray(target)
        command = self.arm_state.command
        if target.shape != command.position.shape:
            raise ValueError(
                f"target shape must be {command.position.shape}, "
                f"got {target.shape}"
            )

        command.position[:] = target
        command.velocity[:] = speed
        self._commit_command()
    # <<<<< 公共接口 <<<<<


    # >>>>> 内部接口 >>>>>
    def _commit_command(self):
        """ 提交最新命令快照 """
        self._command_ready.set()

    def _configure_mode(self):
        """ 配置各关节组的控制模式 """
        for group in self.rebotarm.groups.values():
            if group.num_joints == 0:
                continue
            if not group.mode_pos_vel():
                raise RuntimeError("failed to set posvel mode")

    def _prepare_group_routes(self):
        """ 建立全局关节到各分组的索引映射 """
        joint_indexes = {
            name: index
            for index, name in enumerate(self.rebotarm.joint_names)
        }
        self._group_routes = [
            (
                group,
                [joint_indexes[name] for name in group.joint_names],
            )
            for group in self.rebotarm.groups.values()
            if group.num_joints > 0
        ]
    # <<<<< 内部接口 <<<<<


    # >>>>> 后台线程 >>>>>
    def _communicate(self):
        """ 后台通讯线程 """
        while not self._stop_comunicater.is_set():
            position, velocity, torque, timestamp = (
                self.rebotarm.get_state_with_time()
            )
            feedback = self.arm_state.feedback
            feedback.position[:] = position
            feedback.velocity[:] = velocity
            feedback.torque[:] = torque
            feedback.timestamp = timestamp

            if self._stop_comunicater.is_set():
                break
            if not self._command_ready.is_set():
                continue

            self._command_ready.clear()
            command = self.arm_state.command
            group_commands = [
                (
                    group,
                    command.position[indexes],
                    command.velocity[indexes],
                )
                for group, indexes in self._group_routes
            ]
            for group, position, velocity in group_commands:
                if self._stop_comunicater.is_set():
                    break
                group.send_pos_vel(position, velocity)
    # <<<<< 后台线程 <<<<<
