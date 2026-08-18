import threading
import time
import logging
logger = logging.getLogger("RebotArmController")

import numpy as np

from ..actuator import RebotArm
from ..kinematics import (
    get_end_effector_frame_id,
    load_robot_model,
    solve_clik_step,
    xyz_quat_to_se3,
)
from .buffer_type import ArmState, CLIK, ControllerState, ServoState

class RebotArmController:

    def __init__(self, mode="posvel"):
        self.rebotarm = RebotArm()
        self._arm_group = self.rebotarm.groups["arm"]
        self._gripper_group = self.rebotarm.groups["gripper"]
        self.arm_state = ArmState(
            mode,
            self._arm_group.num_joints,
            self._gripper_group.num_joints,
        )
        self._model = load_robot_model()
        self._data = self._model.createData()
        self._end_frame_id = get_end_effector_frame_id(self._model)
        self.servo_state = ServoState("clik", self._model.nv)
        self._arm_indexes = []
        self._gripper_indexes = []
        self.servo_timeout = 0.2 # 伺服超时时间

        # >>>>> 后台线程 >>>>>
        self.comunicater = None # 后台通讯线程
        self.servol_worker = None # 笛卡尔伺服线程
        # <<<<< 后台线程 <<<<<

        # >>>>> 线程标志 >>>>>
        self._stop_comunicater = threading.Event()
        self._stop_servol = threading.Event()

        self._joint_command_ready = threading.Event()
        self._gripper_command_ready = threading.Event()
        self._servol_command_ready = threading.Event()
        # <<<<< 线程标志 <<<<<

        # >>>>> 控制器占用状态 >>>>>
        self._controller_state = ControllerState.IDLE
        self._control_timestamp = 0.0
        self._control_lock = threading.Lock()
        # <<<<< 控制器占用状态 <<<<<


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
            self._stop_servol.clear()

            # 创建通讯线程
            self.comunicater = threading.Thread(
                target=self._communicate,
                name="comunicater",
                daemon=True,
            )
            # 创建 servol 线程
            self.servol_worker = threading.Thread(
                target=self._servol_control_loop,
                name="servol",
                daemon=True,
            )

            self.comunicater.start()
            self.servol_worker.start()
            time.sleep(1)

            logger.info("连接成功")
        except Exception:
            self.comunicater = None
            self.servol_worker = None
            self.rebotarm.disconnect()
            raise

    def disconnect(self):
        """ 断开真机 """
        communicater = self.comunicater
        servol_worker = self.servol_worker
        self._stop_comunicater.set()
        self._stop_servol.set()
        self._servol_command_ready.set()

        if servol_worker is not None:
            servol_worker.join()
            self.servol_worker = None
        if communicater is not None:
            communicater.join()
            self.comunicater = None

        self._joint_command_ready.clear()
        self._gripper_command_ready.clear()
        self._servol_command_ready.clear()
        self.servo_state.status.q_reference = None
        self._release_control(self._controller_state)
        self.rebotarm.disconnect()
        logger.info("断开连接")

    def servoJ(self, target, speed):
        """ 透传关节指令 """
        target = np.asarray(target)
        command = self.arm_state.command.arm
        if target.shape != command.position.shape:
            raise ValueError(
                f"target shape must be {command.position.shape}, "
                f"got {target.shape}"
            )
        if not self._request_control(ControllerState.SERVOJ):
            logger.warning(f"控制器正被 {self._controller_state} 占用, 拒绝 servoJ 指令")
            return

        command.position[:] = target
        command.velocity[:] = speed
        self._joint_command_ready.set()

    def servoL(self, target, speed, gain, lookahead = 0.1):
        """ 末端笛卡尔伺服 """
        target = xyz_quat_to_se3(target)
        gain = float(gain)
        if not np.isfinite(gain) or gain < 0.0:
            raise ValueError("gain must be finite and non-negative")
        if not self._request_control(ControllerState.SERVOL):
            logger.warning(f"控制器正被 {self._controller_state} 占用, 拒绝 servoL 指令")
            return

        self.servo_state.command = CLIK.Command(
            target=target,
            speed=speed,
            gain=gain,
            lookahead=lookahead,
            timestamp=time.monotonic(),
        )
        self._servol_command_ready.set()
    # <<<<< 公共接口 <<<<<


    # >>>>> 内部接口 >>>>>
    def _request_control(self, state):
        """ 尝试获取或刷新控制器占用权 """
        now = time.monotonic()
        with self._control_lock:
            expired = now - self._control_timestamp >= self.servo_timeout
            if self._controller_state != ControllerState.IDLE and expired:
                self._controller_state = ControllerState.IDLE
            if self._controller_state not in (ControllerState.IDLE, state):
                return False

            self._controller_state = state
            self._control_timestamp = now
            return True

    def _owns_control(self, state):
        """ 检查当前控制方式是否仍持有占用权 """
        now = time.monotonic()
        with self._control_lock:
            if self._controller_state != state:
                return False
            if now - self._control_timestamp >= self.servo_timeout:
                self._controller_state = ControllerState.IDLE
                self._control_timestamp = 0.0
                return False
            return True

    def _release_control(self, state):
        """ 释放指定控制方式持有的占用权 """
        with self._control_lock:
            if self._controller_state == state:
                self._controller_state = ControllerState.IDLE
                self._control_timestamp = 0.0

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
        self._arm_indexes = [
            joint_indexes[name]
            for name in self._arm_group.joint_names
        ]
        self._gripper_indexes = [
            joint_indexes[name]
            for name in self._gripper_group.joint_names
        ]
    # <<<<< 内部接口 <<<<<


    # >>>>> 后台线程 >>>>>
    def _servol_control_loop(self):
        """ servoL 后台控制线程 """
        period = 1.0 / self.rebotarm.servol_rate
        while not self._stop_servol.is_set():
            self._servol_command_ready.wait()
            deadline = time.monotonic()

            command = self.servo_state.command
            now = time.monotonic()
            if now - command.timestamp >= self.servo_timeout:
                self._servol_command_ready.clear()
                if self.servo_state.command is not command:
                    continue
                self.servo_state.status.q_reference = None
                self._release_control(ControllerState.SERVOL)
                break
            if not self._owns_control(ControllerState.SERVOL):
                self._servol_command_ready.clear()
                self.servo_state.status.q_reference = None
                break

            arm_state = self.arm_state
            feedback_valid = (
                arm_state.feedback.timestamp != 0.0
                and now - arm_state.feedback.timestamp < self.servo_timeout
            )
            if feedback_valid:
                q_measured = arm_state.feedback.arm.position.copy()
                q_reference = self.servo_state.status.q_reference
                if q_reference is None:
                    q_reference = q_measured.copy()
                try:
                    result = solve_clik_step(
                        self._model,
                        self._data,
                        self._end_frame_id,
                        q_measured,
                        command.target,
                        period,
                        command.speed,
                        command.gain,
                        command.lookahead,
                        q_reference=q_reference,
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    logger.warning("servoL 计算失败: %s", error)
                else:
                    arm_command = arm_state.command.arm
                    arm_command.position[:] = result.q
                    arm_command.velocity[:] = command.speed
                    self.servo_state.status = CLIK.Status(
                        error=result.error,
                        sigma_min=result.sigma_min,
                        damping=result.damping,
                        speed_scale=result.speed_scale,
                        q_reference=result.q.copy(),
                        timestamp=time.monotonic(),
                    )
                    self._joint_command_ready.set()

            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                self._stop_servol.wait(remaining)
            else:
                deadline = time.monotonic()

    def _communicate(self):
        """ 后台通讯线程 """
        while not self._stop_comunicater.is_set():
            position, velocity, torque, timestamp = (
                self.rebotarm.get_state_with_time()
            )
            feedback = self.arm_state.feedback
            feedback.arm.position[:] = position[self._arm_indexes]
            feedback.arm.velocity[:] = velocity[self._arm_indexes]
            feedback.arm.torque[:] = torque[self._arm_indexes]
            feedback.gripper.position[:] = position[self._gripper_indexes]
            feedback.gripper.velocity[:] = velocity[self._gripper_indexes]
            feedback.gripper.torque[:] = torque[self._gripper_indexes]
            feedback.timestamp = timestamp

            if self._joint_command_ready.is_set():
                self._joint_command_ready.clear()
                command = self.arm_state.command.arm
                position = command.position.copy()
                velocity = command.velocity.copy()
                self._arm_group.send_pos_vel(position, velocity)

            if self._gripper_command_ready.is_set():
                self._gripper_command_ready.clear()
                command = self.arm_state.command.gripper
                position = command.position.copy()
                velocity = command.velocity.copy()
                self._gripper_group.send_pos_vel(position, velocity)
    # <<<<< 后台线程 <<<<<
