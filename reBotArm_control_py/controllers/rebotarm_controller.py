import threading
import time
import logging
logger = logging.getLogger("RebotArmController")
import math

import numpy as np
import pinocchio as pin

from ..actuator import RebotArm
from ..kinematics import (
    get_end_effector_frame_id,
    load_robot_model,
    pad_q_for_model,
    solve_ik,
    xyz_quat_to_se3,
    IKSolverParams
)
from .buffer_type import ControllerState
from .process_command import compute_scale_factor, EMA

class RebotArmController:

    def __init__(self, mode):
        self.rebotarm = RebotArm(mode)
        self._model = load_robot_model()
        self._data = self._model.createData()
        self._end_frame_id = get_end_effector_frame_id(self._model)

        # >>>>> 控制器占用状态 >>>>>
        self._controller_state = ControllerState.IDLE
        self._control_timestamp = 0.0
        self._control_timeout = 0.1
        self._control_lock = threading.Lock()
        # <<<<< 控制器占用状态 <<<<<

        # >>>>> IK 参数 >>>>>
        self.servoL_ik_params = IKSolverParams()
        self.servoL_ik_params.max_iter = 50
        self.servoL_ik_params.position_tolerance = 0.002
        self.servoL_ik_params.orientation_tolerance = math.radians(1.0)
        self.servoL_ik_params.step_size = 0.8
        # <<<<< IK 参数 <<<<<

        self.ema = EMA(0.2) # EMA 滤波器

        # >>>>> servoL 速度前馈状态 >>>>>
        self._servoL_last_q = None
        self._servoL_last_time = 0.0
        # <<<<< servoL 速度前馈状态 <<<<<

        if mode == "mit": # 安全保护，速度控制下一段时间不更新命令就给速度置零
            self.rebotarm.arm_state.command.velocity_timeout = self._control_timeout


    # >>>>> 公共接口 >>>>>
    def connect(self):
        """ 连接真机 """
        try:
            self.rebotarm.connect()
            time.sleep(2)
            logger.info("连接成功")
        except Exception:
            self.rebotarm.disconnect()
            raise

    def disconnect(self):
        """ 断开真机 """
        self._release_control(self._controller_state)
        self.rebotarm.disconnect()
        logger.info("断开连接")

    def set_mit_params(self, name, kp, kd):
        """ 设置 mit 的 kp, kd 参数 """
        if self.rebotarm.mode != "mit":
            raise RuntimeError("仅 MIT 模式支持设置 kp, kd 参数")
        
        for group_name, group in self.rebotarm.groups.items():
            if name not in group.joint_names:
                continue
            index = group.joint_names.index(name)
            command = getattr(self.rebotarm.arm_state.command, group_name)
            command.kp[index] = kp
            command.kd[index] = kd
            if group_name == "arm":
                self.rebotarm._joint_command_ready.set()
            elif group_name == "gripper":
                self.rebotarm._gripper_command_ready.set()
            return
        raise ValueError(f"unknown joint: {name}")

    def servoJ(self, target, speed):
        """ 透传关节指令 """
        target = np.asarray(target)
        command_state = self.rebotarm.arm_state.command

        if not self._request_control(ControllerState.SERVOJ):
            logger.warning(f"控制器正被 {self._controller_state} 占用, 拒绝 servoJ 指令")
            return

        if self.rebotarm.mode == "mit":
            command_state.timestamp = 0.0
        command_state.arm.position[:] = target
        command_state.arm.velocity[:] = speed
        self.rebotarm._joint_command_ready.set()

    def servoL(self, target, speed, lookahead = 0.02):
        """ 
        末端笛卡尔伺服
        lookahead 在 posvel 模式下无效
        """
        if self.rebotarm.mode == "mit" and (
            not np.isfinite(lookahead) or lookahead <= 0.0
        ):
            raise ValueError("lookahead must be finite and positive")
        if not self._request_control(ControllerState.SERVOL):
            logger.warning(f"控制器正被 {self._controller_state} 占用, 拒绝 servoL 指令")
            return

        target = xyz_quat_to_se3(target)
        arm_state = self.rebotarm.arm_state
        q_init = arm_state.feedback.arm.position.copy()
        ik_result = solve_ik(self._model, self._data,
            self._end_frame_id, target, q_init, self.servoL_ik_params)

        if not ik_result.success: # 如果是 mit 模式，需要紧急停速
            if self.rebotarm.mode == "mit":
                arm_state.command.timestamp = 0.0
                arm_state.command.arm.velocity[:] = 0.0
                self.rebotarm._joint_command_ready.set()
            with self._control_lock: # 锁住控制器
                self._controller_state = ControllerState.FAULT
                self._control_timestamp = 0.0
            logger.error("servoL 逆运动学求解失败")
            return

        if self.rebotarm.mode == "mit": # mit 模式使用速度前馈并限速
            command = arm_state.command
            now = time.monotonic()
            
            if now - command.timestamp >= command.velocity_timeout: # 上一条指令已过期, 丢弃滤波与前馈状态
                self.ema.reset()
                self._servoL_last_q = None

            if self._servoL_last_q is None:
                qdot_target = np.zeros_like(ik_result.q) # 首帧无差分可用
            else:
                dt = now - self._servoL_last_time
                qdot_target = (ik_result.q - self._servoL_last_q) / dt
            self._servoL_last_q = ik_result.q.copy()
            self._servoL_last_time = now

            qdot_smooth = self.ema.update(qdot_target) # 平滑速度前馈
            vel_scale = compute_scale_factor(qdot_smooth, speed) # 限速
            velocity = vel_scale * qdot_smooth
            command.timestamp = 0.0
            command.arm.position[:] = ik_result.q
            command.arm.velocity[:] = velocity
            command.timestamp = time.monotonic()
        else: # posvel 模式只限速
            command = arm_state.command.arm
            command.position[:] = ik_result.q
            command.velocity[:] = speed
        self.rebotarm._joint_command_ready.set()

    def clear_fault(self):
        """ 清除警报标志 """
        self._release_control(ControllerState.FAULT)

    def home(self, speed = 0.15, rate = 30.0, tolerance = 0.06, timeout = 30.0): 
        """ 回零 """
        arm_state = self.rebotarm.arm_state
        q_start = arm_state.feedback.arm.position.copy()
        distance = float(np.max(np.abs(q_start)))

        self.clear_fault() # 清故障
        if not self._request_control(ControllerState.HOME):
            logger.warning(f"控制器正被 {self._controller_state} 占用, 拒绝 home 指令")
            return False

        period = 1.0 / rate
        command = arm_state.command
        steps = max(1, int(distance / speed * rate))
        deadline = time.monotonic() + timeout
        # mit 模式 velocity 是速度前馈, 回零靠位置环牵引所以给 0
        # posvel 模式 velocity 是限速, 给 0 机械臂不会动
        feedforward = 0.0 if self.rebotarm.mode == "mit" else speed
        try:
            step = 0
            while True:
                step = min(step + 1, steps) # 插值走完后封顶, 自然转为保持零位
                self._request_control(ControllerState.HOME) # 续期, 防自身占用过期
                command.timestamp = 0.0
                command.arm.position[:] = q_start * (1.0 - step / steps)
                command.arm.velocity[:] = feedforward
                command.timestamp = time.monotonic()
                self.rebotarm._joint_command_ready.set()

                if float(np.max(np.abs(arm_state.feedback.arm.position))) < tolerance:
                    logger.info("回零完成")
                    return True
                if time.monotonic() >= deadline:
                    logger.warning("回零超时")
                    return False
                time.sleep(period)
        finally:
            self._release_control(ControllerState.HOME)

    def fk_curr(self):
        """ 获得当下 TCP """
        return self.fk_from_joint(self.rebotarm.arm_state.feedback.arm.position)

    def fk_from_joint(self, joint):
        """ 从关节角 FK, 返回 [x, y, z, qx, qy, qz, qw] """
        q = pad_q_for_model(self._model, np.asarray(joint, dtype=np.float64))
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)

        oMf = self._data.oMf[self._end_frame_id]
        quaternion = pin.Quaternion(oMf.rotation)

        return np.array([
            *oMf.translation,
            quaternion.x, quaternion.y, quaternion.z, quaternion.w,
        ])

    # <<<<< 公共接口 <<<<<

    # >>>>> 内部方法 >>>>>
    def _request_control(self, state):
        """ 尝试获取或刷新控制器占用权 """
        now = time.monotonic()
        with self._control_lock:
            if self._controller_state == ControllerState.FAULT:
                return False
            expired = now - self._control_timestamp >= self._control_timeout
            if self._controller_state != ControllerState.IDLE and expired:
                self._controller_state = ControllerState.IDLE
            if self._controller_state not in (ControllerState.IDLE, state):
                return False

            self._controller_state = state
            self._control_timestamp = now
            return True

    def _release_control(self, state):
        """ 释放指定控制方式持有的占用权 """
        with self._control_lock:
            if self._controller_state == state:
                self._controller_state = ControllerState.IDLE
                self._control_timestamp = 0.0
    # <<<<< 内部方法 <<<<<
