import threading
import time
import logging

import numpy as np
import pinocchio as pin

from ..actuator import RebotArm
from ..kinematics import (
    load_robot_model,
    pad_q_for_model,
    solve_ik,
    xyz_quat_to_se3,
)
from .controller_state import ControllerState

logger = logging.getLogger("RebotArmController")


class RebotArmController:

    def __init__(self, mode):
        self.rebotarm = RebotArm(mode)
        self._model = load_robot_model(self.rebotarm.config["urdf_path"])
        self._data = self._model.createData()
        self._end_frame_id = self._model.getFrameId(self.rebotarm.config["end_effector_frame"])

        # >>>>> 控制器占用状态 >>>>>
        self._controller_state = ControllerState.IDLE
        self._control_timestamp = 0.0
        self._control_lock = threading.Lock()
        # <<<<< 控制器占用状态 <<<<<

        self.params = self.rebotarm.params
        self.servoL_ik_params = self.params.ik
        self.servoL_reference_params = self.params.reference

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
        """设置指定关节的 MIT 增益。"""
        self.rebotarm.set_mit_params(name, kp, kd)

    def servoJ(self, target, speed):
        """透传关节指令；MIT speed 为速度前馈，posvel speed 为限速。"""
        if not self._request_control(ControllerState.SERVOJ):
            logger.warning(f"控制器正被 {self._controller_state} 占用, 拒绝 servoJ 指令")
            return
        self.rebotarm.submit_arm(target, speed)

    def servoL(self, target, speed, *, acceleration=None, frequency_hz=None):
        """TCP 伺服；speed 为关节限速，加速度和频率参数只在 MIT 下使用。"""
        if not self._request_control(ControllerState.SERVOL):
            logger.warning(f"控制器正被 {self._controller_state} 占用, 拒绝 servoL 指令")
            return
        target = xyz_quat_to_se3(target)
        q_init = self.rebotarm.arm_state.feedback.arm.position.copy()
        ik_result = solve_ik(self._model, self._data, self._end_frame_id,
                             target, q_init, self.servoL_ik_params)
        if not ik_result.success:
            if self.rebotarm.mode == "mit":
                self.rebotarm.hold("ik_failure")
            with self._control_lock:
                self._controller_state = ControllerState.FAULT
                self._control_timestamp = 0.0
            logger.error("servoL 逆运动学求解失败")
            return
        if self.rebotarm.mode == "mit":
            self.rebotarm.submit_servo_target(ik_result.q, speed, acceleration, frequency_hz)
        else:
            # posvel 不读取或校验加速度与自然频率。
            self.rebotarm.submit_arm(ik_result.q, speed)

    def hold(self):
        """取消跟随，保持当前机械臂位姿；不改变夹爪或控制器故障状态。"""
        self.rebotarm.hold()

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
                self.rebotarm.submit_arm(
                    q_start * (1.0 - step / steps), feedforward, expires=True)

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
            expired = now - self._control_timestamp >= self.params.command_timeout
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
