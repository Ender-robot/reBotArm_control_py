"""reBotArm 分组控制系统 — JointGroup 架构。

配置驱动的硬件抽象层：
  - config/rebotarm.yaml 的 hardware_yaml 指定硬件配置；软件默认值见 controllers/control_params.py
  - 关节按 groups 分组，每组独立控制模式

使用示例::

    # 所有分组使用 POS_VEL 模式
    arm = RebotArm("posvel")
    arm.connect()
    arm.submit_arm(joint_pos, joint_speed)

    arm.disconnect()
"""
from __future__ import annotations

import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from motorbridge import Controller, Mode, CallError

from ..kinematics.robot_model import DEFAULT_END_EFFECTOR_FRAME
from .arm_state import ArmState
from .mit_communicator import MitCommunicator
from .posvel_communicator import PosVelCommunicator

_CFG_DIR = Path(__file__).parent.parent.parent / "config"
_GLOBAL_CFG = _CFG_DIR / "rebotarm.yaml"


def _resolve_hw_cfg_path(hw_yaml: str | None = None) -> Path:
    if hw_yaml is None:
        if not _GLOBAL_CFG.exists():
            raise FileNotFoundError(f"{_GLOBAL_CFG} not found")
        data = yaml.safe_load(_GLOBAL_CFG.read_text())
        hw_yaml = data.get("hardware_yaml") if data else None
        if not hw_yaml:
            raise ValueError("hardware_yaml not set in rebotarm.yaml")

    p = Path(hw_yaml)
    if p.is_absolute():
        return p
    path = _CFG_DIR / hw_yaml
    if path.exists():
        return path
    raise FileNotFoundError(f"hardware config not found: {path}")


# --------------------------------------------------------------------------
# 配置加载
# --------------------------------------------------------------------------

@dataclass
class JointCfg:
    name: str
    motor_id: int
    feedback_id: int
    model: str = "4340P"
    vendor: str = "damiao"
    kp: float = 0.0
    kd: float = 0.0
    vel_kp: float = 0.0
    vel_ki: float = 0.0
    pos_kp: float = 0.0
    pos_ki: float = 0.0
    vlim: float = 2.0
    offset: float = 0.0
    gravity_k: float = 0.0
    tau_bias: float = 0.0
    friction: float = 0.0


def load_cfg(hw_yaml: str | None = None) -> dict:
    hw_path = _resolve_hw_cfg_path(hw_yaml)

    with open(hw_path, "r") as f:
        data = yaml.safe_load(f)

    joints = []
    for j in data.get("joints", []):
        mc = j.get("MIT", {})
        pc = j.get("POS_VEL", {})
        calibration = j.get("calibration", {})
        values = {"name": j["name"], "motor_id": int(j["motor_id"]),
                  "feedback_id": int(j["feedback_id"])}
        for key in ("model", "vendor"):
            if key in j:
                values[key] = str(j[key])
        if "vendor" in values:
            values["vendor"] = values["vendor"].lower()
        for section, keys in ((mc, ("kp", "kd")),
                              (pc, ("vel_kp", "vel_ki", "pos_kp", "pos_ki", "vlim")),
                              (calibration, ("offset", "gravity_k", "tau_bias", "friction"))):
            for key in keys:
                if key in section:
                    values[key] = float(section[key])
        joints.append(JointCfg(**values))

    return {
        "hardware_path": str(hw_path.resolve()),
        "name": data.get("name", "reBotArm"),
        "channel": data.get("channel", "/dev/ttyACM0"),
        "comm_rate": float(data.get("comm_rate", 350.0)),
        "urdf_path": data.get("urdf_path"),
        "end_effector_frame": data.get("end_effector_frame", DEFAULT_END_EFFECTOR_FRAME),
        "groups": data.get("groups", {}),
        "joints": joints,
    }


# --------------------------------------------------------------------------
# NoOpGroup — 无执行器时的空操作桩
# --------------------------------------------------------------------------

class NoOpGroup:
    """当配置中不存在 gripper 组时的空实现。

    所有属性和方法与 JointGroup 接口兼容，但不对电机发送任何指令，
    方便用户代码在有/无夹爪时共用同一套逻辑，无需条件判断。
    """

    name: str = "gripper"
    _mode: str = "mit"

    @property
    def num_joints(self) -> int:
        return 0

    @property
    def joint_names(self) -> List[str]:
        return []

    @property
    def mode(self) -> str:
        return "mit"

    def enable(self) -> None:
        pass

    def disable(self) -> None:
        pass

    def mode_mit(self, kp=None, kd=None) -> bool:
        self._mode = "mit"
        return True

    def mode_pos_vel(self, vlim=None) -> bool:
        self._mode = "pos_vel"
        return True

    def mode_vel(self) -> bool:
        self._mode = "vel"
        return True

    def send_mit(self, pos, vel=None, kp=None, kd=None, tau=None) -> None:
        pass

    def send_pos_vel(self, pos, vlim=None) -> None:
        pass

    def send_vel(self, vel) -> None:
        pass

    def get_positions(self) -> np.ndarray:
        return np.array([], dtype=np.float64)

    def get_velocities(self) -> np.ndarray:
        return np.array([], dtype=np.float64)

    def __repr__(self) -> str:
        return "NoOpGroup(gripper, no actuator)"


# --------------------------------------------------------------------------
# JointGroup — 单组关节控制
# --------------------------------------------------------------------------

class JointGroup:
    """一组关节的独立控制器。

    每组拥有独立的控制模式（MIT / POS_VEL）、PID 参数和电机列表，
    可单独使能、切换模式、发送命令。

    由 RebotArm 通过 __getattr__ 代理访问，例如 arm.arm / arm.gripper。
    组内关节数量、顺序由配置决定。
    """

    def __init__(
        self,
        name: str,
        joint_names: List[str],
        all_joints: List[JointCfg],
        motor_map: Dict[str, any],
        ctrl_map: Dict[str, Controller],
        set_mit_gains=None,
    ) -> None:
        self.name = name
        self._jn: List[str] = joint_names
        self._jcfgs: List[JointCfg] = [
            next(j for j in all_joints if j.name == n) for n in joint_names
        ]
        self._mm: Dict[str, any] = motor_map
        self._cm: Dict[str, Controller] = ctrl_map
        self._set_mit_gains = set_mit_gains
        self._mode: str = "mit"
        self._mit_kp: np.ndarray = np.array([j.kp for j in self._jcfgs], dtype=np.float64)
        self._mit_kd: np.ndarray = np.array([j.kd for j in self._jcfgs], dtype=np.float64)
        self._pv_vlim: np.ndarray = np.array([j.vlim for j in self._jcfgs], dtype=np.float64)
        self._gravity_offset = np.array([j.offset for j in self._jcfgs])
        self._gravity_k = np.array([j.gravity_k for j in self._jcfgs])
        self._tau_bias = np.array([j.tau_bias for j in self._jcfgs])
        self._friction = np.array([j.friction for j in self._jcfgs])

    # ── 属性 ────────────────────────────────────────────────────────────

    @property
    def num_joints(self) -> int:
        return len(self._jn)

    @property
    def joint_names(self) -> List[str]:
        return list(self._jn)

    @property
    def mode(self) -> str:
        return self._mode

    # ── 使能 / 失能 ────────────────────────────────────────────────────

    def enable(self) -> None:
        by_vendor: Dict[str, List[str]] = {}
        for jc in self._jcfgs:
            by_vendor.setdefault(jc.vendor, []).append(jc.name)
        for vendor in by_vendor:
            try:
                self._cm[vendor].enable_all()
            except CallError as e:
                print(f"[{self.name}/enable] {e}")
            time.sleep(0.05)

    def disable(self) -> None:
        by_vendor: Dict[str, List[str]] = {}
        for jc in self._jcfgs:
            by_vendor.setdefault(jc.vendor, []).append(jc.name)
        for vendor in by_vendor:
            try:
                self._cm[vendor].disable_all()
            except CallError as e:
                print(f"[{self.name}/disable] {e}")
            time.sleep(0.05)

    # ── 模式切换 ────────────────────────────────────────────────────────

    def _write_pv_params(self, jc: JointCfg, vlim) -> None:
        m = self._mm[jc.name]
        try:
            if jc.vendor == "robstride":
                m.robstride_write_param_f32(0x7017, vlim)
                time.sleep(0.01)
                m.robstride_write_param_f32(0x701F, jc.vel_kp)
                time.sleep(0.01)
                m.robstride_write_param_f32(0x7020, jc.vel_ki)
                time.sleep(0.01)
                m.robstride_write_param_f32(0x701E, jc.pos_kp)
            elif jc.vendor == "damiao":
                m.write_register_f32(25, jc.vel_kp)
                m.write_register_f32(26, jc.vel_ki)
                m.write_register_f32(27, jc.pos_kp)
                m.write_register_f32(28, jc.pos_ki)
            time.sleep(0.02)
        except Exception as e:
            print(f"[{self.name}/pv_params/{jc.name}] {e}")

    def mode_mit(
        self,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> bool:
        self._mode = "mit"
        if kp is not None or kd is not None:
            if self._set_mit_gains is not None:
                self._set_mit_gains(self.name, kp, kd)
            else:
                if kp is not None:
                    self._mit_kp[:] = np.asarray(kp, dtype=np.float64).reshape(-1)
                if kd is not None:
                    self._mit_kd[:] = np.asarray(kd, dtype=np.float64).reshape(-1)
        ok = True
        for jc in self._jcfgs:
            try:
                self._mm[jc.name].ensure_mode(Mode.MIT, 1000)
            except CallError as e:
                print(f"[{self.name}/mode_mit/{jc.name}] {e}")
                ok = False
            time.sleep(0.05)
        time.sleep(0.2)
        return ok

    def mode_pos_vel(
        self,
        vlim: Optional[np.ndarray] = None,
    ) -> bool:
        self._mode = "pos_vel"
        if vlim is not None:
            self._pv_vlim[:] = np.asarray(vlim, dtype=np.float64).reshape(-1)
        ok = True
        for index, jc in enumerate(self._jcfgs):
            self._write_pv_params(jc, float(self._pv_vlim[index]))
            try:
                self._mm[jc.name].ensure_mode(Mode.POS_VEL, 1000)
            except CallError as e:
                print(f"[{self.name}/mode_pos_vel/{jc.name}] {e}")
                ok = False
            time.sleep(0.05)
        time.sleep(0.2)
        return ok

    def mode_vel(self) -> bool:
        self._mode = "vel"
        ok = True
        for jc in self._jcfgs:
            try:
                self._mm[jc.name].ensure_mode(Mode.VEL, 1000)
            except CallError as e:
                print(f"[{self.name}/mode_vel/{jc.name}] {e}")
                ok = False
            time.sleep(0.05)
        time.sleep(0.2)
        return ok

    # ── MIT 发送 ────────────────────────────────────────────────────────

    def send_mit(
        self,
        pos: np.ndarray,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
    ) -> None:
        n = self.num_joints
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        if vel is None:
            vel = np.zeros(n)
        if tau is None:
            tau = np.zeros(n)
        if kp is None:
            kp = self._mit_kp
        if kd is None:
            kd = self._mit_kd

        for i, jc in enumerate(self._jcfgs):
            try:
                self._mm[jc.name].send_mit(
                    float(pos[i]),
                    float(vel[i]),
                    float(kp[i]),
                    float(kd[i]),
                    float(tau[i]),
                )
            except CallError:
                pass

    # ── POS_VEL 发送 ───────────────────────────────────────────────────

    def send_pos_vel(
        self,
        pos: np.ndarray,
        vlim: Optional[np.ndarray] = None,
    ) -> None:
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        if vlim is None:
            vlim = self._pv_vlim
        vlim = np.asarray(vlim, dtype=np.float64).reshape(-1)
        for i in range(min(len(pos), len(vlim))):
            try:
                self._mm[self._jcfgs[i].name].send_pos_vel(
                    float(pos[i]),
                    float(vlim[i]),
                )
            except CallError:
                pass

    # ── VEL 发送 ───────────────────────────────────────────────────────

    def send_vel(self, vel: np.ndarray) -> None:
        vel = np.asarray(vel, dtype=np.float64).reshape(-1)
        for i in range(min(len(vel), self.num_joints)):
            try:
                self._mm[self._jcfgs[i].name].send_vel(float(vel[i]))
            except CallError:
                pass

    # ── 状态读取 ───────────────────────────────────────────────────────

    def _poll_feedback(self) -> None:
        """仅处理 CAN 接收队列中的反馈帧（快速，无总线发送）。"""
        seen: set[str] = set()
        for jc in self._jcfgs:
            if jc.vendor not in seen:
                seen.add(jc.vendor)
                try:
                    self._cm[jc.vendor].poll_feedback_once()
                except Exception:
                    pass

    def _request_feedback(self) -> None:
        """发送显式反馈请求帧 + 处理接收队列（慢，有总线发送）。"""
        for jc in self._jcfgs:
            try:
                self._mm[jc.name].request_feedback()
            except Exception:
                pass
        self._poll_feedback()

    def get_positions(self, request_feedback: bool = True) -> np.ndarray:
        # 始终发送显式请求帧 + 处理接收队列
        # motorbridge 内部会针对 RS/DM 分别处理
        if request_feedback: self._request_feedback()
        
        out: list[float] = []
        for jc in self._jcfgs:
            m = self._mm[jc.name]
            st = m.get_state()
            if st is not None:
                out.append(st.pos)
            else:
                # 缓存为空时回退到 SDO 读取（安全兜底）
                if jc.vendor == "robstride":
                    try:
                        out.append(float(m.robstride_get_param_f32(0x7019)))
                        continue
                    except CallError:
                        pass
                out.append(0.0)
        return np.array(out, dtype=np.float64)

    def get_velocities(self, request_feedback: bool = True) -> np.ndarray:
        # NOTE (RobStride): the cached state has the same staleness problem as
        # get_positions, and the mechVel param (0x701A) was measured NOT to be
        # rad/s on RS firmware (inconsistent scale/sign vs dq/dt, 2026-07-17).
        # For a live velocity on RobStride, finite-difference get_positions().
        if request_feedback:
            self._request_feedback()
        return np.array([
            self._mm[jc.name].get_state().vel
            if self._mm[jc.name].get_state() is not None else 0.0
            for jc in self._jcfgs
        ], dtype=np.float64)

    def __repr__(self) -> str:
        return f"JointGroup({self.name!r}, joints={self.num_joints}, mode={self._mode})"


# --------------------------------------------------------------------------
# RebotArm — 分组控制器容器
# --------------------------------------------------------------------------

class RebotArm:
    """reBotArm 分组控制系统。

    持有多个 JointGroup，每组独立控制模式，独立发送命令，
    在同一个控制循环中按组顺序同步发送，防止总线争用。

    按组访问（通过 __getattr__）::

        arm.arm       # 机械臂关节组
        arm.gripper   # 夹爪关节组（如果有）

    也可以通过 groups 字典::

        arm.groups["arm"]
        arm.groups["gripper"]

    手动添加组::

        arm.add_group("custom", ["joint1", "joint2"])
    """

    def __init__(self, mode, hw_yaml: str | None = None) -> None:
        from ..controllers.control_params import ControlParams

        self.config = load_cfg(hw_yaml)
        cfg = self.config
        self._hardware_path = Path(cfg["hardware_path"])
        self._hw_yaml = self._hardware_path.name
        self.mode = mode
        self._name = cfg["name"]
        self._channel = cfg["channel"]
        self._comm_period = 1.0 / cfg["comm_rate"]
        self._all_joints = cfg["joints"]
        self._groups_def = cfg["groups"]
        self._ctrl_map = {}
        self._motor_map = {}
        self._groups = {}
        self._connected = False
        self._build_groups()
        self.arm_state = ArmState(mode, self.arm.num_joints, self.gripper.num_joints)
        self.params = ControlParams(self.arm.num_joints)
        self._command_lock = threading.Lock()
        self._joint_command_ready = threading.Event()
        self._gripper_command_ready = threading.Event()
        self._arm_revision = 0
        self._feedback_valid = np.zeros(self.num_joints, dtype=bool)
        indexes = {name: index for index, name in enumerate(self.joint_names)}
        self._arm_indexes = [indexes[name] for name in self.arm.joint_names]
        self._gripper_indexes = [indexes[name] for name in self.gripper.joint_names]
        if mode == "mit":
            for name in ("arm", "gripper"):
                group = self._groups[name]
                if group.num_joints:
                    command = getattr(self.arm_state.command, name)
                    # 配置只初始化一次；组接口和命令发送引用同一份运行增益。
                    command.kp = group._mit_kp
                    command.kd = group._mit_kd
        communicator_type = {"mit": MitCommunicator, "posvel": PosVelCommunicator}[mode]
        self.communicator = communicator_type(self)

    def connect(self) -> None:
        """连接总线并启动本模式的唯一后台通讯器。"""
        if self._connected:
            return
        try:
            self._setup_motors()
        except Exception:
            self._close_controllers()
            raise
        self._connected = True
        try:
            for group in self._groups.values():
                if not group.num_joints:
                    continue
                success = group.mode_mit() if self.mode == "mit" else group.mode_pos_vel()
                if not success:
                    raise RuntimeError(f"failed to set {self.mode} mode")
            self._clear_commands()
            self._feedback_valid[:] = False
            self.communicator.reset("connect")
            self.enable_all()
            self.communicator.start()
        except Exception:
            self.disconnect()
            raise

    def _make_controller(self) -> Controller:
        if self._channel.startswith("/dev/tty"):
            return Controller.from_dm_serial(self._channel, 921600)
        return Controller(self._channel)

    def _setup_motors(self) -> None:
        for jc in self._all_joints:
            vendor = jc.vendor
            if vendor not in self._ctrl_map:
                self._ctrl_map[vendor] = self._make_controller()
            ctrl = self._ctrl_map[vendor]

            if vendor == "damiao":
                mot = ctrl.add_damiao_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif vendor == "robstride":
                mot = ctrl.add_robstride_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif vendor == "myactuator":
                mot = ctrl.add_myactuator_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif vendor == "hightorque":
                mot = ctrl.add_hightorque_motor(jc.motor_id, jc.feedback_id, jc.model)
            else:
                raise ValueError(f"Unsupported vendor: {vendor}")

            self._motor_map[jc.name] = mot

    def _build_groups(self) -> None:
        for gname, gdef in self._groups_def.items():
            joints_def = gdef.get("joints", [])
            g = JointGroup(
                name=gname,
                joint_names=joints_def,
                all_joints=self._all_joints,
                motor_map=self._motor_map,
                ctrl_map=self._ctrl_map,
                set_mit_gains=self._set_group_mit_gains,
            )
            self._groups[gname] = g
        if "gripper" not in self._groups:
            self._groups["gripper"] = NoOpGroup()

    # ── 属性 ────────────────────────────────────────────────────────────

    @property
    def num_joints(self) -> int:
        return len(self._all_joints)

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self._all_joints]

    @property
    def groups(self) -> Dict[str, JointGroup]:
        return self._groups

    @property
    def has_gripper(self) -> bool:
        return not isinstance(self._groups.get("gripper", None), NoOpGroup)

    @property
    def hardware_yaml(self) -> str:
        return self._hw_yaml

    def __getattr__(self, name: str) -> any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._groups:
            return self._groups[name]
        raise AttributeError(name)

    # ── 手动添加组 ────────────────────────────────────────────────────

    def add_group(self, name: str, joint_names: List[str]) -> JointGroup:
        if name in self._groups:
            raise ValueError(f"组 {name!r} 已存在")
        g = JointGroup(
            name=name,
            joint_names=joint_names,
            all_joints=self._all_joints,
            motor_map=self._motor_map,
            ctrl_map=self._ctrl_map,
            set_mit_gains=self._set_group_mit_gains,
        )
        self._groups[name] = g
        return g

    # ── 全局使能 / 失能 ────────────────────────────────────────────────

    def enable_all(self) -> None:
        for ctrl in self._ctrl_map.values():
            ctrl.enable_all()
            time.sleep(0.05)

    def disable_all(self) -> None:
        for ctrl in self._ctrl_map.values():
            ctrl.disable_all()
            time.sleep(0.05)

    # ── 零点 ────────────────────────────────────────────────────────────

    def set_zero(self, poll_max: int = 200, poll_interval: float = 0.05) -> None:
        self.disable_all()
        time.sleep(0.3)
        for jc in self._all_joints:
            for _ in range(poll_max):
                for m in self._motor_map.values():
                    try:
                        m.request_feedback()
                    except Exception:
                        pass
                for ctrl in self._ctrl_map.values():
                    try:
                        ctrl.poll_feedback_once()
                    except Exception:
                        pass
                st = self._motor_map[jc.name].get_state()
                if st is not None and st.status_code == 0:
                    break
                time.sleep(poll_interval)
            try:
                self._motor_map[jc.name].set_zero_position()
            except CallError as e:
                print(f"[set_zero] {jc.name}: {e}")
            time.sleep(0.1)

    # ── 全局状态读取 ───────────────────────────────────────────────────

    def get_state(
        self,
        request_feedback: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if request_feedback:
            for m in self._motor_map.values():
                try:
                    m.request_feedback()
                except Exception:
                    pass
        for ctrl in self._ctrl_map.values():
            try:
                ctrl.poll_feedback_once()
            except Exception:
                pass
        pos, vel, torq = [], [], []
        for index, jc in enumerate(self._all_joints):
            st = self._motor_map[jc.name].get_state()
            self._feedback_valid[index] = st is not None
            if st is not None:
                pos.append(st.pos)
                vel.append(st.vel)
                torq.append(st.torq)
            else:
                pos.append(0.0)
                vel.append(0.0)
                torq.append(0.0)
        return (
            np.array(pos, dtype=np.float64),
            np.array(vel, dtype=np.float64),
            np.array(torq, dtype=np.float64),
        )

    def get_state_with_time(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """返回机械臂状态和软件单调时间戳，不是硬件时间戳。"""
        pos, vel, torq = self.get_state()
        timestamp = time.monotonic()
        return pos, vel, torq, timestamp

    def get_positions(self) -> np.ndarray:
        return self.get_state()[0]

    def get_velocities(self) -> np.ndarray:
        return self.get_state()[1]

    def get_torques(self) -> np.ndarray:
        return self.get_state()[2]

    # ── 命令交接 ────────────────────────────────────────────────────────

    def _joint_values(self, value, count, name):
        values = np.broadcast_to(np.asarray(value, dtype=np.float64), (count,))
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite joint values")
        return values

    def submit_arm(self, position, velocity, *, expires=False):
        """原子提交关节直通命令；velocity 在 MIT 为前馈，在 posvel 为限速。"""
        count = self.arm.num_joints
        position = self._joint_values(position, count, "position")
        velocity = self._joint_values(velocity, count, "velocity")
        with self._command_lock:
            state = self.arm_state.command
            state.arm.position[:] = position
            state.arm.velocity[:] = velocity
            state.timestamp = time.monotonic() if expires else 0.0
            if self.mode == "mit":
                state.source = "direct"
                state.sequence += 1
                state.published_ns = time.monotonic_ns()
            self._arm_revision += 1
            self._joint_command_ready.set()

    def submit_servo_target(self, position, speed, acceleration, frequency_hz):
        """提交完整 MIT servoL 目标，返回序号与单调发布时间。"""
        if self.mode != "mit":
            raise RuntimeError("servo targets require MIT mode")
        position, speed, acceleration, frequency_hz = self.communicator.reference.validate(
            position, speed, acceleration, frequency_hz)
        with self._command_lock:
            state = self.arm_state.command
            state.arm.position[:] = position
            state.arm.velocity[:] = 0.0
            state.speed_limit[:] = speed
            state.acceleration_limit[:] = acceleration
            state.frequency_hz = frequency_hz
            state.source = "servol"
            state.sequence += 1
            state.published_ns = time.monotonic_ns()
            state.timestamp = state.published_ns / 1e9
            self._arm_revision += 1
            self._joint_command_ready.set()
            return state.sequence, state.published_ns

    def hold(self, reason="hold"):
        """按当前反馈保持机械臂，不改变增益或夹爪命令。"""
        with self._command_lock:
            state = self.arm_state.command
            state.arm.position[:] = self.arm_state.feedback.arm.position
            state.arm.velocity[:] = 0.0
            state.timestamp = 0.0
            if self.mode == "mit":
                state.source = "hold"
                state.reset_reason = reason
            self._arm_revision += 1
            self._joint_command_ready.set()

    def submit_gripper(self, position, speed=None):
        """提交夹爪目标；MIT 使用位置控制，posvel 默认使用配置限速。"""
        count = self.gripper.num_joints
        position = self._joint_values(position, count, "gripper position")
        if self.mode == "posvel":
            if speed is None:
                speed = self.gripper._pv_vlim if count else np.empty(0)
            speed = self._joint_values(speed, count, "gripper speed")
        with self._command_lock:
            command = self.arm_state.command.gripper
            command.position[:] = position
            if self.mode == "mit":
                command.velocity[:] = 0.0
                command.torque[:] = 0.0
            else:
                command.velocity[:] = speed
            self._gripper_command_ready.set()

    def set_mit_params(self, name, kp, kd):
        if self.mode != "mit":
            raise RuntimeError("仅 MIT 模式支持设置 kp, kd 参数")
        if not np.isfinite(kp) or not np.isfinite(kd) or kp < 0 or kd < 0:
            raise ValueError("kp and kd must be finite and non-negative")
        for group_name, group in self._groups.items():
            names = group.joint_names
            if name not in names:
                continue
            index = names.index(name)
            with self._command_lock:
                command = getattr(self.arm_state.command, group_name)
                command.kp[index] = kp
                command.kd[index] = kd
                if group_name == "arm":
                    self._joint_command_ready.set()
                else:
                    self._gripper_command_ready.set()
            return
        raise ValueError(f"unknown joint: {name}")

    def _set_group_mit_gains(self, name, kp, kd):
        group = self._groups[name]
        if kp is not None:
            kp = self._joint_values(kp, group.num_joints, "kp")
        if kd is not None:
            kd = self._joint_values(kd, group.num_joints, "kd")
        with self._command_lock:
            if kp is not None:
                group._mit_kp[:] = kp
            if kd is not None:
                group._mit_kd[:] = kd
            if self.mode == "mit":
                if name == "arm":
                    self._joint_command_ready.set()
                elif name == "gripper":
                    self._gripper_command_ready.set()

    def update_feedback(self):
        position, velocity, torque, timestamp = self.get_state_with_time()
        with self._command_lock:
            feedback = self.arm_state.feedback
            for name, indexes in (("arm", self._arm_indexes), ("gripper", self._gripper_indexes)):
                group = getattr(feedback, name)
                group.position[:] = position[indexes]
                group.velocity[:] = velocity[indexes]
                group.torque[:] = torque[indexes]
            feedback.timestamp = timestamp

    def _clear_commands(self):
        with self._command_lock:
            command = self.arm_state.command
            command.timestamp = 0.0
            command.arm.position[:] = self.arm_state.feedback.arm.position
            command.arm.velocity[:] = 0.0
            command.gripper.position[:] = self.arm_state.feedback.gripper.position
            command.gripper.velocity[:] = 0.0
            if self.mode == "mit":
                command.source = "hold"
                command.arm.torque[:] = 0.0
                command.gripper.torque[:] = 0.0
            self._arm_revision += 1
            self._joint_command_ready.clear()
            self._gripper_command_ready.clear()

    # ── 生命周期 ────────────────────────────────────────────────────────

    def _close_controllers(self):
        try:
            # 即使一个控制器关闭失败，仍关闭其余资源，并保留异常链。
            with ExitStack() as cleanup:
                for ctrl in self._ctrl_map.values():
                    cleanup.callback(ctrl.close)
                    cleanup.callback(time.sleep, 0.1)
                    cleanup.callback(ctrl.shutdown)
        finally:
            self._ctrl_map.clear()
            self._motor_map.clear()

    def disconnect(self) -> None:
        self.communicator.stop()
        self._clear_commands()
        self.communicator.reset("disconnect")
        self._feedback_valid[:] = False
        if not self._connected:
            return
        try:
            self.disable_all()
        finally:
            try:
                time.sleep(0.5)
                self._close_controllers()
            finally:
                self._connected = False

    def estop(self) -> None:
        self.disable_all()

    # ── 上下文管理器 ───────────────────────────────────────────────────────

    def __enter__(self) -> "RebotArm":
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        gs = ", ".join(f"{k}({g.num_joints}j)" for k, g in self._groups.items())
        return f"RebotArm({self._name!r}, [{gs}])"
