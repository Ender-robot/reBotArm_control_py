"""reBot-DevArm 逆运动学模块。

基于阻尼最小二乘（CLIK）的闭环逆运动学算法，
与 C++ 实现严格对齐：雅可比矩阵计算、自适应阻尼、回退线搜索。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pinocchio as pin

from .forward_kinematics import compute_fk


# ─── 参数与结果数据结构 ────────────────────────────────────────────────────────

@dataclass
class IKParams:
    """IK 求解器参数"""
    max_iter: int = 1000
    tolerance: float = 1e-4    # 收敛阈值 ||err||
    step_size: float = 0.5    # 每步更新的缩放系数
    damping: float = 1e-6      # Tikhonov 正则化系数 λ


@dataclass
class IKResult:
    """IK 求解结果"""
    q: np.ndarray
    success: bool
    error: float       # 最终 ||err||
    iterations: int


@dataclass
class CLIKParams:
    """单步 CLIK 参数"""
    sigma_safe: float = 0.05   # 开始增加阻尼的奇异值阈值
    damping_max: float = 0.05  # 最大阻尼系数


@dataclass
class CLIKResult:
    """单步 CLIK 计算结果"""
    q: np.ndarray
    qdot: np.ndarray
    error: float        # 当前 TCP 位姿误差范数
    sigma_min: float    # 雅可比矩阵最小奇异值
    damping: float      # 本周期实际阻尼系数
    speed_scale: float  # 关节速度整体缩放比例


# Alias，与 C++ 头文件中的命名保持一致
IKSolverParams = IKParams


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def pos_rot_to_se3(
    pos: np.ndarray,
    rot: Optional[np.ndarray] = None,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> pin.SE3:
    """从位置和旋转构建 pinocchio SE3 位姿。

    参数:
        pos:    (3,) 位置 [x, y, z]，单位：米。
        rot:    (3, 3) 旋转矩阵。若提供则忽略 rpy 参数。
        roll:  绕 X 轴转角（弧度），仅当 rot=None 时使用。
        pitch: 绕 Y 轴转角（弧度），仅当 rot=None 时使用。
        yaw:   绕 Z 轴转角（弧度），仅当 rot=None 时使用。

    返回:
        pin.SE3 目标末端位姿。
    """
    if rot is None:
        rot = pin.rpy.rpyToMatrix(roll, pitch, yaw)
    return pin.SE3(rot, pos)


def xyz_quat_to_se3(target: np.ndarray) -> pin.SE3:
    """将 [x, y, z, qx, qy, qz, qw] 转换为 SE3。"""
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (7,):
        raise ValueError(f"target shape must be (7,), got {target.shape}")
    if not np.all(np.isfinite(target)):
        raise ValueError("target must contain only finite values")

    quaternion = target[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        raise ValueError("target quaternion must not be zero")
    qx, qy, qz, qw = quaternion / norm
    rotation = pin.Quaternion(qw, qx, qy, qz).matrix()
    return pin.SE3(rotation, target[:3])


def _clamp_config(model: pin.Model, q: np.ndarray) -> np.ndarray:
    """将 q 限制在关节限位范围内。

    NaN 限位默认用 0，防止 integrate 后出现 NaN。
    """
    lo = np.array([
        float(x) if np.isfinite(x) else 0.0 for x in model.lowerPositionLimit
    ])
    hi = np.array([
        float(x) if np.isfinite(x) else 0.0 for x in model.upperPositionLimit
    ])
    clamped = np.maximum(q, lo)
    clamped = np.minimum(clamped, hi)
    return clamped


def _compute_error(
    model: pin.Model,
    data: pin.Data,
    end_frame_id: int,
    q: np.ndarray,
    target: pin.SE3,
) -> tuple[float, np.ndarray]:
    """计算当前末端位姿与目标位姿之间的 6 维误差 twist。

    返回:
        (err_norm, err_vector)
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T_cur = data.oMf[end_frame_id]
    err = pin.log6(T_cur.inverse() * target).vector
    return float(np.linalg.norm(err)), err


def _compute_pose_error_jacobian(
    model: pin.Model,
    data: pin.Data,
    end_frame_id: int,
    q: np.ndarray,
    target: pin.SE3,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 SE(3) 位姿误差及其 LOCAL 坐标系雅可比。"""
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    T_cur = data.oMf[end_frame_id]
    T_error = T_cur.inverse() * target
    error = pin.log6(T_error).vector
    J_local = pin.getFrameJacobian(
        model,
        data,
        end_frame_id,
        pin.LOCAL,
    )
    J_error = -pin.Jlog6(T_error.inverse()) @ J_local
    return error, J_error


def _compute_adaptive_damping(
    sigma_min: float,
    params: CLIKParams,
) -> float:
    """根据最小奇异值计算自适应阻尼。"""
    if not np.isfinite(params.sigma_safe) or params.sigma_safe <= 0.0:
        raise ValueError("sigma_safe must be finite and positive")
    if not np.isfinite(params.damping_max) or params.damping_max < 0.0:
        raise ValueError("damping_max must be finite and non-negative")
    if not np.isfinite(sigma_min) or sigma_min < 0.0:
        raise ValueError("sigma_min must be finite and non-negative")
    if sigma_min >= params.sigma_safe:
        return 0.0

    ratio = 1.0 - sigma_min / params.sigma_safe
    return params.damping_max * ratio * ratio


def _scale_joint_velocity(
    qdot: np.ndarray,
    speed,
) -> tuple[np.ndarray, float]:
    """按标量或逐关节上限整体缩放关节速度。"""
    qdot = np.asarray(qdot, dtype=np.float64)
    if qdot.ndim != 1 or not np.all(np.isfinite(qdot)):
        raise ValueError("qdot must be a finite one-dimensional array")

    speed = np.asarray(speed, dtype=np.float64)
    if speed.shape == ():
        speed = np.full(qdot.shape, float(speed))
    if speed.shape != qdot.shape:
        raise ValueError(f"speed shape must be {qdot.shape}, got {speed.shape}")
    if not np.all(np.isfinite(speed)) or np.any(speed < 0.0):
        raise ValueError("speed must contain only finite non-negative values")

    moving = np.abs(qdot) > 0.0
    if not np.any(moving):
        return qdot.copy(), 1.0

    scale = float(np.min(speed[moving] / np.abs(qdot[moving])))
    scale = min(1.0, scale)
    return qdot * scale, scale


def _scale_to_joint_limits(
    model: pin.Model,
    q: np.ndarray,
    qdot: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, float]:
    """整体缩放关节速度，使下一关节目标不越过硬限位。"""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    delta = qdot * dt
    scale = 1.0
    for index in range(model.nq):
        if delta[index] > 0.0 and np.isfinite(model.upperPositionLimit[index]):
            remaining = model.upperPositionLimit[index] - q[index]
            scale = min(scale, max(0.0, remaining / delta[index]))
        elif delta[index] < 0.0 and np.isfinite(model.lowerPositionLimit[index]):
            remaining = q[index] - model.lowerPositionLimit[index]
            scale = min(scale, max(0.0, remaining / -delta[index]))

    scale = min(1.0, scale)
    return qdot * scale, float(scale)


# ─── 核心求解器 ────────────────────────────────────────────────────────────────

def solve_ik(
    model: pin.Model,
    data: pin.Data,
    end_frame_id: int,
    target: pin.SE3,
    q_init: np.ndarray,
    params: Optional[IKParams] = None,
    controlled_joints: int | None = None,
) -> IKResult:
    """阻尼最小二乘 CLIK 求解器。

      - LOCAL 坐标系雅可比
      - 自适应阻尼 lam = params.damping * max(1.0, prev_err * 10.0)
      - 回退线搜索（最多折半 4 次）

    参数:
        model:            Pinocchio 机器人模型。
        data:             Pinocchio 数据缓存（需外部创建并传入）。
        end_frame_id:     末端帧索引。
        target:           目标 SE3 位姿。
        q_init:           初始关节配置。若维度小于 model.nq，超出部分视为被动关节补 0；
                          若维度大于 model.nq，多余部分被忽略。
        params:           IK 参数，默认 IKParams{}。
        controlled_joints: 受控关节数量（默认为 model.nq）。
                          传入比 model.nq 小的值时，IK 在完整模型空间求解，
                          但 q_init 只需提供受控关节数，返回值也只截取受控部分。
                          这使得调用方无需感知 URDF 中被动关节的存在。

    返回:
        IKResult，其中 q 为求解得到的关节角（维度与 q_init 一致）。
    """
    if params is None:
        params = IKParams()

    nq = model.nq
    n_ctrl = controlled_joints if controlled_joints is not None else nq

    # 补齐 q_init 到 model.nq
    q = np.zeros(nq)
    n_provided = min(q_init.shape[0], n_ctrl)
    q[:n_provided] = q_init[:n_provided]
    prev_err, err = _compute_error(model, data, end_frame_id, q, target)

    # 初始误差即已满足容差时直接返回
    if prev_err < params.tolerance:
        return IKResult(q=q[:n_ctrl], success=True, error=prev_err, iterations=0)

    for iteration in range(params.max_iter):

        # LOCAL 系体雅可比
        pin.computeJointJacobians(model, data, q)
        J = pin.getFrameJacobian(model, data, end_frame_id, pin.LOCAL)

        # 自适应阻尼：误差越大阻尼越小（Levenberg-Marquardt 风格）
        lam = params.damping * max(1.0, prev_err * 10.0)

        # 阻尼最小二乘 dq = step_size * J^T * (J J^T + λI)^{-1} * err
        JJT = J @ J.T
        JJT[np.arange(JJT.shape[0]), np.arange(JJT.shape[1])] += lam
        dq = params.step_size * J.T @ np.linalg.solve(JJT, err)

        # 回退线搜索：若新误差未减小则缩步，最多折半 4 次
        alpha = 1.0
        for _ in range(4):
            q_new = _clamp_config(model, pin.integrate(model, q, alpha * dq))
            new_err, err_new = _compute_error(model, data, end_frame_id, q_new, target)
            if new_err < prev_err:
                q = q_new
                err = err_new
                prev_err = new_err
                break
            alpha *= 0.5
        else:
            # 线搜索全部失败，保持当前构型继续迭代
            pass

    # 循环结束后再次检查（可能刚收敛或误差已达机器精度）
    if prev_err < params.tolerance:
        return IKResult(q=q[:n_ctrl], success=True, error=prev_err, iterations=params.max_iter)
    return IKResult(q=q[:n_ctrl], success=False, error=prev_err, iterations=params.max_iter)


def solve_ik_with_retry(
    model: pin.Model,
    data: pin.Data,
    end_frame_id: int,
    target: pin.SE3,
    q_seed: np.ndarray,
    params: Optional[IKParams] = None,
    max_retries: int = 8,
) -> IKResult:
    """带随机重试的 IK 求解器。

      - 先用 q_seed 求解一次
      - 若失败则在关节限位内随机采样最多 max_retries 次
      - 返回误差最小的结果

    参数:
        model:        Pinocchio 机器人模型。
        data:         Pinocchio 数据缓存。
        end_frame_id: 末端帧索引。
        target:       目标 SE3 位姿。
        q_seed:       种子关节配置（会被更新为本次最优解）。
        params:       IK 参数。
        max_retries:  随机重试次数。

    返回:
        IKResult。
    """
    if params is None:
        params = IKParams()

    best = solve_ik(model, data, end_frame_id, target, q_seed, params)
    if best.success:
        q_seed[:] = best.q
        return best

    lo = model.lowerPositionLimit
    hi = model.upperPositionLimit
    nq = model.nq

    for _ in range(max_retries):
        q_rand = np.zeros(nq)
        for j in range(nq):
            l = lo[j] if np.isfinite(lo[j]) else -math.pi
            h = hi[j] if np.isfinite(hi[j]) else math.pi
            q_rand[j] = random.uniform(l, h)
        r = solve_ik(model, data, end_frame_id, target, q_rand, params)
        if r.error < best.error:
            best = r
        if best.success:
            break

    q_seed[:] = best.q
    return best

def solve_clik_step(
    model: pin.Model,
    data: pin.Data,
    end_frame_id: int,
    q_measured: np.ndarray,
    target: pin.SE3,
    dt: float,
    speed,
    gain: float,
    lookahead_time: float,
    params: Optional[CLIKParams] = None,
    q_reference: Optional[np.ndarray] = None,
) -> CLIKResult:
    """执行一次速度级 CLIK 计算。"""
    if params is None:
        params = CLIKParams()

    q_measured = np.asarray(q_measured, dtype=np.float64)
    if q_measured.shape != (model.nq,):
        raise ValueError(
            f"q_measured shape must be ({model.nq},), "
            f"got {q_measured.shape}"
        )
    if not np.all(np.isfinite(q_measured)):
        raise ValueError("q_measured must contain only finite values")
    if q_reference is None:
        q_reference = q_measured
    else:
        q_reference = np.asarray(q_reference, dtype=np.float64)
        if q_reference.shape != (model.nq,):
            raise ValueError(
                f"q_reference shape must be ({model.nq},), "
                f"got {q_reference.shape}"
            )
        if not np.all(np.isfinite(q_reference)):
            raise ValueError("q_reference must contain only finite values")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("gain must be finite and non-negative")

    error, J_error = _compute_pose_error_jacobian(
        model,
        data,
        end_frame_id,
        q_measured,
        target,
    )
    singular_values = np.linalg.svd(J_error, compute_uv=False)
    sigma_min = float(singular_values[-1])
    damping = _compute_adaptive_damping(sigma_min, params)

    matrix = J_error @ J_error.T
    matrix[np.diag_indices_from(matrix)] += damping * damping
    qdot = gain * (J_error.T @ np.linalg.solve(matrix, -error))
    qdot, speed_scale = _scale_joint_velocity(qdot, speed)
    qdot, limit_scale = _scale_to_joint_limits(
        model,
        q_measured,
        qdot,
        lookahead_time,
    )
    speed_scale *= limit_scale
    q = pin.integrate(model, q_measured, qdot * lookahead_time)

    if not np.all(np.isfinite(qdot)) or not np.all(np.isfinite(q)):
        raise FloatingPointError("CLIK result contains non-finite values")

    return CLIKResult(
        q=q,
        qdot=qdot,
        error=float(np.linalg.norm(error)),
        sigma_min=sigma_min,
        damping=damping,
        speed_scale=speed_scale,
    )



# ─── 便捷函数 ──────────────────────────────────────────────────────────────────

def compute_ik(
    q_init: np.ndarray | None,
    target_pos: np.ndarray,
    target_rot: np.ndarray | None = None,
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    params: IKSolverParams | None = None,
) -> IKResult:
    """使用默认模型计算 IK（便捷函数）。

    参数:
        q_init:      初始关节配置。传入 ``None`` 则自动使用零位构型。
        target_pos:  目标位置 (3,)，单位：米。
        target_rot:  目标旋转矩阵 (3, 3)，可选。
        roll:        ZYX 欧拉角之 roll，仅当 rot=None 时使用。
        pitch:       ZYX 欧拉角之 pitch。
        yaw:         ZYX 欧拉角之 yaw。
        params:      IK 参数。

    返回:
        IKResult。
    """
    from .robot_model import load_robot_model, get_end_effector_frame_id

    model = load_robot_model()
    data = model.createData()
    frame_id = get_end_effector_frame_id(model)
    target = pos_rot_to_se3(target_pos, target_rot, roll, pitch, yaw)

    if q_init is None:
        q_init = pin.neutral(model)

    return solve_ik(model, data, frame_id, target, q_init, params)
