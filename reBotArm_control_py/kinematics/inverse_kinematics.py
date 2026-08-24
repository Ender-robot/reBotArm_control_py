"""reBot-DevArm 逆运动学模块。

基于阻尼最小二乘（CLIK）的闭环逆运动学算法。
位置和姿态误差按独立容差归一化后，使用无量纲 L2 误差收敛。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pinocchio as pin



# ─── 参数与结果数据结构 ────────────────────────────────────────────────────────

@dataclass
class IKParams:
    """IK 求解器参数"""
    max_iter: int = 500
    position_tolerance: float = 0.002
    orientation_tolerance: float = math.radians(1.0)
    step_size: float = 0.5    # 步长
    damping: float = 1e-6      # Tikhonov 正则化系数 λ


@dataclass
class IKResult:
    """IK 求解结果"""
    q: np.ndarray
    success: bool
    error: float       # 最终归一化 L2 误差
    position_error: float
    orientation_error: float
    iterations: int


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
    """从位置和旋转构建 pinocchio SE3 位姿

    参数:
        pos:    (3,) 位置 [x, y, z]，单位：米
        rot:    (3, 3) 旋转矩阵。若提供则忽略 rpy 参数
        roll:  绕 X 轴转角（弧度），仅当 rot=None 时使用
        pitch: 绕 Y 轴转角（弧度），仅当 rot=None 时使用
        yaw:   绕 Z 轴转角（弧度），仅当 rot=None 时使用

    返回:
        pin.SE3 目标末端位姿
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
) -> tuple[np.ndarray, float, float]:
    """计算当前末端位姿与目标位姿之间的 6 维误差 twist

    返回:
        (err_vector, position_error, orientation_error)
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T_cur = data.oMf[end_frame_id]
    err = pin.log6(T_cur.inverse() * target).vector
    position_error = float(np.linalg.norm(err[:3]))
    orientation_error = float(np.linalg.norm(err[3:]))
    return err, position_error, orientation_error


def _compute_normalized_error(
    position_error: float,
    orientation_error: float,
    params: IKParams,
) -> float:
    """按独立位置、姿态容差计算无量纲归一化 L2 误差。"""
    return math.sqrt(
        (position_error / params.position_tolerance) ** 2
        + (orientation_error / params.orientation_tolerance) ** 2
    )


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
      - 自适应阻尼 lam = params.damping * max(1.0, normalized_err * 10.0)
      - 回退线搜索（最多折半 4 次）
      - 归一化位置、姿态误差的 L2 收敛判定（阈值为 1.0）

    参数:
        model:            Pinocchio 机器人模型。
        data:             Pinocchio 数据缓存（需外部创建并传入）。
        end_frame_id:     末端帧索引。
        target:           目标 SE3 位姿。
        q_init:           初始关节配置。若维度小于 model.nq，超出部分视为被动关节补 0；
                          若维度大于 model.nq，多余部分被忽略。
        params:           IK 参数，默认位置 1 mm、姿态 1°。
        controlled_joints: 受控关节数量（默认为 model.nq）。
                          传入比 model.nq 小的值时，IK 在完整模型空间求解，
                          但 q_init 只需提供受控关节数，返回值也只截取受控部分。
                          这使得调用方无需感知 URDF 中被动关节的存在。

    返回:
        IKResult，其中 ``error`` 为归一化 L2 误差，
        ``position_error`` 单位为米，``orientation_error`` 单位为弧度。
    """
    if params is None:
        params = IKParams()
    if (
        not np.isfinite(params.position_tolerance)
        or params.position_tolerance <= 0.0
    ):
        raise ValueError("position_tolerance must be finite and positive")
    if (
        not np.isfinite(params.orientation_tolerance)
        or params.orientation_tolerance <= 0.0
    ):
        raise ValueError("orientation_tolerance must be finite and positive")

    nq = model.nq
    n_ctrl = controlled_joints if controlled_joints is not None else nq

    # 补齐 q_init 到 model.nq
    q = np.zeros(nq)
    n_provided = min(q_init.shape[0], n_ctrl)
    q[:n_provided] = q_init[:n_provided]
    err, position_error, orientation_error = _compute_error(
        model, data, end_frame_id, q, target
    )
    prev_err = _compute_normalized_error(
        position_error, orientation_error, params
    )

    # 初始误差即已满足容差时直接返回
    if prev_err < 1.0:
        return IKResult(
            q=q[:n_ctrl],
            success=True,
            error=prev_err,
            position_error=position_error,
            orientation_error=orientation_error,
            iterations=0,
        )

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
            err_new, position_error_new, orientation_error_new = _compute_error(
                model, data, end_frame_id, q_new, target
            )
            new_err = _compute_normalized_error(
                position_error_new, orientation_error_new, params
            )
            if new_err < prev_err:
                q = q_new
                err = err_new
                prev_err = new_err
                position_error = position_error_new
                orientation_error = orientation_error_new
                if prev_err < 1.0:
                    return IKResult(
                        q=q[:n_ctrl],
                        success=True,
                        error=prev_err,
                        position_error=position_error,
                        orientation_error=orientation_error,
                        iterations=iteration + 1,
                    )
                break
            alpha *= 0.5
        else:
            # 搜索全部失败，保持当前构型继续迭代
            pass

    return IKResult(
        q=q[:n_ctrl],
        success=False,
        error=prev_err,
        position_error=position_error,
        orientation_error=orientation_error,
        iterations=params.max_iter,
    )


def solve_ik_with_retry(
    model: pin.Model,
    data: pin.Data,
    end_frame_id: int,
    target: pin.SE3,
    q_seed: np.ndarray,
    params: Optional[IKParams] = None,
    max_retries: int = 8,
) -> IKResult:
    """带随机重试的 IK 求解器

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
    """使用默认模型计算 IK（便捷函数）

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
