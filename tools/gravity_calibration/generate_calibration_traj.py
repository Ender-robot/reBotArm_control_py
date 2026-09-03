"""生成用于重力和摩擦力标定的安全恒速激励轨迹。

Joint 1 和 6 固定在零位，只优化 Joint 2-5。轨迹采用周期傅里叶曲线，
按关节空间弧长重采样为恒速，并在写出前检查关节限位、自碰撞、桌面碰撞
以及各活动关节的重力力矩跨度。

运行：python generate_calibration_traj.py --tag demo --duration 300
输出：data/traj_<tag>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import coal
import pinocchio as pin
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reBotArm_control_py.kinematics import _resolve_urdf

ACTIVE_JOINTS = [1, 2, 3, 4]  # joint2-5 的索引（0-based）
TARGET_RATIO = 0.8


def _build_collision_model(model, urdf_path, package_dir):
    """创建只包含非相邻连杆组合的自碰撞模型。"""
    collision_model = pin.buildGeomFromUrdf(
        model,
        urdf_path,
        pin.GeometryType.COLLISION,
        package_dirs=[package_dir],
    )

    for first in range(collision_model.ngeoms):
        for second in range(first + 1, collision_model.ngeoms):
            first_joint = collision_model.geometryObjects[first].parentJoint
            second_joint = collision_model.geometryObjects[second].parentJoint

            same_joint = first_joint == second_joint
            adjacent = (
                model.parents[first_joint] == second_joint
                or model.parents[second_joint] == first_joint
            )
            if not same_joint and not adjacent:
                collision_model.addCollisionPair(pin.CollisionPair(first, second))

    return collision_model


def _geometry_collides_with_table(geometry, placement, table_height):
    """检查一个碰撞几何是否接触或穿过水平桌面。"""
    geometry.computeLocalAABB()
    local_aabb = geometry.aabb_local
    center = (local_aabb.min_ + local_aabb.max_) * 0.5
    half_size = (local_aabb.max_ - local_aabb.min_) * 0.5
    rotation = placement.getRotation()
    translation = placement.getTranslation()
    minimum_z = (
        (rotation @ center + translation)[2]
        - np.abs(rotation[2]) @ half_size
    )
    if minimum_z > table_height:
        return False

    table = coal.Halfspace(np.array([0.0, 0.0, 1.0]), table_height)
    request = coal.CollisionRequest()
    result = coal.CollisionResult()
    return bool(
        coal.collide(
            geometry,
            placement,
            table,
            coal.Transform3s(),
            request,
            result,
        )
    )


def _resample_constant_speed(q_loop, duration, dt, speed):
    """沿闭合关节曲线按弧长重采样，使关节空间速度恒定。"""
    if duration <= 0.0:
        raise ValueError("duration 必须大于 0")
    if dt <= 0.0:
        raise ValueError("dt 必须大于 0")
    if speed <= 0.0:
        raise ValueError("v_target 必须大于 0")

    q_loop = np.asarray(q_loop, dtype=float)
    if q_loop.ndim != 2 or q_loop.shape[0] < 2:
        raise ValueError("q_loop 至少需要两个路点")

    if not np.allclose(q_loop[0], q_loop[-1]):
        q_loop = np.vstack((q_loop, q_loop[0]))

    segment_length = np.linalg.norm(
        np.diff(q_loop[:, ACTIVE_JOINTS], axis=0),
        axis=1,
    )
    keep = np.concatenate(([True], segment_length > 1e-12))
    q_loop = q_loop[keep]
    segment_length = np.linalg.norm(
        np.diff(q_loop[:, ACTIVE_JOINTS], axis=0),
        axis=1,
    )
    loop_length = float(segment_length.sum())
    if loop_length <= 0.0:
        raise ValueError("轨迹弧长必须大于 0")

    cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
    n_intervals = int(round(duration / dt))
    if n_intervals < 1 or not np.isclose(n_intervals * dt, duration):
        raise ValueError("duration 必须是 dt 的整数倍")

    distances = np.arange(n_intervals + 1, dtype=float) * speed * dt
    distances = np.mod(distances, loop_length)
    q = np.empty((n_intervals + 1, q_loop.shape[1]))
    for joint in range(q_loop.shape[1]):
        q[:, joint] = np.interp(distances, cumulative, q_loop[:, joint])

    q[:, 0] = 0.0
    q[:, 5] = 0.0
    return q


def _rotate_loop_for_zero_entry(q_loop, step_length):
    """在距零位为整数步长的位置切开闭环。"""
    q_loop = np.asarray(q_loop, dtype=float)
    if np.allclose(q_loop[0], q_loop[-1]):
        q_loop = q_loop[:-1]

    best = None
    for index in range(len(q_loop)):
        first = q_loop[index]
        second = q_loop[(index + 1) % len(q_loop)]
        first_active = first[ACTIVE_JOINTS]
        delta = second[ACTIVE_JOINTS] - first_active
        quadratic = float(delta @ delta)
        if quadratic <= 1e-18:
            continue

        minimum_ratio = np.clip(
            -float(first_active @ delta) / quadratic,
            0.0,
            1.0,
        )
        minimum_radius = np.linalg.norm(first_active + minimum_ratio * delta)
        maximum_radius = max(
            np.linalg.norm(first_active),
            np.linalg.norm(first_active + delta),
        )
        first_step = max(1, int(np.ceil(minimum_radius / step_length)))
        last_step = int(np.floor(maximum_radius / step_length))

        for n_steps in range(first_step, last_step + 1):
            radius = n_steps * step_length
            linear = 2.0 * float(first_active @ delta)
            constant = float(first_active @ first_active) - radius * radius
            discriminant = linear * linear - 4.0 * quadratic * constant
            if discriminant < 0.0:
                continue
            root = np.sqrt(discriminant)
            for ratio in (
                (-linear - root) / (2.0 * quadratic),
                (-linear + root) / (2.0 * quadratic),
            ):
                if -1e-12 <= ratio <= 1.0 + 1e-12:
                    candidate = (n_steps, index, float(np.clip(ratio, 0.0, 1.0)))
                    if best is None or candidate < best:
                        best = candidate

    if best is None:
        raise RuntimeError("无法在傅里叶闭环上找到恒速零位切入点")

    entry_steps, index, ratio = best
    first = q_loop[index]
    second = q_loop[(index + 1) % len(q_loop)]
    entry = first + ratio * (second - first)
    rotated = [entry]
    for offset in range(1, len(q_loop) + 1):
        rotated.append(q_loop[(index + offset) % len(q_loop)])
    rotated.append(entry)
    return np.asarray(rotated), entry_steps


def _resample_from_zero(q_loop, duration, dt, speed):
    """先从全零位恒速进入闭环，再沿闭环恒速运动。"""
    step_length = speed * dt
    rotated, entry_steps = _rotate_loop_for_zero_entry(q_loop, step_length)
    n_intervals = int(round(duration / dt))
    if n_intervals <= entry_steps:
        raise ValueError("duration 不足以从零位进入激励闭环")

    zero = np.zeros(rotated.shape[1])
    entry = np.linspace(zero, rotated[0], entry_steps + 1)
    remaining_duration = (n_intervals - entry_steps) * dt
    loop = _resample_constant_speed(
        rotated,
        remaining_duration,
        dt,
        speed,
    )
    return np.vstack((entry[:-1], loop))


def _select_excitation_waypoints(q_candidates, gravity, targets):
    """选择覆盖每个活动关节重力跨度目标的候选构型。"""
    active_gravity = np.asarray(gravity)[:, ACTIVE_JOINTS]
    targets = np.asarray(targets)
    available_spans = np.ptp(active_gravity, axis=0)
    if np.any(available_spans < targets):
        raise RuntimeError(
            "安全候选构型的重力力矩跨度不足，无法达到 80% 目标"
        )

    selected = []
    for joint in range(len(ACTIVE_JOINTS)):
        selected.append(int(np.argmin(active_gravity[:, joint])))
        selected.append(int(np.argmax(active_gravity[:, joint])))

    return list(dict.fromkeys(selected))


def _gravity_values(model, data, q_samples):
    values = np.empty((len(q_samples), model.nv))
    for index, q in enumerate(q_samples):
        values[index] = pin.computeGeneralizedGravity(model, data, q)
    return values


def _compute_theoretical_spans(model, bounds, n_samples):
    """按固定随机序列估计当前 URDF 在活动关节域内的理论跨度。"""
    if n_samples < 100:
        raise ValueError("theory_samples 不能小于 100")

    rng = np.random.default_rng(42)
    q_samples = np.zeros((n_samples, model.nq))
    lower = np.array([bounds[j][0] for j in ACTIVE_JOINTS])
    upper = np.array([bounds[j][1] for j in ACTIVE_JOINTS])
    q_samples[:, ACTIVE_JOINTS] = rng.uniform(
        lower,
        upper,
        (n_samples, len(ACTIVE_JOINTS)),
    )
    gravity = _gravity_values(model, model.createData(), q_samples)
    return np.ptp(gravity[:, ACTIVE_JOINTS], axis=0)


def _configuration_is_safe(
    model,
    model_data,
    collision_model,
    collision_data,
    q,
    table_height,
):
    if pin.computeCollisions(
        model,
        model_data,
        collision_model,
        collision_data,
        q,
        True,
    ):
        return False

    pin.updateGeometryPlacements(
        model,
        model_data,
        collision_model,
        collision_data,
        q,
    )
    for geometry_object, placement in zip(
        collision_model.geometryObjects,
        collision_data.oMg,
    ):
        if geometry_object.parentJoint == 0:
            continue
        transform = coal.Transform3s(placement.rotation, placement.translation)
        if _geometry_collides_with_table(
            geometry_object.geometry,
            transform,
            table_height,
        ):
            return False

    return True


def _sample_safe_candidates(
    model,
    collision_model,
    bounds,
    targets,
    table_height,
    seed,
    max_attempts,
):
    """采样安全构型，直到重力跨度具有足够余量。"""
    rng = np.random.default_rng(seed)
    model_data = model.createData()
    collision_data = collision_model.createData()
    lower = np.array([bounds[j][0] for j in ACTIVE_JOINTS])
    upper = np.array([bounds[j][1] for j in ACTIVE_JOINTS])
    q_candidates = []
    gravity = []

    with tqdm(total=max_attempts, desc="搜索安全激励构型", unit="pose") as bar:
        for _ in range(max_attempts):
            q = np.zeros(model.nq)
            q[ACTIVE_JOINTS] = rng.uniform(lower, upper)
            if _configuration_is_safe(
                model,
                model_data,
                collision_model,
                collision_data,
                q,
                table_height,
            ):
                q_candidates.append(q)
                gravity.append(pin.computeGeneralizedGravity(model, model_data, q))

            bar.update(1)
            if len(q_candidates) >= 40:
                spans = np.ptp(np.asarray(gravity)[:, ACTIVE_JOINTS], axis=0)
                if np.all(spans >= targets * 1.05):
                    break

    if len(q_candidates) < 2:
        raise RuntimeError("没有找到足够的安全构型")

    q_candidates = np.asarray(q_candidates)
    gravity = np.asarray(gravity)
    _select_excitation_waypoints(q_candidates, gravity, targets * 1.02)
    return q_candidates, gravity


def _nearest_neighbor_order(waypoints, start):
    remaining = list(range(len(waypoints)))
    order = [remaining.pop(start)]
    while remaining:
        previous = waypoints[order[-1], ACTIVE_JOINTS]
        next_index = min(
            remaining,
            key=lambda index: np.linalg.norm(
                waypoints[index, ACTIVE_JOINTS] - previous
            ),
        )
        remaining.remove(next_index)
        order.append(next_index)
    return order


def _fourier_matrix(phase, n_harmonics):
    columns = [np.ones(len(phase))]
    for harmonic in range(1, n_harmonics + 1):
        columns.append(np.sin(harmonic * phase))
        columns.append(np.cos(harmonic * phase))
    return np.column_stack(columns)


def _fit_fourier_loop(waypoints, n_harmonics, n_samples, bounds):
    phase_nodes = np.linspace(0.0, 2.0 * np.pi, len(waypoints), endpoint=False)
    coefficients = np.linalg.lstsq(
        _fourier_matrix(phase_nodes, n_harmonics),
        waypoints,
        rcond=None,
    )[0]

    phase = np.linspace(0.0, 2.0 * np.pi, n_samples)
    matrix = _fourier_matrix(phase, n_harmonics)
    q_loop = matrix @ coefficients

    for joint in ACTIVE_JOINTS:
        center = coefficients[0, joint]
        deviation = q_loop[:, joint] - center
        scale = 1.0
        if deviation.max() > 0.0:
            scale = min(
                scale,
                (bounds[joint][1] - center) / deviation.max(),
            )
        if deviation.min() < 0.0:
            scale = min(
                scale,
                (bounds[joint][0] - center) / deviation.min(),
            )
        coefficients[1:, joint] *= max(0.0, min(1.0, scale * 0.999))

    q_loop = matrix @ coefficients
    q_loop[:, 0] = 0.0
    q_loop[:, 5] = 0.0
    return q_loop


def _path_is_safe(
    model,
    collision_model,
    q_path,
    bounds,
    table_height,
    show_progress=False,
):
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    if np.any(q_path < lower) or np.any(q_path > upper):
        return False

    model_data = model.createData()
    collision_data = collision_model.createData()
    iterator = q_path
    if show_progress:
        iterator = tqdm(q_path, desc="验证碰撞约束", unit="pose")
    for q in iterator:
        if not _configuration_is_safe(
            model,
            model_data,
            collision_model,
            collision_data,
            q,
            table_height,
        ):
            return False
    return True


def _find_safe_fourier_loop(
    model,
    collision_model,
    bounds,
    q_candidates,
    gravity,
    targets,
    table_height,
    n_harmonics,
    dt,
    speed,
):
    selected = _select_excitation_waypoints(
        q_candidates,
        gravity,
        targets * 1.02,
    )
    waypoints = q_candidates[selected]
    data = model.createData()

    for start in range(len(waypoints)):
        order = _nearest_neighbor_order(waypoints, start)
        for ordered_indices in (order, list(reversed(order))):
            ordered = waypoints[ordered_indices]
            coarse_loop = _fit_fourier_loop(
                ordered,
                n_harmonics,
                129,
                bounds,
            )
            if not _path_is_safe(
                model,
                collision_model,
                coarse_loop,
                bounds,
                table_height,
            ):
                continue

            coarse_gravity = _gravity_values(model, data, coarse_loop)
            spans = np.ptp(coarse_gravity[:, ACTIVE_JOINTS], axis=0)
            if np.any(spans < targets):
                continue

            coarse_length = np.linalg.norm(
                np.diff(coarse_loop[:, ACTIVE_JOINTS], axis=0),
                axis=1,
            ).sum()
            step = speed * dt * 0.5
            n_samples = max(1025, int(np.ceil(coarse_length / step)) + 1)
            dense_loop = _fit_fourier_loop(
                ordered,
                n_harmonics,
                n_samples,
                bounds,
            )
            if _path_is_safe(
                model,
                collision_model,
                dense_loop,
                bounds,
                table_height,
                show_progress=True,
            ):
                return dense_loop

    raise RuntimeError("未找到同时满足关节限位、自碰撞和桌面约束的傅里叶轨迹")


def _validate_final_trajectory(
    model,
    q_traj,
    dt,
    speed,
    theoretical_spans,
):
    if not np.all(q_traj[0] == 0.0):
        raise RuntimeError("轨迹未从 URDF 全零位开始")
    if not np.all(q_traj[:, 0] == 0.0) or not np.all(q_traj[:, 5] == 0.0):
        raise RuntimeError("Joint 1 或 Joint 6 未保持在零位")

    velocities = np.linalg.norm(
        np.diff(q_traj[:, ACTIVE_JOINTS], axis=0),
        axis=1,
    ) / dt
    speed_error = np.max(np.abs(velocities - speed))
    if speed_error > speed * 0.01:
        raise RuntimeError(f"关节空间速度偏差过大: {speed_error:.6f} rad/s")

    gravity = _gravity_values(model, model.createData(), q_traj)
    spans = np.ptp(gravity[:, ACTIVE_JOINTS], axis=0)
    targets = theoretical_spans * TARGET_RATIO
    if np.any(spans < targets):
        raise RuntimeError("最终轨迹的重力力矩跨度未达到理论值的 80%")
    return spans, velocities


def generate_smooth_trajectory(
    duration=180.0,
    n_harmonics=4,
    v_target=0.15,
    dt=0.02,
    z_min=0.02,
    seed=0,
    urdf_path=None,
    theory_samples=50000,
    max_candidates=400,
):
    """生成并硬验收安全、恒速、充分激励的标定轨迹。"""
    if n_harmonics < 1:
        raise ValueError("n_harmonics 必须大于 0")
    if z_min < 0.0:
        raise ValueError("z_min 不能小于 0")

    print("加载机器人模型...")
    urdf_path, package_dir = _resolve_urdf(urdf_path)
    model = pin.buildModelFromUrdf(urdf_path)
    if model.nq != 6:
        raise ValueError(f"该脚本要求 6 自由度模型，当前 nq={model.nq}")

    collision_model = _build_collision_model(model, urdf_path, package_dir)
    if len(collision_model.collisionPairs) == 0:
        raise RuntimeError("碰撞模型没有有效的自碰撞对")

    bounds = [
        (float(model.lowerPositionLimit[i]), float(model.upperPositionLimit[i]))
        for i in range(model.nq)
    ]
    print(f"自碰撞对: {len(collision_model.collisionPairs)}")
    print(f"桌面安全高度: z >= {z_min:.3f} m")

    print(f"从当前 URDF 计算理论重力跨度（{theory_samples} 个构型）...")
    theoretical_spans = _compute_theoretical_spans(
        model,
        bounds,
        theory_samples,
    )
    targets = theoretical_spans * TARGET_RATIO
    for index, joint in enumerate(ACTIVE_JOINTS):
        print(
            f"  joint{joint + 1}: 理论 {theoretical_spans[index]:.3f} N·m, "
            f"目标 {targets[index]:.3f} N·m"
        )

    q_candidates, gravity = _sample_safe_candidates(
        model,
        collision_model,
        bounds,
        targets,
        z_min,
        seed,
        max_candidates,
    )
    print(f"安全候选构型: {len(q_candidates)}")

    q_loop = _find_safe_fourier_loop(
        model,
        collision_model,
        bounds,
        q_candidates,
        gravity,
        targets,
        z_min,
        n_harmonics,
        dt,
        v_target,
    )
    loop_length = np.linalg.norm(
        np.diff(q_loop[:, ACTIVE_JOINTS], axis=0),
        axis=1,
    ).sum()
    entry_distance = np.min(
        np.linalg.norm(q_loop[:, ACTIVE_JOINTS], axis=1)
    )
    if duration * v_target < loop_length + entry_distance:
        minimum_duration = (loop_length + entry_distance) / v_target
        raise RuntimeError(
            f"轨迹时长不足以覆盖完整激励闭环，至少需要 {minimum_duration:.1f} s"
        )

    q_traj = _resample_from_zero(q_loop, duration, dt, v_target)
    q_traj = q_traj.round(6)
    if not _path_is_safe(
        model,
        collision_model,
        q_traj,
        bounds,
        z_min,
        show_progress=True,
    ):
        raise RuntimeError("最终轨迹存在自碰撞或桌面碰撞")
    spans, velocities = _validate_final_trajectory(
        model,
        q_traj,
        dt,
        v_target,
        theoretical_spans,
    )

    print("\n最终硬验收通过:")
    for index, joint in enumerate(ACTIVE_JOINTS):
        ratio = spans[index] / theoretical_spans[index] * 100.0
        print(f"  joint{joint + 1}: {spans[index]:.3f} N·m ({ratio:.1f}%)")
    print(
        f"  关节空间速度: min={velocities.min():.6f}, "
        f"mean={velocities.mean():.6f}, max={velocities.max():.6f} rad/s"
    )
    print("  自碰撞: 无")
    print(f"  活动连杆最低高度: >= {z_min:.3f} m")

    return {
        "dt": dt,
        "v_target": v_target,
        "q": q_traj.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="生成安全恒速标定轨迹")
    parser.add_argument("--tag", required=True, help="输出文件标签")
    parser.add_argument("--duration", type=float, default=180.0, help="轨迹时长（秒）")
    parser.add_argument("--n-harmonics", type=int, default=4, help="每个关节的谐波数")
    parser.add_argument(
        "--z-min",
        type=float,
        default=0.02,
        help="活动连杆碰撞几何距 z=0 桌面的最小高度（米）",
    )
    parser.add_argument("--v-target", type=float, default=0.15, help="目标关节空间速度（rad/s）")
    parser.add_argument("--dt", type=float, default=0.02, help="采样时间步长（秒）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument(
        "--theory-samples",
        type=int,
        default=50000,
        help="从 URDF 估计理论重力跨度的采样数",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=400,
        help="安全激励构型的最大搜索次数",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="输出目录",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"traj_{args.tag}.json"

    print(f"生成优化标定轨迹: {args.tag}")
    print(
        f"参数: duration={args.duration}s, n_harmonics={args.n_harmonics}, "
        f"dt={args.dt}s, v_target={args.v_target} rad/s"
    )
    print("目标：无自碰撞、不碰桌面、恒速、重力跨度分别达到理论值的 80%")
    print()

    start_time = time.time()
    trajectory = generate_smooth_trajectory(
        duration=args.duration,
        n_harmonics=args.n_harmonics,
        v_target=args.v_target,
        dt=args.dt,
        z_min=args.z_min,
        seed=args.seed,
        theory_samples=args.theory_samples,
        max_candidates=args.max_candidates,
    )
    elapsed = time.time() - start_time

    output_path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False))
    print(f"\n生成完成（耗时 {elapsed:.1f}s）")
    print(f"路点数: {len(trajectory['q'])}")
    print(f"已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
