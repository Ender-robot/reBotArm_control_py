"""从静态正反向保持数据中拟合重力补偿系数。

输入由 replay_record.py 生成。正反向同一 waypoint 的静态力矩先配对平均，
再拟合：

    tau_j = gravity_k_j * g_j(q)

本脚本只计算 gravity_k。输出配置中的 offset、friction、tau_bias 全部为 0，
运行时补偿模型仍由 actuator 保留。

运行：修改下方路径后直接运行本文件
输出：data/calib_gravity_<tag>.json
"""

import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from reBotArm_control_py.kinematics import load_robot_model


DATA_DIR = Path(__file__).parent / "data"
INPUT_PATH = DATA_DIR / "sweep_static_No1.json"
OUTPUT_PATH = DATA_DIR / "calib_gravity_No1.json"

MIN_STATIC_SAMPLES = 3
MIN_GRAVITY_SPAN = 0.1  # N·m
HUBER_DELTA = 0.5       # N·m
IRLS_ITERATIONS = 8
GRAVITY_JOINTS = [1, 2, 3, 4]


def aggregate_static_samples(samples):
    """按 waypoint 配对正反向保持数据并取平均。

    返回配对后的实际姿态、力矩和原始 waypoint 索引。缺少任一方向的点
    会被跳过，避免单向静摩擦污染重力拟合。
    """
    grouped = {}
    for sample in samples:
        waypoint_index = int(sample["waypoint_index"])
        pass_name = sample["pass"]
        grouped.setdefault(waypoint_index, {}).setdefault(pass_name, []).append(sample)

    q_values = []
    tau_values = []
    waypoint_indices = []
    for waypoint_index in sorted(grouped):
        passes = grouped[waypoint_index]
        if "forward" not in passes or "reverse" not in passes:
            continue

        forward_q = np.mean([sample["q"] for sample in passes["forward"]], axis=0)
        reverse_q = np.mean([sample["q"] for sample in passes["reverse"]], axis=0)
        forward_tau = np.mean(
            [sample["tau"] for sample in passes["forward"]], axis=0,
        )
        reverse_tau = np.mean(
            [sample["tau"] for sample in passes["reverse"]], axis=0,
        )
        q_values.append((forward_q + reverse_q) / 2.0)
        tau_values.append((forward_tau + reverse_tau) / 2.0)
        waypoint_indices.append(waypoint_index)

    if not q_values:
        raise ValueError("没有找到同时包含 forward 和 reverse 的静态路径点")

    return (
        np.asarray(q_values, dtype=float),
        np.asarray(tau_values, dtype=float),
        waypoint_indices,
    )


def load_sweep(path):
    """加载静态 sweep，返回配对后的 (q, tau, waypoint_indices)。"""
    document = json.loads(Path(path).read_text())
    samples = document["samples"]
    if not samples or "waypoint_index" not in samples[0]:
        raise ValueError("输入不是静态采样数据，请先运行 replay_record.py")
    return aggregate_static_samples(samples)


def gravity_batch(model, data, q_all):
    """批量计算重力力矩，返回 (n, 6)。"""
    result = np.zeros((len(q_all), model.nv))
    for i, q in enumerate(q_all):
        result[i] = pin.computeGeneralizedGravity(model, data, q)
    return result


def huber_weights(residual, delta):
    """Huber 损失对应的 IRLS 权重。"""
    absolute = np.abs(residual)
    weights = np.ones_like(absolute)
    large = absolute > delta
    weights[large] = delta / absolute[large]
    return weights


def fit_joint(gravity, torque):
    """只用重力列拟合单关节 gravity_k。"""
    if len(gravity) < MIN_STATIC_SAMPLES:
        return None

    span = float(np.max(gravity) - np.min(gravity))
    matrix = np.asarray(gravity, dtype=float)[:, None]
    target = np.asarray(torque, dtype=float)

    weights = np.ones(len(target))
    coefficient = None
    for _ in range(IRLS_ITERATIONS):
        root = np.sqrt(weights)[:, None]
        coefficient, _, _, _ = np.linalg.lstsq(
            matrix * root, target * root[:, 0], rcond=None,
        )
        residual = matrix[:, 0] * coefficient[0] - target
        weights = huber_weights(residual, HUBER_DELTA)

    residual = matrix[:, 0] * coefficient[0] - target
    scale = np.linalg.norm(matrix, axis=0)
    scale[scale < 1e-12] = 1.0

    return {
        "k": float(coefficient[0]),
        "rms": float(np.sqrt(np.mean(residual ** 2))),
        "samples": int(len(target)),
        "gravity_span": span,
        "condition": float(np.linalg.cond(matrix / scale)),
        "gravity_identifiable": span >= MIN_GRAVITY_SPAN,
    }


def fit_all(q, tau, model, verbose=True):
    """拟合所有可辨识的重力关节。"""
    data = model.createData()
    gravity = gravity_batch(model, data, q)

    results = {}
    for joint in range(6):
        if joint not in GRAVITY_JOINTS:
            results[joint] = None
        else:
            results[joint] = fit_joint(gravity[:, joint], tau[:, joint])

        if verbose:
            mark = "✓" if results[joint] else "×"
            print(f"  joint{joint+1}: {mark}")

    return results


def print_results(results):
    """打印重力拟合结果与可信度诊断。"""
    print()
    print("=" * 72)
    print("重力标定结果（仅 gravity_k，其余标定项为 0）")
    print("=" * 72)
    print(f"{'关节':7s} {'gravity_k':>12s} {'rms':>10s} "
          f"{'g跨度':>10s} {'样本':>8s} {'诊断':12s}")
    print("-" * 72)

    for joint in range(6):
        result = results.get(joint)
        name = f"joint{joint+1}"
        if result is None:
            print(f"{name:7s} {'-':>12s} {'-':>10s} {'-':>10s} "
                  f"{'-':>8s} 不可辨识")
            continue

        notes = []
        if not result["gravity_identifiable"]:
            notes.append("g≈0")
        if result["gravity_span"] < 2.0:
            notes.append("跨度小")
        status = " ".join(notes) if notes else "✓"
        print(
            f"{name:7s} {result['k']:12.4f} {result['rms']:10.4f} "
            f"{result['gravity_span']:10.3f} {result['samples']:8d} "
            f"{status:12s}"
        )

    print("=" * 72)
    print()
    print("参考：gravity_k 理论值应接近 1.0。")
    print("      偏离过大说明数据信息量不足或 URDF 质量参数有误。")
    print()


def save_results(results, output_path):
    """保存为 config 可用的格式。"""
    joints = []
    for joint in range(6):
        result = results.get(joint)
        entry = {
            "name": f"joint{joint+1}",
            "gravity_k": 0.0,
            "friction": 0.0,
            "offset": 0.0,
            "tau_bias": 0.0,
        }
        if result is not None:
            entry.update({
                "gravity_k": (
                    result["k"] if result["gravity_identifiable"] else 0.0
                ),
                "rms": result["rms"],
                "gravity_span": result["gravity_span"],
                "condition": result["condition"],
                "samples": result["samples"],
            })
        joints.append(entry)

    output_path.write_text(json.dumps(
        {"method": "gravity_only", "joints": joints},
        indent=2, ensure_ascii=False,
    ))


def main():
    q, tau, waypoint_indices = load_sweep(INPUT_PATH)
    print(f"配对静态路径点: {len(waypoint_indices)}")
    print("拟合中（纯重力模型，无摩擦项）...")

    model = load_robot_model()
    results = fit_all(q, tau, model)

    print_results(results)
    save_results(results, OUTPUT_PATH)
    print(f"已保存到 {OUTPUT_PATH}")
    print()
    print("填入 config/rebotarm_dm.yaml 的 calibration 段：")
    print("  gravity_k → gravity_k")
    print("  offset、friction、tau_bias 均为 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
