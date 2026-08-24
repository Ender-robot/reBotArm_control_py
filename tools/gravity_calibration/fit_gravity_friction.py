"""全域线性标定：只辨识重力因子与库仑摩擦。

与 fit_calibration.py 的区别：固定 offset=0、bias=0，只解 k 和 fric。

    tau_j = k_j * g_j(q) + fric_j * sign(v_j)

去掉 offset 后模型对参数完全线性，无需网格搜索；去掉 bias 后消除了
与 fric 的共线性（两者都是常数项量级，实测 bias 会吸走 fric 的贡献，
joint3 曾拟合出 bias=-4.3 这种无物理意义的值）。

代价：无法补偿真实存在的零点偏移和力矩偏置，拟合残差会略大。
收益：参数量减半、无局部极小、少量数据即可稳定求解，且每个参数
都有明确物理意义——k 应接近 1，fric 应为正。

运行：python fit_gravity_friction.py data/sweep_<tag>.json
输出：data/calib_<tag>_gf.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from reBotArm_control_py.kinematics import load_robot_model


VELOCITY_DEADBAND = 0.01   # 速度死区，低于此值视为静止 (rad/s)
MIN_SAMPLES_PER_SIGN = 20  # 每个方向最少样本数
MIN_GRAVITY_SPAN = 0.1     # 重力力矩最小跨度 (N·m)
HUBER_DELTA = 0.5          # Huber 损失转折点 (N·m)
IRLS_ITERATIONS = 8        # 迭代重加权次数

# k 有物理意义的关节。joint1/joint6 绕 Z 轴，重力力矩恒为 0。
GRAVITY_JOINTS = [1, 2, 3, 4]


def load_sweep(path):
    """加载 sweep 数据，返回 (q, v, tau)。"""
    document = json.loads(Path(path).read_text())
    samples = document["samples"]
    return (
        np.array([s["q"] for s in samples]),
        np.array([s["v"] for s in samples]),
        np.array([s["tau"] for s in samples]),
    )


def gravity_batch(model, data, q_all):
    """批量计算重力力矩，返回 (n, 6)。"""
    result = np.zeros((len(q_all), model.nv))
    for i, q in enumerate(q_all):
        result[i] = pin.computeGeneralizedGravity(model, data, q)
    return result


def huber_weights(residual, delta):
    """Huber 损失对应的 IRLS 权重。

    静摩擦-滑动切换会产生力矩尖峰，平方损失对这些离群点过于敏感。
    """
    absolute = np.abs(residual)
    weights = np.ones_like(absolute)
    large = absolute > delta
    weights[large] = delta / absolute[large]
    return weights


def fit_joint(gravity, velocity, torque, fit_friction=True):
    """解单关节的 (k, fric)。

    模型对参数线性，用 IRLS + 加权最小二乘闭式求解。

    Args:
        gravity: (n,) 该关节的重力力矩
        velocity: (n,) 该关节速度
        torque: (n,) 该关节实测力矩
        fit_friction: False 时只解 k（用于重力力矩恒为 0 的关节）

    Returns:
        {"k", "friction", "rms", "samples", ...} 或 None
    """
    moving = np.abs(velocity) > VELOCITY_DEADBAND
    if moving.sum() < MIN_SAMPLES_PER_SIGN * 2:
        return None

    g = gravity[moving]
    direction = np.sign(velocity[moving])
    target = torque[moving]

    forward = int((direction > 0).sum())
    backward = int((direction < 0).sum())
    bidirectional = min(forward, backward) >= MIN_SAMPLES_PER_SIGN
    span = float(g.max() - g.min())

    # 单向数据时 fric 与常数无法区分，退化为只解 k
    columns = [g]
    if fit_friction and bidirectional:
        columns.append(direction)
    matrix = np.column_stack(columns)

    weights = np.ones(len(target))
    coefficients = None
    for _ in range(IRLS_ITERATIONS):
        root = np.sqrt(weights)[:, None]
        coefficients, _, _, _ = np.linalg.lstsq(
            matrix * root, target * root[:, 0], rcond=None,
        )
        residual = matrix @ coefficients - target
        weights = huber_weights(residual, HUBER_DELTA)

    residual = matrix @ coefficients - target
    rms = float(np.sqrt(np.mean(residual ** 2)))

    scale = np.linalg.norm(matrix, axis=0)
    scale[scale < 1e-12] = 1.0
    condition = float(np.linalg.cond(matrix / scale))

    return {
        "k": float(coefficients[0]),
        "friction": (
            float(coefficients[1]) if len(coefficients) > 1 else None
        ),
        "rms": rms,
        "samples": int(moving.sum()),
        "forward": forward,
        "backward": backward,
        "gravity_span": span,
        "condition": condition,
        "bidirectional": bidirectional,
        "gravity_identifiable": span >= MIN_GRAVITY_SPAN,
    }


def split_fit(gravity, velocity, torque):
    """分方向独立拟合，用于交叉检验。

    若摩擦模型完整，正反向解出的 k 应一致。差异大说明还有未建模的
    方向相关效应（粘性摩擦、齿隙），此时 k 的可信度下降。
    """
    moving = np.abs(velocity) > VELOCITY_DEADBAND
    result = {}
    for sign, name in ((1, "forward"), (-1, "backward")):
        subset = moving & (np.sign(velocity) == sign)
        if subset.sum() < MIN_SAMPLES_PER_SIGN:
            result[name] = None
            continue
        g = gravity[subset]
        matrix = np.column_stack([g, np.ones(subset.sum())])
        coefficients, _, _, _ = np.linalg.lstsq(
            matrix, torque[subset], rcond=None,
        )
        result[name] = {
            "k": float(coefficients[0]),
            "offset": float(coefficients[1]),
        }
    return result


def fit_all(q, v, tau, model, verbose=True):
    """全关节拟合。"""
    data = model.createData()
    gravity = gravity_batch(model, data, q)

    results = {}
    for joint in range(6):
        # 重力力矩恒为 0 的关节，k 无意义，只标摩擦
        span = gravity[:, joint].max() - gravity[:, joint].min()
        fit_friction = True
        result = fit_joint(
            gravity[:, joint], v[:, joint], tau[:, joint], fit_friction,
        )
        if result is not None:
            result["split"] = split_fit(
                gravity[:, joint], v[:, joint], tau[:, joint],
            )
        results[joint] = result
        if verbose:
            mark = "✓" if result else "×"
            print(f"  joint{joint+1}: {mark}")

    return results


def print_results(results):
    """打印结果表格与可信度诊断。"""
    print()
    print("=" * 88)
    print("标定结果（只标 k 与 friction，offset=0 bias=0）")
    print("=" * 88)
    print(f"{'关节':7s} {'k':>8s} {'fric':>8s} {'rms':>8s} "
          f"{'g跨度':>8s} {'cond':>7s} {'正反k差':>9s} {'诊断':12s}")
    print("-" * 88)

    for joint in range(6):
        result = results.get(joint)
        name = f"joint{joint+1}"

        if result is None:
            print(f"{name:7s} {'-':>8s} {'-':>8s} {'-':>8s} "
                  f"{'-':>8s} {'-':>7s} {'-':>9s} 运动不足")
            continue

        k_text = f"{result['k']:8.4f}"
        fric_text = (
            f"{result['friction']:8.4f}"
            if result["friction"] is not None else "       -"
        )

        # 正反向 k 差异——摩擦模型完整性的检验
        split = result.get("split", {})
        forward = split.get("forward")
        backward = split.get("backward")
        if forward and backward and abs(forward["k"]) > 1e-6:
            difference = abs(forward["k"] - backward["k"])
            reference = max(abs(forward["k"]), abs(backward["k"]))
            split_text = f"{100*difference/reference:8.1f}%"
            split_ratio = difference / reference
        else:
            split_text = "        -"
            split_ratio = 0.0

        notes = []
        if not result["gravity_identifiable"]:
            notes.append("g≈0")
        if not result["bidirectional"]:
            notes.append("单向")
        if result["condition"] > 100:
            notes.append("病态")
        if split_ratio > 0.25:
            notes.append("k存疑")
        if result["gravity_span"] < 2.0 and result["gravity_identifiable"]:
            notes.append("跨度小")
        status = " ".join(notes) if notes else "✓"

        print(
            f"{name:7s} {k_text} {fric_text} {result['rms']:8.4f} "
            f"{result['gravity_span']:8.3f} {result['condition']:7.1f} "
            f"{split_text} {status:12s}"
        )

    print("=" * 88)
    print()
    print("诊断标记：")
    print("  g≈0      重力力矩跨度 < 0.1 N·m，k 无物理意义"
          "（joint1/joint6 绕 Z 轴必然如此）")
    print("  单向     缺少反向样本，friction 不可辨识")
    print("  病态     设计矩阵条件数 > 100")
    print("  k存疑    正反向 k 差异 > 25%，摩擦模型未吃净方向效应")
    print("  跨度小   g 跨度 < 2 N·m，k 受噪声影响大")
    print()
    print("参考：k 理论值应接近 1.0，friction 应为正值。")
    print("      偏离过大说明数据信息量不足或 URDF 质量参数有误。")
    print()


def save_results(results, output_path):
    """保存为 config 可用的格式。"""
    joints = []
    for joint in range(6):
        result = results.get(joint)
        entry = {"name": f"joint{joint+1}"}
        if result is None:
            entry.update({
                "gravity_k": None, "friction": None,
                "offset": 0.0, "tau_bias": 0.0,
            })
        else:
            entry.update({
                "gravity_k": (
                    result["k"] if result["gravity_identifiable"] else None
                ),
                "friction": result["friction"],
                # 本脚本不辨识这两项，显式写 0 以便直接抄进 config
                "offset": 0.0,
                "tau_bias": 0.0,
                "rms": result["rms"],
                "gravity_span": result["gravity_span"],
                "condition": result["condition"],
                "samples": result["samples"],
            })
        joints.append(entry)

    output_path.write_text(json.dumps(
        {"method": "gravity_friction_only", "joints": joints},
        indent=2, ensure_ascii=False,
    ))


def main():
    parser = argparse.ArgumentParser(
        description="只标定重力因子与库仑摩擦（线性模型）",
    )
    parser.add_argument("input", type=Path, help="sweep_<tag>.json")
    parser.add_argument("--stride", type=int, default=1,
                        help="样本降采样步长，加速大数据集")
    args = parser.parse_args()

    q, v, tau = load_sweep(args.input)
    if args.stride > 1:
        q, v, tau = q[::args.stride], v[::args.stride], tau[::args.stride]

    print(f"加载 {len(q)} 个样本"
          + (f"（降采样 1/{args.stride}）" if args.stride > 1 else ""))
    print()
    print("拟合中（线性模型，无网格搜索）...")

    model = load_robot_model()
    results = fit_all(q, v, tau, model)

    print_results(results)

    tag = args.input.stem.replace("sweep_", "")
    output = args.input.parent / f"calib_{tag}_gf.json"
    save_results(results, output)
    print(f"已保存到 {output}")
    print()
    print("填入 config/rebotarm_dm.yaml 的 calibration 段：")
    print("  gravity_k → gravity_k    friction → friction")
    print("  offset 与 tau_bias 保持 0（本方法不辨识）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
