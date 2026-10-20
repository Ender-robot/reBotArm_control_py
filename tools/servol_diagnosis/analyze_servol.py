"""分析 servoL(mit) 诊断数据, 输出结论与曲线图。

运行：python analyze_servol.py data/servol_<tag>.json
输出：data/servol_<tag>_report.json, data/servol_<tag>_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_arrays(data):
    """把 samples 里的逐周期字段抽成 (N,) 或 (N, 6) 数组。"""
    samples = data["samples"]
    fields = [
        "q_feedback", "qdot_feedback", "torque_feedback", "q_ik",
        "delta", "cmd_position", "cmd_velocity",
    ]
    arrays = {name: np.array([s[name] for s in samples]) for name in fields}
    arrays["t"] = np.array([s["t"] for s in samples])
    arrays["phase"] = np.array([s["phase"] for s in samples])
    arrays["ik_iterations"] = np.array([s["ik_iterations"] for s in samples])
    arrays["ik_success"] = np.array([s["ik_success"] for s in samples])
    arrays["servol_duration"] = np.array([s["servol_duration"] for s in samples])
    return arrays


def dominant_frequency(signal, rate):
    """返回信号去均值后的主频 (Hz) 与该频点的幅值占比。"""
    centered = signal - signal.mean()
    if not np.any(centered):
        return 0.0, 0.0
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(centered), 1.0 / rate)
    peak = int(np.argmax(spectrum[1:])) + 1
    total = spectrum[1:].sum()
    return float(freqs[peak]), float(spectrum[peak] / total) if total else 0.0


def analyze_hold_phases(arrays, rate, joint_names):
    """抖动统计。

    优先用静止保持阶段 (目标不动, 任何波动都是自激)。中断录制可能没跑到
    保持段, 此时退化为对运动阶段做去趋势统计。
    """
    result = {}
    phases = ["z_hold", "x_hold"]
    if not np.isin(arrays["phase"], phases).any():
        phases = sorted(set(arrays["phase"].tolist()))
        result["_note"] = "无静止保持段, 下列统计对运动阶段做了去趋势处理"

    for phase in phases:
        mask = arrays["phase"] == phase
        if mask.sum() < 8:
            continue
        detrend = phase.endswith("_move")
        stats = []
        for j, name in enumerate(joint_names):
            cmd_vel = arrays["cmd_velocity"][mask, j]
            delta = arrays["delta"][mask, j]
            qdot = arrays["qdot_feedback"][mask, j]
            if detrend: # 运动段有真实速度趋势, 需扣除后才能看抖动
                cmd_vel = cmd_vel - np.poly1d(
                    np.polyfit(np.arange(len(cmd_vel)), cmd_vel, 3)
                )(np.arange(len(cmd_vel)))
            freq, share = dominant_frequency(cmd_vel, rate)
            stats.append({
                "joint": name,
                "cmd_velocity_std": round(float(cmd_vel.std()), 6),
                "cmd_velocity_peak_to_peak": round(float(np.ptp(cmd_vel)), 6),
                "delta_std": round(float(delta.std()), 8),
                "delta_mean": round(float(delta.mean()), 8),
                "qdot_feedback_std": round(float(qdot.std()), 6),
                "cmd_velocity_dominant_hz": round(freq, 3),
                "dominant_freq_share": round(share, 4),
                "sign_flips_per_second": round(
                    float(np.count_nonzero(np.diff(np.sign(qdot))) / (mask.sum() / rate)),
                    2,
                ),
            })
        result[phase] = stats
    return result


def plot_curves(arrays, data, output_path):
    """绘制诊断曲线: 每行一个关注量, 6 关节同图。"""
    t = arrays["t"]
    names = data["joint_names"]
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    for j, name in enumerate(names):
        axes[0].plot(t, arrays["cmd_velocity"][:, j], linewidth=0.8, label=name)
        axes[1].plot(t, arrays["delta"][:, j], linewidth=0.8, label=name)
        axes[2].plot(t, arrays["qdot_feedback"][:, j], linewidth=0.8, label=name)
        axes[3].plot(t, arrays["torque_feedback"][:, j], linewidth=0.8, label=name)

    axes[0].set_ylabel("cmd velocity\n(rad/s)")
    axes[0].set_title(
        f"servoL(mit) diagnosis  "
        f"rate={data['rate']}Hz"
    )
    axes[1].set_ylabel("position error delta\n(rad)")
    axes[2].set_ylabel("feedback velocity\n(rad/s)")
    axes[3].set_ylabel("feedback torque\n(Nm)")

    axes[3].set_xlabel("time (s)")

    # 标出静止保持区间
    for axis in axes:
        for phase, color in [("z_hold", "0.85"), ("x_hold", "0.92")]:
            mask = arrays["phase"] == phase
            if mask.any():
                axis.axvspan(t[mask].min(), t[mask].max(), color=color, zorder=0)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, ncol=6, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="分析 servoL(mit) 诊断数据")
    parser.add_argument("input", type=Path, help="输入 servol_<tag>.json")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    arrays = load_arrays(data)
    rate = data["rate"]

    report = {
        "source": args.input.name,
        "rate": rate,
        "aborted": data.get("aborted", False),
        "planned_samples": data.get("planned_samples"),
        "comm_rate": data["comm_rate"],
        "sample_count": len(data["samples"]),
        "ik_failures": int((~arrays["ik_success"]).sum()),
        "ik_iterations_mean": round(float(arrays["ik_iterations"].mean()), 2),
        "ik_iterations_max": int(arrays["ik_iterations"].max()),
        "servol_duration_mean_ms": round(
            float(arrays["servol_duration"].mean() * 1e3), 3
        ),
        "servol_duration_max_ms": round(
            float(arrays["servol_duration"].max() * 1e3), 3
        ),
        "gravity_k": data["gravity_k"],
        "friction": data["friction"],
        "hold_phase_oscillation": analyze_hold_phases(
            arrays, rate, data["joint_names"]
        ),
    }

    tag = args.input.stem.replace("servol_", "")
    report_path = args.input.parent / f"servol_{tag}_report.json"
    curves_path = args.input.parent / f"servol_{tag}_curves.png"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    plot_curves(arrays, data, curves_path)

    print(f"报告 -> {report_path}")
    print(f"曲线 -> {curves_path}")
    print(f"\nIK 失败 {report['ik_failures']} 次, "
          f"迭代均值 {report['ik_iterations_mean']}, "
          f"servoL 耗时均值 {report['servol_duration_mean_ms']} ms")


if __name__ == "__main__":
    main()

