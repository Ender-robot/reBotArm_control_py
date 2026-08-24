"""将拖动轨迹重整为关节空间等速轨迹。

运行：python reshape_traj.py data/drag_<tag>.json
输出：data/traj_<tag>.json
"""

import argparse
import json
from pathlib import Path
import numpy as np


def moving_average(data, window_size):
    """简单移动平均滤波。"""
    if window_size <= 1:
        return data

    # 对每个关节独立滤波
    result = np.zeros_like(data)
    for j in range(data.shape[1]):
        result[:, j] = np.convolve(data[:, j], np.ones(window_size) / window_size, mode='same')

    return result


def reshape_trajectory(
    samples: list,
    v_target: float = 0.09,
    dt: float = 0.033,
    smooth_window: int = 5,
) -> dict:
    """重整为等弧长采样的轨迹。

    Args:
        samples: [{"t": float, "q": [6]}, ...]
        v_target: 目标关节速度 (rad/s)
        dt: 输出轨迹的时间步长 (s)
        smooth_window: 滤波窗口大小（样本数）

    Returns:
        {"dt": float, "v_target": float, "q": [[6], ...]}
    """
    q_raw = np.array([s["q"] for s in samples])

    # 1. 低通滤波去抖动
    if smooth_window > 1:
        q_smooth = moving_average(q_raw, smooth_window)
    else:
        q_smooth = q_raw

    # 2. 计算累积弧长
    dq = np.diff(q_smooth, axis=0)
    ds = np.linalg.norm(dq, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])

    # 3. 等间距重采样
    s_total = s[-1]
    if s_total < 1e-6:
        # 轨迹几乎静止，返回起点和终点
        return {
            "dt": dt,
            "v_target": v_target,
            "q": [q_smooth[0].tolist(), q_smooth[-1].tolist()],
        }

    delta_s = v_target * dt
    n_steps = int(np.ceil(s_total / delta_s))
    s_new = np.linspace(0, s_total, n_steps + 1)

    # 对每个关节独立线性插值
    q_new = np.zeros((len(s_new), 6))
    for j in range(6):
        q_new[:, j] = np.interp(s_new, s, q_smooth[:, j])

    return {
        "dt": dt,
        "v_target": v_target,
        "q": q_new.round(6).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="重整拖动轨迹为等速")
    parser.add_argument("input", type=Path, help="输入 drag_<tag>.json")
    parser.add_argument("--v-target", type=float, default=0.1, help="目标速度 (rad/s)")
    parser.add_argument("--dt", type=float, default=0.033, help="输出时间步长 (s)")
    parser.add_argument("--smooth", type=int, default=5, help="滤波窗口大小")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    result = reshape_trajectory(
        data["samples"],
        v_target=args.v_target,
        dt=args.dt,
        smooth_window=args.smooth,
    )

    # 输出到同目录，drag_ 改为 traj_
    tag = args.input.stem.replace("drag_", "")
    output = args.input.parent / f"traj_{tag}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"重整完成:")
    print(f"  输入样本数: {len(data['samples'])}")
    print(f"  输出路点数: {len(result['q'])}")
    print(f"  目标速度: {args.v_target} rad/s")
    print(f"  时间步长: {args.dt} s")
    print(f"  总时长: {len(result['q']) * args.dt:.2f} s")
    print(f"  已保存到 {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
