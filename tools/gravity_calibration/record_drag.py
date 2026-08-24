"""录制手动拖动轨迹（电机不使能，纯编码器读数）。

运行：python record_drag.py --duration 60 --tag demo
输出：data/drag_<tag>.json
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

from reBotArm_control_py.actuator import RebotArm


def record_drag(rebotarm, duration: float, rate: float = 30.0) -> dict:
    """录制拖动轨迹。

    Args:
        rebotarm: 已连接的 RebotArm 实例（电机不使能）
        duration: 录制时长（秒）
        rate: 采样频率（Hz）

    Returns:
        {"rate": float, "samples": [{"t": float, "q": [6 floats]}, ...]}
    """
    period = 1.0 / rate
    samples = []
    start_time = time.monotonic()
    deadline = start_time

    while time.monotonic() - start_time < duration:
        now = time.monotonic()
        if now < deadline:
            time.sleep(max(0.0, deadline - now))

        q = rebotarm.arm.get_positions()
        samples.append({
            "t": round(time.monotonic() - start_time, 6),
            "q": [round(float(x), 6) for x in q],
        })
        deadline += period

    return {"rate": rate, "samples": samples}


def print_motion_summary(samples):
    """打印各关节累计行程。"""
    q_array = np.array([s["q"] for s in samples])
    travel = np.sum(np.abs(np.diff(q_array, axis=0)), axis=0)
    print("\n各关节累计行程 (rad):")
    for j in range(6):
        print(f"  joint{j+1}: {travel[j]:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="录制手动拖动轨迹")
    parser.add_argument("--duration", type=float, default=60.0, help="录制时长（秒）")
    parser.add_argument("--rate", type=float, default=30.0, help="采样频率（Hz）")
    parser.add_argument("--tag", required=True, help="输出文件标签")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data", help="输出目录")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"drag_{args.tag}.json"

    # 先用 posvel 连接，再切换到 MIT 模式
    rebotarm = RebotArm(mode="posvel")
    try:
        rebotarm.connect()

        # 切换到 MIT 模式，设置 kp=0, kd=0 使电机可自由拖动
        rebotarm.arm.mode_mit(kp=np.zeros(6), kd=np.zeros(6))
        time.sleep(0.5)

        print(f"开始录制 {args.duration:.1f} 秒，采样 {args.rate:.0f} Hz")
        print("请手动缓慢拖动机械臂...\n")

        result = record_drag(rebotarm, args.duration, args.rate)

        print_motion_summary(result["samples"])

        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"已保存 {len(result['samples'])} 个样本到 {output_path}")

    finally:
        rebotarm.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
