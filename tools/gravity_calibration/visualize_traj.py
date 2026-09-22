"""Web 可视化标定轨迹。

使用 MeshCat 在浏览器中预览轨迹，支持局域网访问。

运行：python visualize_traj.py data/traj_<tag>.json
访问：打开终端显示的 URL（局域网内可访问）
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from example.sim.visualizer import Visualizer


def visualize_trajectory(trajectory_path: Path, play_speed: float = 1.0):
    """在 MeshCat 中可视化轨迹。

    Args:
        trajectory_path: 轨迹 JSON 文件路径
        play_speed: 播放速度倍率（1.0 = 实时）
    """
    # 加载轨迹
    print(f"加载轨迹: {trajectory_path}")
    trajectory = json.loads(trajectory_path.read_text())
    q_list = [np.array(q) for q in trajectory["q"]]
    dt = trajectory["dt"]
    v_target = trajectory["v_target"]

    print(f"  路点数: {len(q_list)}")
    print(f"  目标速度: {v_target} rad/s")
    print(f"  时间步长: {dt} s")
    print(f"  预计时长: {len(q_list) * dt:.1f} s")
    print()

    # 创建可视化器
    print("启动 MeshCat 可视化器...")
    viz = Visualizer(open_browser=True)
    print()
    print("=" * 60)
    print("可视化器已启动！")
    print(f"访问地址: {viz.meshcat.url()}")
    print("=" * 60)
    print()

    # 计算末端轨迹路径
    print("计算末端轨迹...")
    from reBotArm_control_py.kinematics import compute_fk

    ee_path = []
    for q in q_list:
        _, _, T = compute_fk(viz.model, q)
        ee_path.append(T[:3, 3].tolist())

    # 显示起点
    print("显示起点姿态...")
    viz.update(q_list[0])
    viz.draw_ref_path(ee_path)
    time.sleep(2.0)

    # 播放轨迹
    print(f"\n开始播放轨迹（速度 {play_speed}x）...")
    print("按 Ctrl+C 停止播放")
    print()

    try:
        viz.play_trajectory(
            name=trajectory_path.stem,
            dt=dt / play_speed,
            q_list=q_list,
            path=ee_path,
        )

        print("\n轨迹播放完成！")
        print("可视化器将保持运行，按 Ctrl+C 退出。")

        # 保持运行
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n用户中断，退出。")


def main():
    parser = argparse.ArgumentParser(description="Web 可视化标定轨迹")
    parser.add_argument("input", type=Path, help="输入 traj_<tag>.json")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="播放速度倍率（1.0 = 实时）",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误: 文件不存在: {args.input}")
        return 1

    visualize_trajectory(args.input, play_speed=args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
