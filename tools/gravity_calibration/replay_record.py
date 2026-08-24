"""posvel 回放重整轨迹，正反各一遍，录制力矩数据。

运行：python replay_record.py data/traj_<tag>.json
输出：data/sweep_<tag>.json
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

from reBotArm_control_py.actuator import RebotArm


MAX_TORQUE = 25.0  # N·m
FEEDBACK_FROZEN_CYCLES = 30


class SafetyAbort(Exception):
    """安全监控触发中止。"""
    pass


def check_safety(samples, current_tau):
    """安全检查：力矩超限、反馈冻结。"""
    # 力矩超限
    if np.abs(current_tau).max() > MAX_TORQUE:
        joint_id = int(np.argmax(np.abs(current_tau)))
        raise SafetyAbort(
            f"力矩超限: joint{joint_id+1} = {current_tau[joint_id]:.2f} N·m "
            f"(上限 {MAX_TORQUE} N·m)"
        )

    # 反馈冻结检查
    if len(samples) > FEEDBACK_FROZEN_CYCLES:
        window = samples[-FEEDBACK_FROZEN_CYCLES:]
        q_diff = np.abs(np.diff([s["q"] for s in window], axis=0)).max()
        tau_diff = np.abs(np.diff([s["tau"] for s in window], axis=0)).max()
        if q_diff < 1e-9 and tau_diff < 1e-9:
            raise SafetyAbort(
                f"反馈数值冻结超过 {FEEDBACK_FROZEN_CYCLES} 个采样 "
                "(电机断电或通信中断)"
            )


def move_to_home(rebotarm, v_move=0.05):
    """返回零点（全零位置）。"""
    q_home = np.zeros(6)
    q_current = rebotarm.arm_state.feedback.arm.position.copy()
    distance = np.linalg.norm(q_home - q_current)

    if distance < 0.01:
        print("已在零点附近，跳过回零")
        return

    print(f"返回零点 (距离 {distance:.4f} rad，速度 {v_move:.3f} rad/s)...")

    rebotarm.arm_state.command.arm.position[:] = q_home
    rebotarm.arm_state.command.arm.velocity[:] = v_move
    rebotarm._joint_command_ready.set()

    # 等待到位
    timeout = distance / v_move * 2.0 + 10.0
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        q_current = rebotarm.arm_state.feedback.arm.position.copy()
        if np.linalg.norm(q_home - q_current) < 0.01:
            print("已到达零点")
            return
        time.sleep(0.1)

    print("警告: 回零超时，但会继续断开")


def move_to_start(rebotarm, start_q, v_move):
    """以慢速移动到轨迹起点。"""
    current_q = rebotarm.arm_state.feedback.arm.position.copy()
    distance = np.linalg.norm(start_q - current_q)

    if distance < 0.01:
        print(f"当前位置已接近起点 (距离 {distance:.4f} rad)，跳过归位")
        return

    print(f"归位到起点 (距离 {distance:.4f} rad，速度 {v_move:.3f} rad/s)...")

    rebotarm.arm_state.command.arm.position[:] = start_q
    rebotarm.arm_state.command.arm.velocity[:] = v_move
    rebotarm._joint_command_ready.set()

    # 等待到位
    timeout = distance / v_move * 2.0 + 5.0  # 理论时间的2倍 + 5秒余量
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        current_q = rebotarm.arm_state.feedback.arm.position.copy()
        if np.linalg.norm(start_q - current_q) < 0.005:
            print("已到达起点")
            return
        time.sleep(0.05)

    raise SafetyAbort(f"归位超时 ({timeout:.1f} s)")


def replay_pass(rebotarm, waypoints, dt, vlim, pass_name):
    """回放一遍轨迹（正序或反序）。"""
    samples = []
    start_time = time.monotonic()

    for i, q_cmd in enumerate(waypoints):
        deadline = start_time + i * dt
        now = time.monotonic()
        if now < deadline:
            time.sleep(max(0.0, deadline - now))

        # 发送命令
        rebotarm.arm_state.command.arm.position[:] = q_cmd
        rebotarm.arm_state.command.arm.velocity[:] = vlim
        rebotarm._joint_command_ready.set()

        # 等待一小段让电机响应
        time.sleep(dt * 0.3)

        # 采样
        feedback = rebotarm.arm_state.feedback
        q = feedback.arm.position.copy()
        v = feedback.arm.velocity.copy()
        tau = feedback.arm.torque.copy()

        sample = {
            "t": round(time.monotonic() - start_time, 6),
            "pass": pass_name,
            "q": q.round(6).tolist(),
            "q_cmd": [round(float(x), 6) for x in q_cmd],
            "v": v.round(6).tolist(),
            "tau": tau.round(6).tolist(),
        }
        samples.append(sample)

        check_safety(samples, tau)

        if (i + 1) % 10 == 0:
            print(f"  {pass_name}: {i+1}/{len(waypoints)} 点", end="\r")

    print(f"  {pass_name}: {len(waypoints)}/{len(waypoints)} 点 - 完成")
    return samples


def replay_and_record(rebotarm, trajectory: dict, vlim_scale: float = 1.5) -> dict:
    """回放轨迹并录制数据。

    Args:
        rebotarm: 已连接的 RebotArm 实例
        trajectory: {"dt": float, "v_target": float, "q": [[6], ...]}
        vlim_scale: vlim = v_target * vlim_scale

    Returns:
        {"samples": [{"t", "pass", "q", "q_cmd", "v", "tau"}, ...]}
    """
    waypoints = np.array(trajectory["q"])
    dt = trajectory["dt"]
    v_target = trajectory["v_target"]
    vlim = v_target * vlim_scale

    # 切换到 posvel 模式
    rebotarm.arm.mode_pos_vel()
    time.sleep(0.5)

    # 归位
    move_to_start(rebotarm, waypoints[0], v_move=v_target * 0.5)
    time.sleep(1.0)

    # 正序回放
    print("正序回放...")
    samples_forward = replay_pass(rebotarm, waypoints, dt, vlim, "forward")

    time.sleep(1.0)

    # 反序回放
    print("反序回放...")
    samples_reverse = replay_pass(rebotarm, waypoints[::-1], dt, vlim, "reverse")

    return {"samples": samples_forward + samples_reverse}


def main():
    parser = argparse.ArgumentParser(description="回放轨迹并录制力矩")
    parser.add_argument("input", type=Path, help="输入 traj_<tag>.json")
    parser.add_argument("--vlim-scale", type=float, default=1.5, help="vlim = v_target * scale")
    args = parser.parse_args()

    trajectory = json.loads(args.input.read_text())

    tag = args.input.stem.replace("traj_", "")
    output = args.input.parent / f"sweep_{tag}.json"

    rebotarm = RebotArm(mode="posvel")
    try:
        rebotarm.connect()
        print(f"轨迹路点数: {len(trajectory['q'])}")
        print(f"目标速度: {trajectory['v_target']} rad/s")
        print(f"vlim: {trajectory['v_target'] * args.vlim_scale:.3f} rad/s")
        print()

        result = replay_and_record(rebotarm, trajectory, args.vlim_scale)

        output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n已保存 {len(result['samples'])} 个样本到 {output}")

        # 返回零点
        print()
        move_to_home(rebotarm, v_move=0.05)

    except SafetyAbort as e:
        print(f"\n安全中止: {e}")
        print("尝试返回零点...")
        try:
            move_to_home(rebotarm, v_move=0.05)
        except Exception as home_error:
            print(f"回零失败: {home_error}")
            print("电机保持当前位置，请手动处理后按 Ctrl+C 断开")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
        return 1

    finally:
        rebotarm.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
