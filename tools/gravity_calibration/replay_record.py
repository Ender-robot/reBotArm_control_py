"""按轨迹路径点静态保持并录制重力标定数据。

轨迹由 generate_calibration_traj.py 生成，本脚本只改变回放方式：
每隔 POINT_STRIDE 个路径点到位后静态保持，采集一段反馈数据；正反各一遍，
供拟合脚本按 waypoint_index 配对平均，以降低方向相关摩擦的影响。

运行：修改下方路径后直接运行本文件
输出：data/sweep_static_<tag>.json
"""

import json
import time
from pathlib import Path

import numpy as np

from reBotArm_control_py.actuator import RebotArm


DATA_DIR = Path(__file__).parent / "data"
TRAJECTORY_PATH = DATA_DIR / "traj_No1.json"
OUTPUT_PATH = DATA_DIR / "sweep_static_No1.json"

POINT_STRIDE = 50       # 每隔多少个生成轨迹点采集一个静态点
SETTLE_TIME = 0.5       # 到位且速度稳定后，额外等待时间 (s)
HOLD_TIME = 1.0         # 稳定后的静态采样时间 (s)
SAMPLE_RATE = 30.0      # 静态采样频率 (Hz)
POSITION_TOLERANCE = 0.005  # 到位位置误差 (rad)
VELOCITY_TOLERANCE = 0.02   # 稳定速度阈值 (rad/s)
SETTLE_TIMEOUT = 10.0       # 单个路径点最大等待时间 (s)
V_LIMIT_SCALE = 1.5

MAX_TORQUE = 25.0  # N·m
FEEDBACK_FROZEN_CYCLES = 30


class SafetyAbort(Exception):
    """安全监控触发中止。"""
    pass


def select_waypoints(waypoints, reverse=False):
    """按固定步长选择路径点，并始终保留最后一个点。"""
    if len(waypoints) == 0:
        return []

    indices = list(range(0, len(waypoints), POINT_STRIDE))
    if indices[-1] != len(waypoints) - 1:
        indices.append(len(waypoints) - 1)
    selected = [(index, waypoints[index]) for index in indices]
    return selected[::-1] if reverse else selected


def check_safety(samples, current_tau, check_frozen=True):
    """安全检查：力矩超限、反馈冻结。"""
    if np.abs(current_tau).max() > MAX_TORQUE:
        joint_id = int(np.argmax(np.abs(current_tau)))
        raise SafetyAbort(
            f"力矩超限: joint{joint_id+1} = {current_tau[joint_id]:.2f} N·m "
            f"(上限 {MAX_TORQUE} N·m)"
        )

    # 静态保持时 q 和 tau 稳定是正常现象，不能判定为反馈冻结。
    if check_frozen and len(samples) > FEEDBACK_FROZEN_CYCLES:
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

    timeout = distance / v_move * 2.0 + 5.0
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        current_q = rebotarm.arm_state.feedback.arm.position.copy()
        if np.linalg.norm(start_q - current_q) < 0.005:
            print("已到达起点")
            return
        time.sleep(0.05)

    raise SafetyAbort(f"归位超时 ({timeout:.1f} s)")


def _send_waypoint(rebotarm, q_cmd, vlim):
    """发送一个位置保持命令。"""
    rebotarm.arm_state.command.arm.position[:] = q_cmd
    rebotarm.arm_state.command.arm.velocity[:] = vlim
    rebotarm._joint_command_ready.set()


def _wait_until_settled(rebotarm, q_cmd, samples):
    """等待实际位置和速度达到静态采样条件。"""
    deadline = time.monotonic() + SETTLE_TIMEOUT
    while time.monotonic() < deadline:
        feedback = rebotarm.arm_state.feedback.arm
        q = feedback.position.copy()
        v = feedback.velocity.copy()
        tau = feedback.torque.copy()
        check_safety(samples, tau, check_frozen=False)

        position_ok = np.max(np.abs(q - q_cmd)) <= POSITION_TOLERANCE
        velocity_ok = np.max(np.abs(v)) <= VELOCITY_TOLERANCE
        if position_ok and velocity_ok:
            time.sleep(SETTLE_TIME)
            feedback = rebotarm.arm_state.feedback.arm
            q = feedback.position.copy()
            v = feedback.velocity.copy()
            if (
                np.max(np.abs(q - q_cmd)) <= POSITION_TOLERANCE
                and np.max(np.abs(v)) <= VELOCITY_TOLERANCE
            ):
                return

        time.sleep(0.02)

    raise SafetyAbort(f"路径点稳定超时 ({SETTLE_TIMEOUT:.1f} s)")


def _record_hold(rebotarm, waypoint_index, q_cmd, vlim, pass_name, samples):
    """在一个路径点保持并采集静态样本。"""
    _send_waypoint(rebotarm, q_cmd, vlim)
    _wait_until_settled(rebotarm, q_cmd, samples)

    sample_count = max(1, round(HOLD_TIME * SAMPLE_RATE))
    period = 1.0 / SAMPLE_RATE
    start_time = time.monotonic()
    deadline = start_time

    for _ in range(sample_count):
        now = time.monotonic()
        if now < deadline:
            time.sleep(deadline - now)

        feedback = rebotarm.arm_state.feedback.arm
        q = feedback.position.copy()
        v = feedback.velocity.copy()
        tau = feedback.torque.copy()
        sample = {
            "t": round(time.monotonic() - start_time, 6),
            "pass": pass_name,
            "waypoint_index": waypoint_index,
            "q": q.round(6).tolist(),
            "q_cmd": q_cmd.round(6).tolist(),
            "v": v.round(6).tolist(),
            "tau": tau.round(6).tolist(),
        }
        samples.append(sample)
        check_safety(samples, tau, check_frozen=False)
        deadline += period


def replay_static_pass(rebotarm, waypoints, vlim, pass_name, reverse=False):
    """按选定路径点静态保持一遍。"""
    samples = []
    selected = select_waypoints(waypoints, reverse=reverse)
    for point_number, (waypoint_index, q_cmd) in enumerate(selected, start=1):
        _record_hold(
            rebotarm,
            waypoint_index,
            q_cmd,
            vlim,
            pass_name,
            samples,
        )
        print(
            f"  {pass_name}: {point_number}/{len(selected)} 点 - "
            f"waypoint {waypoint_index}"
        )
    return samples


def replay_and_record(rebotarm, trajectory):
    """正反向静态回放并录制数据。"""
    waypoints = np.array(trajectory["q"])
    v_target = trajectory["v_target"]
    vlim = v_target * V_LIMIT_SCALE

    rebotarm.arm.mode_pos_vel()
    time.sleep(0.5)

    move_to_start(rebotarm, waypoints[0], v_move=v_target * 0.5)
    time.sleep(1.0)

    print("正向静态采样...")
    samples_forward = replay_static_pass(
        rebotarm, waypoints, vlim, "forward",
    )

    time.sleep(1.0)

    print("反向静态采样...")
    samples_reverse = replay_static_pass(
        rebotarm, waypoints, vlim, "reverse", reverse=True,
    )

    return {"samples": samples_forward + samples_reverse}


def main():
    trajectory = json.loads(TRAJECTORY_PATH.read_text())

    rebotarm = RebotArm(mode="posvel")
    try:
        rebotarm.connect()
        selected_count = len(select_waypoints(np.array(trajectory["q"])))
        print(f"轨迹路点数: {len(trajectory['q'])}")
        print(f"静态采样点数: {selected_count}")
        print(f"路径点步长: {POINT_STRIDE}")
        print(f"稳定等待: {SETTLE_TIME:.2f} s")
        print(f"保持采样: {HOLD_TIME:.2f} s @ {SAMPLE_RATE:.1f} Hz")
        print()

        result = replay_and_record(rebotarm, trajectory)
        OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n已保存 {len(result['samples'])} 个样本到 {OUTPUT_PATH}")

        print()
        move_to_home(rebotarm, v_move=0.05)

    except SafetyAbort as error:
        print(f"\n安全中止: {error}")
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
