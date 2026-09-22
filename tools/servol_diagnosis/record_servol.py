"""录制 servoL(mit) 诊断数据。

复用 example/2_test_servoL.py 的虚拟路径生成逻辑, 但在每个控制周期
记录进出 servoL 的全部中间量, 用于定位 mit 模式抖动来源。

运行：python record_servol.py --tag No1
输出：data/servol_<tag>.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

from reBotArm_control_py.controllers import RebotArmController
from reBotArm_control_py.controllers.process_command import compute_scale_factor
from reBotArm_control_py.kinematics import (
    compute_fk,
    load_robot_model,
    pad_q_for_model,
    solve_ik,
    xyz_quat_to_se3,
)


# 与 example/2_test_servoL.py 保持一致的运动参数
INTEGRATION_STEP = 0.003
Z_FINAL_INTEGRATION_DISTANCE = 0.15
X_FINAL_INTEGRATION_DISTANCE = 0.15
Z_HOLD_TIME = 2.0
SERVOL_SPEED = [1.5, 3.1, 3.1, 3.1, 1.5, 1.5]


def build_targets(initial_position, quaternion, step):
    """生成与 2_test_servoL.py 相同的虚拟路径, 附带阶段标签。"""
    targets = []

    def append(offset, phase):
        target = np.concatenate((initial_position + offset, quaternion))
        targets.append((target, phase))

    distance = 0.0
    while distance < Z_FINAL_INTEGRATION_DISTANCE:
        distance = min(distance + step, Z_FINAL_INTEGRATION_DISTANCE)
        append(np.array([0.0, 0.0, distance]), "z_move")

    z_offset = np.array([0.0, 0.0, Z_FINAL_INTEGRATION_DISTANCE])
    for _ in range(int(Z_HOLD_TIME * 30.0)):
        append(z_offset, "z_hold")

    distance = 0.0
    while distance < X_FINAL_INTEGRATION_DISTANCE:
        distance = min(distance + step, X_FINAL_INTEGRATION_DISTANCE)
        append(z_offset + np.array([distance, 0.0, 0.0]), "x_move")

    final_offset = z_offset + np.array([X_FINAL_INTEGRATION_DISTANCE, 0.0, 0.0])
    for _ in range(int(Z_HOLD_TIME * 30.0)):
        append(final_offset, "x_hold")

    return targets


def record_sample(controller, target, phase, lookahead, speed, start_time):
    """执行一次 servoL 并记录该周期的全部中间量。

    servoL 内部不暴露中间量, 因此这里在调用前后各采一次状态, 并用与
    servoL 完全相同的输入重算一遍 IK, 得到未经滤波的原始前馈作为对照。
    """
    arm_state = controller.rebotarm.arm_state
    command = arm_state.command

    # servoL 内部会用实时反馈做 q_init, 这里先记下调用前的反馈
    q_before = arm_state.feedback.arm.position.copy()
    qdot_before = arm_state.feedback.arm.velocity.copy()
    torque_before = arm_state.feedback.arm.torque.copy()
    ema_before = None if controller.ema.value is None else controller.ema.value.copy()
    t_call = time.monotonic()

    controller.servoL(target, speed, lookahead)

    # 用调用前的反馈重算原始前馈, 与实际下发的滤波后速度对照
    ik_result = solve_ik(
        controller._model,
        controller._data,
        controller._end_frame_id,
        xyz_quat_to_se3(target),
        q_before,
        controller.servoL_ik_params,
    )
    qdot_raw = (ik_result.q - q_before) / lookahead
    vel_scale = compute_scale_factor(qdot_raw, speed)

    return {
        "t": round(t_call - start_time, 6),
        "phase": phase,
        "target_xyz": [round(float(x), 6) for x in target[:3]],
        "q_feedback": [round(float(x), 6) for x in q_before],
        "qdot_feedback": [round(float(x), 6) for x in qdot_before],
        "torque_feedback": [round(float(x), 6) for x in torque_before],
        "q_ik": [round(float(x), 6) for x in ik_result.q],
        "ik_success": bool(ik_result.success),
        "ik_iterations": int(ik_result.iterations),
        "ik_position_error": round(float(ik_result.position_error), 8),
        # 位置误差 delta 是寄生刚度的来源: tau 中含 (kd/lookahead)*delta
        "delta": [round(float(x), 8) for x in (ik_result.q - q_before)],
        "qdot_raw": [round(float(x), 6) for x in qdot_raw],
        "vel_scale_raw": round(float(vel_scale), 6),
        "ema_state_before": None if ema_before is None else
            [round(float(x), 6) for x in ema_before],
        # 实际下发给电机的指令
        "cmd_position": [round(float(x), 6) for x in command.arm.position],
        "cmd_velocity": [round(float(x), 6) for x in command.arm.velocity],
        "cmd_timestamp": round(float(command.timestamp), 6),
        "controller_state": str(controller._controller_state),
        "servol_duration": round(time.monotonic() - t_call, 6),
    }


def record(controller, rate, lookahead, alpha):
    """跑一遍虚拟路径并逐周期录制。"""
    controller.ema.alpha = alpha
    controller.ema.reset()

    feedback = controller.rebotarm.arm_state.feedback
    while feedback.timestamp == 0.0:
        time.sleep(1.0 / rate)

    model = load_robot_model()
    joint_position = pad_q_for_model(model, feedback.arm.position.copy())
    position, rotation, _ = compute_fk(model, joint_position)
    quaternion = pin.Quaternion(rotation).coeffs()

    targets = build_targets(position.copy(), quaternion, abs(INTEGRATION_STEP))
    speed = np.array(SERVOL_SPEED)
    period = 1.0 / rate

    samples = []
    aborted = False
    start_time = time.monotonic()
    deadline = start_time
    try:
        for target, phase in targets:
            now = time.monotonic()
            if now < deadline:
                time.sleep(deadline - now)
            samples.append(
                record_sample(controller, target, phase, lookahead, speed, start_time)
            )
            deadline += period
    except KeyboardInterrupt:
        # 抖动严重时无法跑完全程, 已采到的部分同样有分析价值
        aborted = True
        print(f"\n收到 Ctrl+C, 保留已录制的 {len(samples)} 个采样点")

    arm_group = controller.rebotarm.groups["arm"]
    return {
        "rate": rate,
        "aborted": aborted,
        "planned_samples": len(targets),
        "lookahead": lookahead,
        "alpha": alpha,
        "speed": [float(x) for x in speed],
        "joint_names": list(arm_group.joint_names),
        # kd/lookahead 是寄生刚度, 与 kp 同量级即说明前馈在充当额外 kp
        "kp": [float(x) for x in arm_group._mit_kp],
        "kd": [float(x) for x in arm_group._mit_kd],
        "gravity_k": [float(x) for x in arm_group._gravity_k],
        "friction": [float(x) for x in arm_group._friction],
        "tau_bias": [float(x) for x in arm_group._tau_bias],
        "comm_rate": 1.0 / controller.rebotarm._comm_period,
        "initial_tcp": [float(x) for x in position],
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="录制 servoL(mit) 诊断数据")
    parser.add_argument("--tag", required=True, help="输出文件标签")
    parser.add_argument("--rate", type=float, default=30.0, help="控制频率 (Hz)")
    parser.add_argument("--lookahead", type=float, default=0.033, help="前馈时间常数 (s)")
    parser.add_argument("--alpha", type=float, default=0.6, help="EMA 平滑系数")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="输出目录",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"servol_{args.tag}.json"

    time.sleep(5)
    controller = RebotArmController("mit")
    try:
        controller.connect()
        result = record(controller, args.rate, args.lookahead, args.alpha)
        # 先落盘再回零: 回零耗时数秒, 期间的意外不应连带丢掉数据
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        status = "中断" if result["aborted"] else "完整"
        print(f"{status}录制 {len(result['samples'])}/"
              f"{result['planned_samples']} 个采样点 -> {output_path}")
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C, 尚未开始录制")
    finally:
        time.sleep(1.0) # 等上一条 servoL 的控制权超时释放
        if controller.home(): # disconnect 会掉力矩, 回零失败就不能断开
            controller.disconnect()
        else:
            print("回零失败, 保持通电, 请手动扶住机械臂后再断开")


if __name__ == "__main__":
    main()
