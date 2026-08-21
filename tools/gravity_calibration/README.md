# DM 机械臂重力标定

本目录使用当前项目配置完成 DM 版本机械臂的重力辨识和悬浮验证：

- 硬件配置由 `config/rebotarm.yaml` 选择；
- 电机、串口、MIT 增益和关节分组来自对应 hardware YAML；
- 重力模型和关节限制来自同一配置指定的 URDF；
- 只有 joint2、joint3、joint4、joint5 会进入 MIT 并使能；
- joint1、joint6 和 gripper 在整个标定过程中保持未使能；
- 新数据写入 `data/dm/`。

默认安全力矩上限为 URDF effort 的 50%。当前模型对应：

| 关节 | 默认安全力矩上限 |
|---|---:|
| joint2、joint3 | 13.5 N·m |
| joint4、joint5 | 3.5 N·m |

标定过程会使用实时关节位置计算重力前馈。PD 扫描的主动关节不发送
重力前馈，其估算力矩为 `kp * (target - q) - kd * velocity`。安全中止时
先在当前位置保持，再返回启动姿态、逐步卸载并失能。

## 安全准备

1. 将机械臂放在不会自碰、不会碰桌面、线缆不受拉扯的空间。
2. 准备急停，并在首次运行时使用较小运动范围和较低速度。
3. 确认 `config/rebotarm.yaml` 当前指向正确的 DM hardware YAML。
4. 所有 `--end`、`--pre`、`--pose` 都是绝对关节角，不是增量。
5. 示例角度只演示命令形式；必须根据实际启动姿态和无碰撞空间重新选取。

## 执行顺序

以下命令均从项目 `thirdparty/reBotArm_control_py/` 目录执行。

### 1. 无力矩方向检查

```bash
../../.venv/bin/python tools/gravity_calibration/phase_a_sign_probe.py 120
```

电机保持未使能。一次只手动移动一个关节，确认打印的 joint1～joint6
与实际关节对应，反馈连续且方向与当前 URDF 约定一致。

### 2. joint3 小范围首次扫描

先从 joint3 开始，选择当前姿态附近一个安全的绝对终点。例如：

```bash
../../.venv/bin/python tools/gravity_calibration/pd_sweep_id.py \
  --joint joint3 \
  --end -0.5 \
  --speed 0.08 \
  --tag first
```

脚本按照“当前角度 → `--end` → 当前角度”扫描。若需要先把其他负载关节
移到无碰撞姿态，可重复传入 `--pre`：

```bash
../../.venv/bin/python tools/gravity_calibration/pd_sweep_id.py \
  --joint joint3 \
  --end -0.8 \
  --pre joint2=-0.7 \
  --pre joint4=0.3
```

### 3. 拟合 joint3

```bash
../../.venv/bin/python tools/gravity_calibration/fit_sweeps.py \
  "tools/gravity_calibration/data/dm/pdsweep_joint3_*.json"
```

输出含义：

| 字段 | 含义 |
|---|---|
| `off` | 电机零点相对 URDF 的角度偏移 |
| `k` | URDF 重力力矩比例修正 |
| `bias` | 常量力矩偏置 |
| `fric` | 正反向扫描估算的库仑摩擦项 |
| `rms` | 拟合残差 |

若起始区域可能接触桌面，可用 `--qmin` 排除该区域。也可用
`--urdf path/to/model.urdf` 离线比较其他模型；省略时始终加载当前配置。

### 4. 扫描其余负载关节

确认 joint3 流程和安全回退正常后，依次扫描：

1. joint2
2. joint4
3. joint5

每个关节都先用小范围、低速度扫描，再逐步扩大覆盖范围。不要扫描 joint1
或 joint6：当前流程只针对承受主要重力负载的 joint2～joint5。

### 5. 汇总拟合

```bash
../../.venv/bin/python tools/gravity_calibration/fit_sweeps.py \
  "tools/gravity_calibration/data/dm/pdsweep_joint2_*.json" \
  "tools/gravity_calibration/data/dm/pdsweep_joint3_*.json" \
  "tools/gravity_calibration/data/dm/pdsweep_joint4_*.json" \
  "tools/gravity_calibration/data/dm/pdsweep_joint5_*.json"
```

同一关节应尽量使用多个无接触姿态的扫描结果交叉检查。MIT kp 的实际力矩
比例误差会直接反映到拟合的 `k` 中，因为这里没有独立力矩传感器。

### 6. joint3 悬浮验证

把拟合得到的 `k`、`bias` 和 `off` 分别作为 `--k`、`--c` 和
`--offset`。先验证 joint3：

```bash
../../.venv/bin/python tools/gravity_calibration/auto_float_test.py \
  --joint joint3 \
  --pose -0.5 \
  --float-s 12 \
  --k 1.0 \
  --c 0.0 \
  --offset 0.0
```

脚本会从启动姿态移动到绝对 `--pose`，逐步去掉 kp，保留 kd 和重力
前馈，最后返回启动姿态。重点观察漂移量和积分残差；首次验证仍应缩短
`--float-s` 并随时准备急停。

### 7. 验证其余负载关节

joint3 验证稳定后，再按 joint2、joint4、joint5 的顺序分别代入各自拟合
结果。每次只验证一个主动关节，其余负载关节由实时重力前馈和配置中的
MIT PD 增益保持。

## 脚本说明

| 脚本 | 功能 |
|---|---|
| `phase_a_sign_probe.py` | 未使能状态下记录六轴位置和运动方向 |
| `pd_sweep_id.py` | 单关节绝对目标往返 PD 扫描并记录估算力矩 |
| `fit_sweeps.py` | 将扫描 JSON 拟合到当前 URDF 重力模型 |
| `auto_float_test.py` | 单关节 PD 淡出和重力前馈悬浮验证 |
| `dm_calibration.py` | DM 配置、逐电机使能、力矩限制和模型计算适配层 |
