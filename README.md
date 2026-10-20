# reBotArm 控制接口

## 通信与命令交接

`RebotArmController` 原有的 `connect/disconnect`、`servoJ`、`home`、FK、
清错及 MIT 增益设置接口保持原调用方式，新增 `hold()` 保持机械臂，夹爪独立控制。

公共控制器只做控制权管理和 IK，通过驱动的 `submit_arm`、
`submit_servo_target`、`submit_gripper`、`hold` 提交完整命令。
应用代码不再直接修改命令缓冲区或操作通知事件。

- `actuator/rebotarm.py`：硬件配置、连接、关节分组、反馈及原子命令提交。
- `actuator/arm_state.py`、`controllers/controller_state.py`：各自所在层的状态定义。
- `actuator/mit_communicator.py`、`posvel_communicator.py`：各模式的后台状态和发送。
- `actuator/CommunicationBase.py`：唯一通信线程的生命周期和周期调度。
- `controllers/control_params.py`：控制器与后台共享的软件默认参数。
- `trajectory/mit_reference.py`：纯数值参考算法，不操作硬件。

两种模式都只保留最新命令，不建立运动队列。命令写入、通知和消费快照
共用一把锁；参考推进、动力学计算和硬件发送在锁外进行。
MIT 每周期发送六轴参考，posvel 在有新命令时发送，夹爪按独立通知发送。
通信频率和显式反馈请求策略保持原设置；断开、重连清除待发命令。

硬件参数只从实际选定的 YAML 加载一次。软件默认参数集中在
`controller.params`，与后台共享；`servoL_ik_params` 和
`servoL_reference_params` 分别引用其中的 IK、参考参数对象。
在连接前配置默认参数，避免后台运行时并发修改参数对象。

## MIT `servoL` 参考生成

`servoL(target, speed, *, acceleration=None, frequency_hz=None)`
兼容原有两参数调用：`target` 是七数 TCP 位姿，`speed`
是关节速度上限（rad/s）。前端求解 IK 后发布最新关节目标；现有 MIT
后台线程在每个通信周期生成一致的位置、速度参考，并发送给电机。
新目标覆盖旧目标，不建立运动队列；持续输入不会重置参考。

仅 MIT `servoL` 使用下列参数。posvel 的 `servoL` 直接忽略这些参数，
继续发送 IK 位置和速度上限。`servoJ`、回零、夹爪不经过参考模型。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `servoL_reference_params.frequency_hz` | 40 Hz | 二阶临界阻尼参考模型的自然频率，调高跟随更快，调低平滑更强 |
| `servoL_reference_params.max_acceleration` | 每关节 20 rad/s² | 正常跟随时的加速度上限，可按关节分别设置 |

在连接前设置参数，例如：

```python
from reBotArm_control_py.controllers import RebotArmController

controller = RebotArmController("mit")
controller.servoL_reference_params.frequency_hz = 30.0
controller.servoL_reference_params.max_acceleration[:] = 15.0
```

也可单次调用 `controller.servoL(target, speed, acceleration=15.0, frequency_hz=30.0)`。
`speed` 和 `acceleration` 支持标量或逐关节数组，单位分别为 rad/s、rad/s²。
省略新增参数时使用默认对象；显式值随本次目标保存，不修改默认对象，
下一次省略参数的调用仍使用默认值。自然频率须为有限正数，速度非负、加速度为正。
posvel 完全忽略加速度和自然频率，包括其类型及有效性。

40 Hz 是参考模型响应参数，与 YAML 的 `comm_rate` 分别设置。
默认值是未经真机调优的初值。未触发约束、低频小幅运动时，连续模型
自身的相位滞后约为 `2 / (2π × 40) ≈ 8 ms`，不代表实际 TCP 延迟。

每个关节根据位置误差 `e` 计算：

```text
ω = 2π × frequency_hz
v_des = sign(e) × min(ω|e|/2, √(A|e|), speed)
a = clip(2ω(v_des - v_ref), -A, A)
```

`√(A|e|)` 提供提前制动的速度包络。内部积分子步最多 1 ms，仅用于
数值稳定，不增加线程或总线发送次数。速度前馈直接使用该模型的
`v_ref`，不再对 IK 结果差分或做 EMA。运行中降低限速时，按加速度
上限减速，因此已有速度可能短暂高于新的上限。本轮不限制 jerk。
电机 Kp/Kd 和现有重力、摩擦、偏置力矩补偿沿用原值。

首次激活从实际关节反馈位置开始，速度为零，首周期保持。
`hold()` 立即发布保持命令并取消跟随。输入超过默认 100 ms 未更新、
IK 失败、参考越过现有关节范围，或调度间隔超过
`max(50 ms, 3 × 通信周期)` 时，按当时反馈保持一次并清除参考。
保持不随后续反馈漂移；恢复需要新目标，从当时实际位置重新开始。
紧急保持优先于正常加速度约束。增益更新不续期、不重新激活目标。
超时来自共享的 `controller.params.command_timeout`；调度保护来自同一对象的
`scheduler_gap_min` 和 `scheduler_gap_periods`，控制器、后台不再各设一份。

## 诊断录制

工作区的 `tmp/record_vr_teleop.py` 仍在顶部硬编码配置。
新录制格式为 3，保存参数、源码快照以及：

- `mit_target`：目标发布序号、时间、IK 位置、限速及本次实际使用的加速度、自然频率。
- `mit_reference`：每周期锁内消费时刻、序号、参考位置/速度/加速度、反馈、周期和保持原因。
- `group_send.*` / `motor_send.*`：实际发送参数、时间及异常。

MIT 的 PC 链路分析通过发布序号、锁内消费时间、参考周期、发送参数和
全部电机调用关联，单独计数被后续目标覆盖的输入。录制丢失或关联证据
不完整时，不把缺失记录推断为覆盖，结果标记覆盖统计不完整。
旧格式仍使用原有参数匹配。
目标到参考、参考到缓存反馈的关节误差分别统计；原始时间序列可用于
后续分析参考模型和反馈响应的滞后。
源码快照包含参数定义、两个通讯器、共享调度及轨迹参考算法。
`tools/servol_diagnosis` 已移除失效的 EMA 配置，采样最后一次完成六轴
发送调用的参考与序号，不再将新发布的 IK 目标当成实际发送参考。

PC 链路终点是发送函数返回，不代表电机收到或执行；缓存反馈也不是
电机采样时刻。离线验证不能证明实际 TCP 跟踪或全链路延迟达标。
