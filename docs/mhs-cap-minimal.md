# RoboRAG MHS + CaP v0.1

这是 RoboRAG 自己的具身 MHS。MCP 负责传输，项目定义机器人操作的语义。
参考 [MCP 工具规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
和 [官方 Python SDK v1](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x)。
SDK 上界固定 `<2`，保持仓库 Python 3.10+ 支持；当前验证环境为 Python 3.11、MCP 1.30.0。
不声明兼容 Anthropic 同名 MHS，也不等待它开放。

## 闭环

```text
任务 + 最新观测 + 当前回合程序/回执
             ↓
       一个模型生成代码
             ↓
       CaP 有界程序运行器
             ↓
       RoboRAG MHS Python facade
             ↓  MCP stdio
       MHS server → 预编写本体 skill → 设备
             ↑                  ↓
             └── 回执、错误、新观测 ──┘
```

模型按语义需要生成一段程序；本地执行条件和循环，skill 内的运动步不调用模型。
失败中止当前程序的剩余部分，已执行动作保留。下一轮获得完整执行反馈和新观测，
决定如何修正。没有额外监督模型，没有检索依赖。

## 项目 MHS 的最小契约

每个服务进程拥有一个设备会话。第一版只允许一个控制客户端；调用串行且阻塞。
所有位移使用设备声明的参考系和米，不能隐式做坐标变换。

- `mhs_describe` / `robot.describe()`：协议版本 `roborag-mhs/0.1`、设备标识、
  支持的机械臂、坐标系、单位、技能描述及实际限制；MCP 同时提供每个工具的输入 schema。
  同一描述可通过资源 `mhs://device/capabilities` 读取。
- `mhs_observe` / `robot.observe()`：`session_id`、`state_seq`、来源、坐标系、
  本体状态和可用图像。`state_seq` 统计成功返回的底层执行步，不是相机帧号或墙钟时间。
- `mhs_move` / `robot.move(arm, position)`：绝对 xyz 目标，保持起始观察中的末端姿态和夹爪目标。
- `mhs_gripper` / `robot.gripper(arm, opening)`：显式夹爪命令，0 关闭、1 打开。
  指令值与测量值区分；关闭夹爪不能自动解释成抓住物体。
- `mhs_stop` / `robot.stop()`：锁存本会话停止，后续运动被拒绝；重新运行需要新会话。
  该同步接口在当前阻塞调用结束后生效，不是硬件急停。

成功的运动回执包含 `session_id / skill / status / start_seq / end_seq / executed_steps`。
`status=completed` 表示技能按其控制器规则返回；`task_success=null`，不替代任务评测。
设备异常通过 MCP `isError` 返回，CaP 保存错误并终止程序尾部。
RoboDojo 中途失败的错误含已执行步数；通信失败则视为执行状态未知，关闭会话、不自动重发。
本版没有断线恢复、跨会话幂等或跨进程恢复保证。

## 运行

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[cap]'
# 确定性假模型 + 合成设备；真实 MCP stdio 连接
.venv/bin/python -m roborag.cap.demo --output runs/mhs-normal
# 第一次移动拒绝，模型替身收到反馈后重新生成程序
.venv/bin/python -m roborag.cap.demo --output runs/mhs-repair --reject-first-move
# 独立启动具身工具服务，供支持 MCP stdio 的客户端连接
.venv/bin/python -m roborag.mhs.server
```

服务 stdout 专用于 MCP，不能往其中打印调试文本。输出目录必须是新目录。
`events.jsonl` 保存能力、输入观察、决策、逐次 skill 返回和异常；
`program-NNN.py` 保存实际程序；`summary.json` 区分模型结束、轮数耗尽和基础设施失败。
合成验收标为 `synthetic_check`，永远不写成原生机器人成功率。

已有可用的登录态 CLI 时，可运行真实模型入口（需要模型推理额度）：

```bash
COMPARE_CODEX_BIN=/absolute/path/to/codex \
  .venv/bin/python -m roborag.cap.demo --brain codex --model gpt-6-astra \
  --effort xhigh --max-rounds 8 --output runs/mhs-astra
```

模型只收到提供的上下文和图像，返回 `kind/code/note`，不直接使用 shell、文件或额外工具。
它生成的程序由 CaP 运行器解释。模型调用沿用本仓库的账户 CLI transport，
不等同官方论文 API 条件；此次验证没有运行付费模型实验。

替换设备服务：提供受信任的 JSON 配置，字段为 `command`、`args`、可选 `env`，
通过 `--server-config /absolute/path/server.json --brain codex --task '...'` 使用。
客户端会执行该配置指定的本地命令并检查 MHS 工具及版本；不要使用模型输出作为启动命令。

## 程序能力与本体技能

```python
obs = robot.observe()
current = obs["arms"]["left"]["position"]
for z in [current[2] + 0.02, current[2] + 0.04]:
    robot.move("left", [current[0], current[1], z])
robot.gripper("left", 0)
```

支持局部变量、JSON 字面量、索引、数值运算、比较、if/else、有限 for、break/continue。
每段最多 64 次设备调用、5000 次解释器操作，range 最多 100 次。
完整语法先检查再执行，不支持 import、任意 Python 对象访问、函数定义、while 或第三方包。
这是用于第一版的 Python 子集，不是完整 Python 沙箱。

`roborag.mhs.robodojo.RoboDojoDevice(env)` 包装已经初始化的单环境：
平移由本地线性插值实现，保留另一只臂、末端姿态和夹爪指令。
读取原有 EE 状态并检查逐步位置残差；超过距离/步数预算时拒绝，不截断计划。
受控末端为 `link6`，不是指尖。它依赖现有 `env.take_action` 的 IK 与底层执行，
本接口不新增碰撞检测，也不改变原生评分。

初始化/销毁仿真与 GPU 生命周期仍由调用方负责；驱动须在仿真所属线程调用。
MCP 默认入口只启动合成设备，尚未提供 RoboDojo 服务启动器。
可通过 `encode_images(raw_observation)` 回调提供 RGB 图像，使用
`{"type":"input_image","image_url":"data:image/jpeg;base64,..."}` 格式；
未配置回调时 `images=[]`，不隐式提供物体坐标、接触状态或评分。

## 验证边界与后续

已验证真实 MCP 子进程中的发现、连续技能、条件/循环、拒绝反馈、修正、会话隔离与停止；
RoboDojo driver 使用可控环境替身验证步长、部分执行与跟踪中断。
原生仿真、真实大模型判断和真实抓取尚待单独验证，不将合成夹持逻辑作为运动技能成果。

2026-10-23 检查记录：新增 `test_mhs_cap.py` 的 18 项测试全部通过。
已有完整测试集在新增最后两项集成用例之前运行 39 项，32 项通过、7 项旧视觉检索用例
因隔离工作区未安装完整 RoboICL 运行时而跳过；随后新增集成用例及全部 MHS/CaP 用例通过。
CLI 模型传输测试使用本地可控替身，不能当成 Astra/Sol 能力验证。

当前工作记忆保留当前回合全部程序和回执，受回合轮数限制；没有长期经验写入或 RAG。
后续先完成一个原生短任务，再讨论任务结束时写什么经验、规划前/失败后/主动请求何时检索。
