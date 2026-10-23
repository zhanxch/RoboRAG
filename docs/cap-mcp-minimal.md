# 标准 MCP + CaP + RoboDojo 最小闭环

分支：`feat/mhs-cap-minimal`。本版使用标准 MCP，不定义额外 MHS 协议。
参考 [MCP Tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)、
[Resources](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) 与
[官方 Python SDK](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x)。
SDK 依赖固定为 `>=1.28,<2`，当前验证环境为 Python 3.11 / MCP 1.30.0。

## 目标与边界

统一的是 RoboDojo 面向模型和程序的观测、动作和反馈入口。已有相机、状态读取、
`env.take_action`、IK 和原生终止条件由适配器接入 MCP；不改写仿真器或任务定义。
CaP 既能调用这个服务，也能连接使用不同工具名、参数与机器人形态的标准 MCP 服务。

```text
任务 + MCP 能力目录 + 当前观测 + 程序执行记录
                        ↓
                   一个模型生成代码
                        ↓
               CaP 有界 Python 解释器
                        ↓
                标准 MCP client / stdio
                        ↓
                RoboDojo MCP server
                        ↓
                本地仿真 worker / driver
                        ↓
                原生控制器、传感器、环境
                        ↑
                  标准 MCP 结果 / 图像 / 错误
```

MCP 服务和 Isaac Sim 使用各自的 Python 进程，避免 Isaac 自带异步库影响 MCP SDK。
worker 只处理服务进程发来的本地串行命令，所有仿真调用保持在主线程。
这条私有进程管道是实现细节，对外入口只有标准 MCP。

## RoboDojo 的 MCP 能力

工具通过标准 `tools/list` 提供 JSON Schema，通过 `tools/call` 执行：

- `robot_state`：任务指令、两臂 link6 位姿、关节位置、夹爪指令值与会话序号。
- `camera_observe`：原生相机 RGB，以标准 MCP `ImageContent` 返回，附相机标签；可选相机子集。
- `move_ee`：单臂绝对 xyz 与可选 wxyz 姿态；在本体侧插值并调用原生 IK。
- `move_joints`：单臂关节目标，依据原生关节限位检查并插值；没有限位时不发布此工具。
- `set_gripper`：显式开合指令，0 关闭、1 打开；保持两臂位姿。
- `execution_status`：最近一次执行回执、已知完成的步数、停止状态。
- `stop_motion`：锁存后续动作禁用；当前阻塞调用结束后生效。

资源通过标准 `resources/list` / `resources/read` 读取：
`robodojo://robot` 为设备参考、单位、坐标系和限制；
`robodojo://state` 为当前本体状态；`robodojo://execution` 为当前执行回执。
这些名称及数据字段属于此适配器，其他设备可以发布自己的能力。
没有把 `move_ee`、左右臂、夹爪或 `state_seq` 强制规定为 MCP 协议字段。

当前双 ARX-X5 的位移单位为米，关节单位为弧度，坐标为环境原点/世界坐标轴，
受控 link6 并非指尖/TCP。移动保持另一臂和夹爪目标，未指定姿态时保持姿态。
本地检查运动距离/步数、关节限位、IK 连续性与执行跟踪残差。
未新增碰撞规划、物体检测、抓取识别或硬件急停。
夹爪测量值没有可靠来源，返回 `null`，不把指令值冒充测量值。

MCP 承载这些能力及其物理语义，但并不提供机器人控制器或实时伺服保证。
当前实际支持单个 RoboDojo 环境、一个服务会话和一次 episode；
没有宣称已适配灵巧手、人形或轮式机器人。

## CaP 如何调用

```python
state = mcp.call_tool("robot_state", {})["structuredContent"]
p = state["arms"]["left"]["position"]
for z in [p[2] + 0.01, p[2]]:
    mcp.call_tool("move_ee", {"arm": "left", "position": [p[0], p[1], z]})
receipt = mcp.read_resource("robodojo://execution")
```

通用客户端只提供 `mcp.call_tool(name, arguments)` 和 `mcp.read_resource(uri)`。
连接时协商协议与能力，完整读取分页目录，按工具发布的 schema 校验参数。
原样保留 `content` 和可选 `structuredContent`；文本工具不必返回 JSON。
MCP 图像传入模型时转换为真实图像附件，程序内部仍接收标准 MCP 数据。

harness 不猜测哪些工具是观测或停止。主机通过 `before_round` 和 `on_exit` 配置调用，
RoboDojo 启动器配置每轮状态/RGB和退出停止；连接其他设备时可使用其自己的工具。
本版使用 Tools 与 Resources；Prompts、订阅、能力目录热更新和实验性 Tasks 尚未使用。
MCP 不要求每个服务或客户端实现全部可选能力。

支持变量、JSON 字面量、索引、数值运算、比较、if/else、有限 for、break/continue。
每段最多 64 次 MCP 调用、5000 次解释器操作、range 最多 100 次；语法先检查再执行。
不支持 import、任意 Python 属性、函数定义、while 或第三方包。这是 Python 子集解释器。

动作错误以标准 `isError` 返回；程序尾部停止，已执行动作不回滚。
下一轮同一模型收到完整调用反馈和新观测，再生成程序。
`executed_steps` 只统计已成功返回的底层调用；底层调用中途抛错时还标记
`in_flight_step_unknown`，不能假定失败动作未运动。通信失败关闭会话，不自动重发。
`stop_motion` 不是异步急停；MCP 取消请求本身也不等于物理停止。

## 启动与记录

先按 [runtime](runtime.md) 配好现有 RoboDojo 资产与运行时。
本项目 MCP 环境与已有仿真环境分开，不需要向 Isaac 环境安装 MCP。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[cap]'
# 脚本驱动原生仿真，验证状态/RGB、1 cm 末端往返、关节/夹爪保持及停止
.venv/bin/python scripts/run_cap_robodojo.py --runtime /absolute/path/to/local.runtime.json \
  --task stack_blocks --layout 0 --smoke --output runs/native-smoke
# 使用已配置的账户 CLI 调用模型；会消耗推理额度
.venv/bin/python scripts/run_cap_robodojo.py --runtime /absolute/path/to/local.runtime.json \
  --task stack_blocks --model gpt-6-astra --effort xhigh --output runs/native-cap
# 仅生成配置，供其他标准 MCP 客户端启动 RoboDojo，无需 CaP
.venv/bin/python scripts/run_cap_robodojo.py --runtime /absolute/path/to/local.runtime.json \
  --task stack_blocks --prepare-only --output runs/native-service
```

最后一条产生 `mcp-server.json`，其中的 `command / args / env` 可放入客户端的 stdio 服务配置，
客户端初始化超时应设为 300 秒以容纳仿真启动；服务在连接时初始化场景。
`worker.json` 保存本机仿真启动参数。每次新会话使用新输出目录，避免覆盖证据。
服务 stdout 仅用于 MCP，仿真日志使用 stderr。

`cap/events.jsonl` 记录能力、每轮输入、程序及逐次调用结果；`program-NNN.py` 保存代码；
`summary.json` 区分模型结束、轮数耗尽与基础设施错误，任务成绩默认 `null`。
`run.json` 保存依赖版本、布局和源文件摘要；`native/device-final.json` 保存最终设备状态。
模型主动 finish 是模型的判断，不是原生评测成绩。该入口没有运行原生评分汇总器。

无需仿真时可运行 `python -m roborag.cap.demo --output runs/fixture --reject-first-move`。
默认使用合成设备/脚本模型；`--server-config file.json --brain codex --task '...'`
可连接其他受信任 MCP stdio 服务。配置是主机提供的本地命令，不来自模型生成代码。

## 验证与后续

原生接口烟测使用 `stack_blocks` 布局 0，验证标准 MCP 传递真实 RGB、本体状态、
设备参考、末端往返、关节/夹爪调用和停止。脚本不做抓取，也不作为模型效果或任务成功率。
失败尝试与后续修复记录保留在各自 runs 目录，没有覆盖失败结果。
单元/跨进程测试覆盖不同工具名、文本/结构化/图像结果、资源、分页、schema、
部分执行、错误反馈、修正、进程退出与会话隔离。

2026-10-23 验证：`unittest discover -s tests -v` 共 48 项，41 项通过，
7 项旧视觉检索测试因此隔离环境未安装可选机器人检索依赖跳过。
其中本次 MCP/CaP 测试 25 项全部通过；CLI 模型测试使用替身。
原生记录为 `runs/robodojo-mcp-smoke-v4`：3 路 RGB、15 次底层动作调用成功返回，
1 cm 抬升指令对应实测 8.95 mm，返回起点误差 0.87 mm，最终 `stopped=true`。
`task_success=null`；没有运行真实模型的抓取或原生任务评分。
合成错误恢复记录为 `runs/standard-mcp-final`：第 1 次移动被拒绝后，第 3 轮结束并通过合成检查。

当前工作记忆保留程序、反馈和观测；不建立长期经验库，不默认每步检索。
下一步先用模型完成可复现的原生短任务，再研究何时 RAG，以及经验如何写入、验证和维护。
