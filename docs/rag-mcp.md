# GPT 规划与经验检索的 MCP 接入

项目标题：RoboRAG: Interaction-Aware Retrieval-Augmented Manipulation。
这是当前实现说明；检索改善成功率、动作精度或成本仍需实验验证。

## 两类图的职责

经验 graph 记录示范的初始情境、阶段、操作与反馈，`next` 只表示历史时间顺序。
GPT 当前 plan 表达待执行意图，按当前图像绑定对象，包含每步目标、完成观察和引用。
两者可使用类似阶段结构，但不要求节点对应、顺序一致或阶段数相同。
例如白块已经拿起时，新 plan 应跳过示范中的接近和抓取；历史完成反馈不代表当前成功。

当前任务的完成条件单独来自任务说明，优先于历史经验。例如本地公开任务定义要求堆成三层并双臂归位。
因此 GPT 可以跳过某条示范中不适用的搬运步骤，但不能因为把图当作“仅供参考”而忽略当前任务的归位要求。
历史 `next` 不是强制规划依赖，当前任务的明确成功条件仍是约束。

训练示范提取出的阶段链可作为“已观察轨迹”的标注候选；不是所有场景唯一正确 plan 的 GT。
目前 graph 仍是自动标注，没有宣称人工验证过阶段边界或因果前置条件。

## 实现与调用顺序

`rag_mcp.py` 使用官方 `mcp==2.3.0`，提供 Streamable HTTP `/mcp`：

1. `observe`：当前 RGB、相机标定与本体状态；不读取物体真值。
2. `retrieve_evidence`：本地检索 1–2 条历史参考，默认 6,000 字符预算；没有在线检索模型调用。
3. `read_action_reference`：按引用展开一段真实 before/actions/after，可选历史图片与数值动作；`section=end` 可查看阶段末尾。
4. `record_plan`：记录 GPT 新生成的目标、当前对象、完成观察、来源和调整理由。
5. `execute_chunk`：默认执行 GPT 编写的 1–24 行新动作，返回实际动作回执与新画面。

叠衣额外提供 `measure_scene`：复用本地 HexaShell 基线的 RGB 衣物边界、平面与内侧抓取点测量，
不消耗控制步数。去掉上游返回的启发式 `fold_construction`；折叠目标和动作继续由 GPT 生成。
这是当前视觉工具的复用，不是 RAG 算法贡献，也不代表测量准确或抓取成功。

`read_action_reference(section=close|open)` 可按夹爪命令跨过 0.5 的时刻选窗口，
避免“抓取阶段”的起始窗口实际只有远处接近动作。未找到事件时明确返回起始窗口和 fallback；
命令事件不等于物理抓取或释放证据。叠衣提示先展开一次近期抓取参考，其余按阶段需要读取。

动作执行前必须已经检索、记录 plan。该约束只验证流程，不替 GPT 决定计划内容。
没有机械重放历史轨迹的 API；执行器只接收 GPT 为当前场景新生成的动作。
默认 `--controller delta` 与示范使用同一套 14 维 world link6 delta 表示，减少模型额外的表示转换。
每臂每行平移范数最多 0.035 m、旋转向量范数最多 0.35 rad；从实测起点开始，后续累加前一个请求目标。
`segments=[{delta: [...14 个数...], repeat: N}, ...]` 是重复动作行的无损简写，总共仍最多 24 行；
完整验证后才执行，日志保留编码和展开动作。它不生成目标、不插值、不改变控制频率。
另可用 `targets=[{steps:N,left:{tcp_xyz:[x,y,z],quaternion_wxyz:[w,x,y,z],gripper:0},...}]`：
GPT 指定当前目标和时长，确定性转换器将指尖目标换成 link6，并做线性平移/世界旋转插值。
总共仍最多 24 步，沿用同一个增量执行器和误差终止条件。旋转期间是 link6 插值，不保证指尖走直线。
这个输入省掉 GPT 在 Python 中反复换算坐标；它是控制接口变化，不应归因成 RAG 算法收益。
实际平移跟踪误差超过 4 cm 时停止剩余行，下一段从新观测重新规划。
`return_home=true, actions=[]` 在撤离物体后将双臂回到本回合起始关节姿态。

保留 `--controller tcp` 作为较早的实验选项：每段 1–8 个绝对指尖 TCP 路点，局部 IK 插值，
失败、裁剪或明显未到位时停止剩余路点，支持 `home=true` 路点。TCP 与 link6 引用不能直接混用。
目前成功的回合采用 delta 接口；TCP 版本的两次完整堆叠回合均失败，不能把 TCP 版本宣称为已验证有效。
夹爪命令不用于确认抓取，任务完成由 RoboDojo 原生评价器判断。

## 上下文与成本

整个机器人回合复用一个原生 Codex 会话，稳定工具定义、初始经验和 plan 保留在历史中，追加动作与反馈。
TCP 接口默认只发头部图像，delta 接口默认头部和右腕；可按抓取侧选择相机。
静态头部标定不反复发，移动腕相机标定须重新获取。
历史动作数组、图片按需展开。`execute_chunk` 已附新图，不必再调用一次 observe。

这是提高缓存复用和减少工具往返的实现方式，是否省钱以实测为准：
记录 input、cached input、output、可得的 cache-write、动作段数、实际动作行/路点数、原生控制步数和墙钟时间。
原生控制步指 RoboDojo 动作派发计数，不是 PhysX 更细的积分步数。
Codex 的 completed turn 汇总可能包含多个模型请求，不能将其当作一次模型调用。
当前登录方式未提供逐回合账单，美元费用保持未知；缓存命中率也不能代替总成本。

## 运行

在已配置 `local.runtime.json`、RoboDojo 资产、图库及 HexaShell 依赖的环境中：

```bash
<policy_python> -m pip install --target .runtime/mcp -r experiments/rag-mcp-requirements.txt
<policy_python> scripts/run_rag_mcp.py --output runs/my-rag-mcp-stack --layout 0 --controller delta --effort medium
<policy_python> scripts/run_rag_mcp.py --task fold_clothes --variant standard --layout 0 --effort xhigh --output runs/my-rag-mcp-fold
```

MCP 依赖只对 MCP 进程设置 PYTHONPATH，不修改仿真器的依赖。
模型只看到当前 RGB、本体状态与取出的经验；源码、评分器和物体真值位于模型可读目录之外。
`--no-rag` 提供相同机器人接口和规划流程但返回空经验，可用于同场景消融。

### 维护与统计

命令行脚本只负责包入口，`rag_runner.py` 管理进程生命周期；`rag_tasks.py` 选择任务及场景变体，
任务提示与控制协议放在 `prompts/`。新增任务不再改写启动器中的堆叠提示或硬编码场景名。
`episode_files.py` 统一原子 JSON 写入与中断 JSONL 读取；`rag_accounting.py` 负责回合汇总。
实际控制步数优先使用原生终态和动作日志计数，不能只累加收到的 MCP 回执。
中断回合只知道已落盘部分时标为 `lower_bound`，没有计数证据时为 `unknown`；不会把缺失当零。
正常零动作退出仍保留原生评分，工作流字段单独标为未完成。

叠衣先导使用同一个 delta 控制器及三条 TRAIN 示范；标准/杂物场景共享任务经验，
布局与原生成功条件保持各自任务定义。叠衣测试和 HexaShell 对照见后续独立实验记录，
接入支持不代表已经取得成功率提升。

每次运行写入独立目录：

- `run.json`：场景、模型、配置和关键源文件/索引哈希。
- `mcp-calls.jsonl`：查询、实际经验结果、GPT plan 和执行回执。
- `gpt-plan.json`：GPT 为本回合生成的最新计划。
- `observations/`、`actions.jsonl`、`commands.jsonl`：新图像、本体状态、物理控制与动作段证据。
- `agent-events.jsonl`、`agent-exit.json`：原生 Codex 事件与实际报告的 token 使用量。
- `result.json`：只有完成原生评价后才写入结果；异常写入 `exit.json`。

图库前端继续负责浏览经验图与检索证据；在线回合 plan 不会覆盖或冒充图库中的历史 graph。

## 设计依据

[MCP 官方 Python SDK](https://py.sdk.modelcontextprotocol.io/) 用于协议与工具 schema；
[Codex MCP 配置](https://learn.chatgpt.com/docs/extend/mcp) 用于连接本地 HTTP 服务。
保留稳定前缀并追加交互记录的缓存设计参考 [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)。
实际使用量以 CLI 日志为准，不假设 CLI 暴露全部 API 缓存设置。
示范动作格式沿用项目固定的 RoboICL 转换，运动执行复用现有 HexaShell/原生 RoboDojo 适配器。
当前堆叠任务的完成条件来自 `vendor/RoboICL/configs/tasks/stack_blocks.json` 中保存的公开任务说明，
并与原生任务实现 `task/RoboDojo/tasks/stack_blocks.py` 核对。

## 2026-10-24 开发实测

`rag-mcp-stack-l0-v5-delta` 使用 GPT-6 Astra / medium，完成标准 layout 0：
原生 `score=100`、`success=true`，179 个原生控制步。18 次 `execute_chunk` 中，
17 次执行新生成的 152 行增量动作，最后 1 次归位派发 27 个控制步。
归位过程中原生评价器已判成功并终止，不能把终止时的姿态说成每个关节精确回到零位。

一次初始检索、三次动作参考展开、一次 GPT plan 记录；没有执行中反复检索。
GPT 生成 4 步新 plan，选择右臂完成搬运并保留木纹块为底座，明确跳过参考中的底座搬运和左臂操作。
调用日志验证 `retrieve_evidence → record_plan → execute_chunk`。

本回合报告 input 929,882 tokens，其中 cached input 856,064（约 92.1%），
未缓存 input 73,818，output 3,506（reasoning 358 为其中的细分），报告的 cache-write 为 0。
模型进程墙钟 250.9 秒，含仿真启动与清理的总墙钟 289.5 秒。没有逐回合美元账单。
这证明了缓存确实复用及多步动作按段派发；尚不能证明总体成本最优或相对无 RAG 更省钱。

所有尝试保存在 `results/rag-mcp-stack-pilot-20261024.json`：v1 收到终止信号，
v2 因回执副本目录未创建而中断；两者终端 token 使用量缺失，不计为零。
v3 完整但仅 15 分，v4 完整但 0 分，均 `success=false`；v5 为成功回合。
这些是同一开发场景上的迭代调试，期间改了任务提示、控制表示和推理档位，
没有无 RAG 的匹配对照，不能用于宣称检索因果收益或留出集成功率。
