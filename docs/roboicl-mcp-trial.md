# Astra xhigh：RoboICL 经标准 MCP 的叠衣服试验

2026-10-23。测试 `fold_clothes / standard / seed 0 / layout 0`。
保留历史 RoboICL dialogue policy、一次 TRAIN 示范和上下文组织，只更换机器人 I/O 通道。
本轮不让模型生成 CaP 程序，不开启 RAG。
正式试验正在运行，最终原生评分和消耗将在完成后记录。

## 冻结条件

- 模型 `gpt-6-astra`，推理强度 `xhigh`，沿用已登录 Codex CLI；记录每次真实调用与 usage。
- 相同的 12 个 TRAIN 块、12 个 LIVE 块、50 张图像上限、10 步预测/执行 horizon。
- 同一份公开训练 episode 0 的冻结示范、原生布局和已有任务说明；原生控制预算 500 步。
- 上游 RoboICL 为历史 `c0e58e456ce4b2ea1c401d1b261d6e0327c6e04c` 及当时同一补丁。
  启动前逐文件检查历史哈希，再复制策略与配置到 run 目录供实际进程加载。
- 本体控制沿用原生 IK 连续性检查；RoboICL 的动作解析、跟踪中断、纠错与历史压缩逻辑保持原样。
- 依赖固定在 [requirements-roboicl-mcp.txt](../experiments/requirements-roboicl-mcp.txt)。

历史 Astra 对照为 100 分、完整成功、247 步、25 次模型调用；
输入 2,140,492 tokens，输出 58,291 tokens，总计 2,198,783 tokens，约 53.17 分钟。
这是事先选出的成功布局，不代表随机任务上的成功率。

## MCP 执行路径

RoboICL policy 继续生成原来的 `act` 动作张量。主机按原 rollout 顺序通过 MCP 调用：

1. `read_observation`：返回白名单本体状态、任务指令、相机标定和标准 MCP PNG 图像。
   PNG 保持 RGB 像素，策略收到后继续执行原来的 JPEG / 三相机拼接处理。
2. `control_step`：将模型已经校验、转换的双臂绝对目标执行一个原生控制步。
   不增加 CaP 的轨迹插值，不重新定义动作尺度。IK 拒绝事件通过标准 `isError` 和结构化结果返回。
3. `episode_status`：只返回控制步数、步数上限和终止标志。
4. `finish_evaluation`：仅当原生环境已终止时完成原生评分和视频写入；不会把成绩反馈给策略。

所有机器人观测与动作均经过标准 MCP stdio；MCP 服务在独立 Python 中运行，仿真留在自己的主线程。
评分使用原生 `env.run_eval()`，模型不接触运行中的奖励、物体真值或任务实现。
这次测量的是 MCP 接入后的 RoboICL 表现和通信成本；模型 token 差异也受随机生成及执行轨迹影响。

## 预检与证据

`runs/roboicl-mcp-fold-preflight-v2-20261023` 在零模型调用下验证了一个原生控制步，
首轮指令、工具 schema、序列化文本以及 24 张 TRAIN 图像全部与历史 Astra 一致。
首个预检因隔离环境缺少示范解码依赖退出，原始记录保留，没有计作模型失败或任务零分。
预检有意在一步后结束，不产生可比较的任务成绩。

正式运行目录：`runs/astra-roboicl-mcp-fold-clothes-l0-20261023`。
`frozen-plan.json` 保存模型、布局、示范、配置与源码哈希；`initial-context-comparison.json`
验证首次模型输入；`mcp-calls.jsonl` 保存工具调用、控制参数、耗时和返回字节数；
`policy/` 保存原 RoboICL 轨迹与反馈；`decisions/` 保存逐调用输入/输出、usage 和命令行配置。
`native/evaluation.json` 指向原生评分文件；三路视频与执行步数在最终汇总时交叉核对。

```bash
.venv/bin/python -m pip install -r experiments/requirements-roboicl-mcp.txt
.venv/bin/python scripts/run_roboicl_mcp.py \
  --historical-plan /absolute/path/to/sol-roboicl-fold-clothes-l0-20261023.json \
  --runtime /absolute/path/to/local.runtime.json \
  --model gpt-6-astra --effort xhigh \
  --output runs/astra-roboicl-mcp-fold-clothes-l0-20261023
```

本机模型 CLI 使用 `--proxy http://127.0.0.1:7897`；其他机器按自己的网络设置选择。
历史计划及资产是已有实验的本机输入，不随代码仓库发布。`--preflight-only` 不调用模型。
汇总运行中的记录时，未完成调用的 usage 保留未知；失败/中断回合不自动记零分。
输入 tokens 已含缓存输入，不重复相加；reasoning tokens 单列观察，不再加到总输出。
账户 CLI 的 token 报告不等于实际美元账单。
MCP 调用耗时包含本体执行、渲染、图像编码和通信，不能直接解释为协议本身的额外开销。

完成后使用以下命令交叉核对原生评分、控制步数、三路视频帧数、视频解码、源码快照和逐调用 usage：

```bash
.venv/bin/python scripts/summarize_roboicl_mcp.py \
  runs/astra-roboicl-mcp-fold-clothes-l0-20261023 \
  --baseline-report /absolute/path/to/results/sol-roboicl-fold-clothes-l0-20261023.json \
  --output results/astra-roboicl-mcp-fold-clothes-l0-20261023.json --audit-videos
```

GPU 同时有原有 ComfyUI 和其他线程的衣物技能训练，墙钟时间不作严格受控的速度结论。
历史模型服务日期不同，不能把单回合差异完全归因于 MCP。
