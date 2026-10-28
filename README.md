# RoboRAG: Interaction-Aware Retrieval-Augmented Manipulation

当前分支 `refactor/v3-clean` **直接从原始 fold v3 提交 `1e9ffa0` 开始重构**。模型观察现场，检索历史示范，记录自己的计划，再生成短动作段并查看反馈。本轮拆分原版 MCP 文件、补充来源记录和批次 seed 透传，并增加独立的只读交互监控。

**首轮重构 `b2d5686` 的完整 fold_clothes 测试为 35/50，未达到预先登记的历史 v3 37/50 验收线。** 原版在当前环境的 50 例诊断对照也已结束：33 成功、12 失败、5 例因容量或连接问题未评分，两轮结果单独保留。当前分支随后修复了已查明的基础接口问题，尚未进行新一轮原生效果测试，不能继承旧成绩。详见[验证结果](docs/v3-clean-validation-20261027.md)。

## 当前执行方式

原版已经通过 `record_plan` 记录 `current_scene`、`steps` 和 `adaptation`。模型依据当前观察决定下一步。当前 fold 默认使用 manipulation SOP 示范目录，按明确引用展开经验片段；SOP 保留 v3 实际使用的历史内容，不要求当前计划按示范节点执行。沿用原版控制器，没有加入 graph plan、阶段准入或阶段驱动执行。

独立 SOP 包只含 3 份示范概览、16 段已返回的操作文字、19 个实际展开的窗口和 37 张历史头部图片，约 3.56 MB。完整轨迹与原始动作数组不进入运行时包。缺失片段明确报错；不会回退读取旧完整图库。目录、来源和验证见 [SOP 包说明](docs/rag-mcp.md#独立-manipulation-sop-包)。

检索参数继续接受“双臂”等等价写法。2026-10-28 在用户要求下，执行调用重新对齐原始 v3：每次 1–24 行，目标编译保持模型指定步数，超速或超长请求在执行前拒绝，不自动延长或拆包续跑。恢复原动作 schema、说明、反馈及归位路径。此前无上限与自动延长版本的结果单独保留；本次效果另行测试。SOP 包和 `record_plan` 不变。

独立监控读取已有计划、动作意图与反馈，显示当前交互。它不调用模型、不向模型提供上下文、不控制机器人。模型写下的“抓住”“完成”只是声明，任务成败以原生评价为准。接口与验证边界见[v3 重构说明](docs/v3-clean-refactor.md)。

## 运行 fold_clothes

在本分支的工作目录中，使用 `local.runtime.json` 指定的策略 Python 环境。原生运行需要现有 RoboDojo/Isaac Sim、资产、HexaShell、MCP 依赖及已登录的策略 CLI；配置入口见[MCP 运行说明](docs/rag-mcp.md#运行)和[依赖配置](docs/runtime.md#local-setup)。这里的 `python` 指策略解释器。

当前工作区已生成 `artifacts/fold-v3-sop`。在保留原始 v3 日志和 TRAIN 归档的机器上，可离线重建到一个不存在的输出目录：

```bash
PYTHONPATH=src:.runtime/mcp python scripts/build_v3_sop_package.py
PYTHONPATH=src:.runtime/mcp python scripts/verify_v3_sop_package.py
```

fold 默认读取独立 SOP 包；包不存在时明确提示构建，不自动回退旧图库。显式 `--bank` 仍可用于其他任务或登记过的旧库对照。输出目录必须是新的；以下命令会启动仿真并调用模型：

```bash
python scripts/run_rag_mcp.py --task fold_clothes \
  --seed 1 --variant standard --layout 0 --controller delta \
  --model gpt-6-astra --effort xhigh --max-wall-seconds 1800 \
  --output runs/fold-single
```

在另一终端查看运行中的交互，或去掉 `--follow` 读取已有轨迹快照：

```bash
PYTHONPATH=src python -m roborag.interaction_monitor runs/fold-single --follow
```

关闭监控不影响策略。每例保留 `run.json`、`gpt-plan.json`、`mcp-calls.jsonl`、`commands.jsonl`、图像和模型用量。取得原生评分后写入 `result.json`；异常退出保留 `exit.json`，缺失评分不会当作零分或成功。

## 代码与验证入口

- `src/roborag/rag_runner.py`：启动模型、MCP 和仿真进程，保存回合结果及来源。
- `src/roborag/mcp/`：从原版 `rag_mcp.py` 移出的会话、观察、检索、计划与动作工具；旧导入入口保留。
- `src/roborag/sop_store.py`：独立示范目录、轻量文字检索与按引用读取经验片段；不读取完整轨迹。
- `src/roborag/interaction_monitor.py`：独立读取交互日志。
- `prompts/`：沿用原版任务提示，动作协议说明同步已批准的接口变化。
- `experiments/fold_clothes/`：预先登记的条件与固定场景。
- `tests/fixtures/v3-contract.json`：移动代码前捕获的原版契约，不能从新实现重生成来通过测试。

已有本机 MCP 依赖安装在 `.runtime/mcp` 时，可运行离线检查：

```bash
PYTHONPATH=src:.runtime/mcp python -m unittest discover -s tests -v
```

这些检查以原版契约为基准，显式列出基础接口修复的允许差异，并核对未改动的计划、观察、参考展开、seed 透传和监控行为，不调用模型或仿真。原生测试、参考文本和图像回放的证据统一见[验证结果](docs/v3-clean-validation-20261027.md)。

## 研究背景与历史实现

项目复用 HexaShell 的几何与视觉测量，以及 RoboICL/RoboDojo 的仿真和原生评价接口。检索对成功率或成本的独立贡献尚未通过消融验证；当前结果也不代表真机验证。

[研究动机](docs/motivation.md)、[早期双库设计](docs/architecture.md)、[经验图库](docs/graph-bank.md)和[历史试验](docs/results.md)保留此前研究内容。早期双库原型及其结果与当前 v3 MCP 路径分别记录。大型轨迹、图像、图库和本机配置不纳入 Git。
