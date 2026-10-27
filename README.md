# RoboRAG: Interaction-Aware Retrieval-Augmented Manipulation

当前分支 `refactor/v3-clean` **直接从原始 fold v3 提交 `1e9ffa0` 开始重构**。本轮仅拆分原版 MCP 文件并增加独立日志监控，沿用原版 `record_plan` 与动作执行，不引入阶段式执行架构。对齐方法、监控入口和测试边界见 [v3 重构说明](docs/v3-clean-refactor.md)。以下保留原始版本的研究背景和历史实现说明，不表示本轮采用双库或新增阶段策略。

**面向交互阶段与经验的检索增强机器人操作**

RoboRAG 研究如何根据当前交互阶段，存储、检索并使用历史交互经验，以改善 Agent 的后续决策。
经验表示与更新、阶段感知的检索与使用、跨后端验证是拟贡献；MCP 是支撑接口，不单独宣称算法创新。
当前实现与实验结果分别记录，研究目标不代表已经证明的收益。

## 经验图库与 GPT 规划

[交互经验图库](docs/graph-bank.md) 提供真实示范 graph、阶段、动作前后证据和浏览前端。
graph 记录历史，`next` 是示范顺序；GPT 根据当前初始图像、任务要求和少量检索参考生成新 plan，
再生成当前场景的新 action chunk，二者不需要与历史节点一一对应。

[MCP 接入](docs/rag-mcp.md) 已跑通“检索 → GPT 新 plan → 分段执行”。标准积木 layout 0 开发回合
取得原生 **100 分、success=true**，179 个原生控制步、18 次动作段调用，缓存输入占比约 92.1%。
该回合使用与示范一致的 14 维 link6 增量动作；此前失败尝试也保留在报告中。
这验证了流程可运行，不证明 RAG 相对无 RAG 的成功率或成本优势。

下述双库方法与配对结果是此前保留的实现和实验，与此次 MCP 开发回合分开记录。

## Motivation

ICL 使模型能够通过输入中的示例适应任务，推理时无需更新参数。随着可用示例增加，如何选择上下文成为重要问题：模型每次能读到的经验有限，而经验与当前问题的相关性会影响其价值。检索为从更大的经验集合中选择上下文提供了一条路径。

机器人将这一问题带入连续决策。历史示范提供可迁移的操作经验，当前执行产生的观察、动作和反馈则揭示现场状态及刚刚发生的错误。RoboRAG 因此分别管理两种经验：

- **长期 RAG 库**：执行前从训练集示范及经过筛选的历史轨迹构建，跨 episode 复用；一次评测中固定版本、只读。
- **短期 RAG 库**：每次 episode 从空库开始，随执行写入已观察到的交互；检索本次任务中的相关经历，episode 结束时关闭。

研究问题是：**在固定上下文预算下，分别检索长期经验和短期交互，能否改善机器人对新场景的适应、长程任务执行和失败恢复？** 这是待验证的假设。

研究背景及来源见 [motivation](docs/motivation.md)，工程设计见 [architecture](docs/architecture.md)，验证路径见 [experiments](docs/experiments.md)。

## Pilot status

真实 RoboDojo `stack_blocks` 已在开发布局 0 和冻结后验证布局 1 均取得 **100/100、完整成功**，分别用 195/302 步、20/31 次模型调用。相同配置的固定上下文对照也都成功，分别用 240/237 步、24/24 次调用。本轮打通了流程，未显示稳定的检索增益；这是两个布局的小规模试点。完整结果和投币试验中的负面结果均保留在 [results](docs/results.md)。

## Current scope

当前版本包含双库核心与真实 RoboDojo 执行接入：

- 离线 JSON 快照构建与加载；长期库只接收训练集中的示范或整理后的经验。
- episode 独立的短期内存库、只检索截至当前时刻可见的历史、关闭后清空。
- 机器人及动作接口匹配、两路独立检索配额、最新交互保留。
- 公开 HDF5 训练轨迹下载、固定版本及 SHA256 校验、视觉与动作片段入库。
- 冻结 ResNet-18 多视角特征、本体状态检索；多示范库先按初始画面空间特征选轨迹，再检索片段。
- RoboDojo 相机、动作控制、模型调用、原生评分和可审计检索日志。
- `zero`、`fixed`、`long_only`、`dual` 四种比较模式；固定选段和双库检索使用相同模型与片段配额。

纯 Python 核心保留词项检索，用于验证隔离和时序约束；真实机器人接入使用视觉检索。上下文配额以片段数计，并受上游图像上限约束；这不等于严格相同的 token 消耗。当前完成了小规模试点，**不将模型的“已抓住”等意图文字当作成功，不将相机测试或中断回合计入成绩**。已完成结果见 [results](docs/results.md)，过程见 [工作记录](docs/progress.md)。

## Run locally

Python 3.10+，核心示例只依赖标准库。在仓库根目录运行：

```bash
PYTHONPATH=src python3 -m roborag.demo
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

demo 在临时目录构建长期库，模拟一次插入失败后的检索，打印两路检索结果，再关闭本次短期库。所有数据、动作和机器人标识均为合成示例，不连接硬件或模型服务。

真实 RoboDojo 环境与模型传输配置见 [runtime](docs/runtime.md)。配置完成后，用策略 Python 环境运行：

```bash
python scripts/fetch_demonstrations.py --task stack_blocks --episodes 0,1,2
python scripts/build_visual_bank.py --manifest data/runtime-data/stack_blocks/train/sources-0-1-2.json --output data/corpora/stack_blocks/v2.json
python scripts/run_episode.py --task stack_blocks --mode dual --layout 0 --long-k 8 --short-k 4 --effort medium --corpus data/corpora/stack_blocks/v2.json --output runs/stack-dual-l0
python scripts/summarize_runs.py --output runs/summary.json
```

## Repository

```text
docs/                 动机、双库设计与实验计划
src/roborag/schema.py  经验记录、当前查询与上下文结构
src/roborag/memory.py  长期快照与 episode 短期库
src/roborag/retrieval.py  两路检索与上下文组装
src/roborag/demo.py    合成案例
src/roborag/robot_memory.py  真实视觉片段检索与上下文组装
src/roborag/vision.py  多视角与初始场景编码
scripts/              数据入库、仿真启动和结果汇总
tests/                隔离、时序与检索边界验证
```

机器人接入复用 RoboICL 的观测/动作协议、IK 与 RoboDojo 适配器；项目动机从 ICL 与检索展开，工程依赖来源在 [runtime](docs/runtime.md) 中明确记录。大型轨迹、图像、索引、运行日志和凭据均不纳入 Git。
