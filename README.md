# RoboRAG

**Retrieval-Augmented In-Context Learning for Robot Control**

RoboRAG 探索如何为机器人持续选择有用的上下文：执行前构建长期经验库，执行中维护独立的短期经验库，再将两路检索结果提供给具有 in-context learning（ICL）能力的策略模型。

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
