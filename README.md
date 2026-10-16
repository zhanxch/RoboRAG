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

## Current scope

当前版本是可运行的双库原型，包含：

- 离线 JSON 快照构建与加载；长期库只接收训练集中的示范或整理后的经验。
- episode 独立的短期内存库、只检索截至当前时刻可见的历史、关闭后清空。
- 机器人及动作接口匹配、两路独立检索配额、最新交互保留。
- 明确分离的上下文输出、快照哈希，以及用于验证机制的合成案例。

检索器目前使用词项集合相似度，预算以记录条数计。**尚未实现视觉 embedding、真实 token/image 预算、模型调用、机器人控制或仿真评测；合成示例没有性能结论。** 原型代码为新建实现，未包含上游控制框架代码。

## Run locally

Python 3.10+，原型运行及测试只使用标准库。在仓库根目录运行：

```bash
PYTHONPATH=src python3 -m roborag.demo
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

demo 在临时目录构建长期库，模拟一次插入失败后的检索，打印两路检索结果，再关闭本次短期库。所有数据、动作和机器人标识均为合成示例，不连接硬件或模型服务。

## Repository

```text
docs/                 动机、双库设计与实验计划
src/roborag/schema.py  经验记录、当前查询与上下文结构
src/roborag/memory.py  长期快照与 episode 短期库
src/roborag/retrieval.py  两路检索与上下文组装
src/roborag/demo.py    合成案例
tests/                隔离、时序与检索边界验证
```

下一步先确定一种仿真环境及其轨迹格式，完成真实数据入库，再固定控制模型进行检索消融。大型轨迹、图像、索引、运行日志和凭据均不纳入 Git。

