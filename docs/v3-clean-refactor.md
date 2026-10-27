# 从原始 v3 开始的重构

本分支直接从 `1e9ffa05682fb750bd940938b6b8bd6bf6ac091f` 分出，没有以阶段式候选为底座，也没有移植它的执行架构。原始 v3 的 50 例成绩是 37/50；那是原版的成绩，新分支的效果需要单独测试。

## 当前执行路径

观察当前图像 → 按当前场景检索完整示范 → `record_plan` 记录模型自己的步骤计划 → 展开所需动作参考 → 执行短动作段 → 模型查看反馈并选择下一步。

v3 已经有显式计划，字段是 `current_scene`、`steps`、`adaptation`。每步包含 `subgoal`、`current_objects`、`completion_observation`、`evidence_refs`。历史 graph 是检索材料，当前计划不必按历史节点排列。模型的内部推理无法从这些字段完整观察；这里沿用可核验的原版接口，不另外设计 graph plan。

动作沿用原版 `actions`、`segments`、`targets` 和 `return_home`，同样的 14D delta、数值限制、图像反馈及控制器。没有阶段 SOP、阶段准入、检查点、固定阶段顺序、阶段限定检索或阶段状态驱动动作。

## 代码改动

- `rag_mcp.py` 的原始定义按职责移动到 `mcp/`，保留旧导入入口与工具注册顺序。工具函数、输入类型和回合状态类的 AST（包括说明文本）与原始版本逐项相等。
- `rag_runner.py` 仅补充新模块的来源哈希记录；策略、仿真、检索和提示词保留原版。
- 批次入口补齐显式 seed 的透传与去重，避免命令默认 seed 0 覆盖登记的 seed 1；这只决定测试场景，不改策略。
- `interaction_monitor.py` 是独立的只读日志查看器。它显示现有计划、当前动作的子目标、预期反馈、控制步和已返回反馈。它不会调用模型、写 agent 工作目录、提供模型上下文或限制动作。

监控标签直接取模型已经写下的子目标，不强制映射到预先定义的交互阶段。请求中的“抓住”只是意图；收到反馈、控制器 `plan_ok`、步数耗尽也不等于抓取或任务成功。成功仍由原生结果提供。

```bash
PYTHONPATH=src:.runtime/mcp python -m unittest discover -s tests -v
PYTHONPATH=src python -m roborag.interaction_monitor /path/to/episode --follow
```

监控输出到独立终端或另一个日志文件。关闭或崩溃不影响策略进程。原版轨迹已有 `mcp-calls.jsonl`、`commands.jsonl`、图像和计划，可以直接查看，不需要重新执行机器人来生成监控数据。

## 对齐验证

`tests/fixtures/v3-contract.json` 在移动代码前从原始提交捕获，禁止从新实现重生成以通过测试。测试比较全部 6 个模型工具的顺序、说明与 schema、组装后的提示词、核心策略文件哈希及移动定义的 AST。

`scripts/verify_v3_history.py` 使用原始图库，重放原版全部 50 例中实际发生的检索与动作参考请求，逐条比较模型可见文本。它不调用模型、不移动机器人；它验证重构的输入与检索语义，不能替代原生效果测试。

后续 fold 测试固定原版的 50 个场景、seed 1、原始图库、`gpt-6-astra/xhigh`、delta 控制器与每例 1800 秒预算。保存所有失败和中断；不通过反复挑选成功尝试宣称还原。测试期间不更改策略、plan 格式或动作方案。

旧阶段式候选按用户要求停止：6 例已完成，1 成功、5 失败；第 7 例中断，43 例未启动。详见 `results/context-restoration-stopped-20261027.json`，它不是完整 50 例成绩。
