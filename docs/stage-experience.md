# 阶段 SOP 与历史经验

`staged-v1` 在原 TRAIN 图库之外增加 `tasks/<task>/sop.json`。规则来源是提交 `210fae4` 中三个任务已经使用的 `prompt.txt`，不是论文草稿或从示范时序自动推出来的流程。每个 SOP 锁定原提示词、原 execution-index 和对应图文件的 SHA-256；提示词或图库不一致时拒绝继续沿用映射，必须发布新的 SOP 版本。

SOP 的 `global_rule_refs` 和各阶段 `rule_refs` 引用原文的完整段落或列表项。例如 `p03.b06` 是第 4 段第 6 个列表项，索引从 0 开始计算段落、从 1 开始计算列表项。返回文字保持原文，不进行摘要改写。全覆盖检查保证每段原有策略至少出现在全局规则或一个阶段；模型启动时只接收全局规则，阶段细则在进入阶段时准备。倒球抓取阶段允许提前准备 `controlled_pour`，用于选择能够完成全倾倒的抓姿。

阶段模板是可用于当前计划的规则集合。叠衣的 `fold_sleeve` 可绑定两只不同袖子并实例化两次，示范里的左袖优先不构成约束；两袖之后翻折下摆来自公共任务要求。推 T 的接触、推移、释放检查和归位可以在当前计划内处理小幅修正、重新接触和换臂。倒球将 `controlled_pour` 与 `restore_place` 分开，前者只关联 3 段真实向外倾倒，避免词法相近时把扶正参考误作为倾倒动作。换手是有适用条件的分支。

`StageLibrary` 从独立映射构建 sidecar，不调用图库重建函数，也不写 TRAIN 文件。sidecar 的每项把 SOP 阶段连接到真实 graph phase、半开帧区间和已有 chunk IDs；版本同时记录 SOP、提示词、图和 execution-index 哈希。HDF5 的路径与来源哈希来自冻结图，本文所述关联没有重新核验 HDF5 内容或重新标注图像。

规则、历史事实、未来计划和当前执行记录分别保留自己的 role。阶段关联是人工阅读现有 TRAIN 操作标注后的对应关系，不是因果证明。混合片段可能同时包含上一动作释放和下一动作准备，因此可以映射多个阶段。历史反馈保留遮挡和不确定描述；阶段首末姿态、夹爪命令、空杯或近似对齐都不能单独证明当前任务成功。准备结果中的 `temporal_neighbors` 只允许查看相邻历史情境，不能当成阶段匹配或计划依赖。

检索先限定任务与 SOP 阶段的来源集合，再复用现有 BM25、前置情境匹配、操作臂过滤和每条示范的多样性选择。已有规则无需在线模型标注。没有匹配的失败前置情境时返回空例子和明确限制，不将普通示范伪装成恢复经验。几何始终来自当前 RGB 测量；本层不选择动作、不读取隐藏真值、不使用成功 oracle。

同回合内，版本、阶段和规范化查询完全相同的准备结果直接复用；信息缺口、状态变化或异常的显式 `refresh` 绕过缓存。诊断区分缓存命中、实际检索、刷新请求和已在上下文中的引用。缓存命中仅证明请求相同，不证明经验仍适用。实际采用哪项历史操作、如何适配当前场景和预期效果由执行记录保存。本层不将新的经历写回长期 RAG。

可复用接口：

```python
from roborag.experience import StageLibrary, persistent_guidance

initial_task_text = persistent_guidance("pour_balls_into_vase")  # 不需要加载图库
library = StageLibrary(bank, "pour_balls_into_vase")
overview = library.overview()  # 目标、条件、完成证据、分支和 lookahead
prepared = library.prepare("controlled_pour", observation="左手持杯在瓶口上方")
sidecar = library.export_index()  # 由会话写入回合目录，不写入 bank
```

这些检查验证规则完整性、版本隔离、阶段检索方向和缓存行为；不等价于原生任务成功率验证。旧默认 `legacy` 与冻结图库保持可回退，只有符合预先登记的原生验证要求后才应提升新架构为默认。

`tests/test_stage_experience.py` 使用临时合成词法图库，无私有 TRAIN、HDF5、图像或仿真依赖；规则覆盖检查直接读取已发布提示词与 SOP。真实冻结图库的本机检查单独保存在 `results/staged-experience-check-20261027.json`，记录全部阶段的查询、选中来源、版本、帧区间/chunk 校验以及运行前后未变的图 JSON 哈希。两类证据分别覆盖可移植的协议行为与本机数据映射。
