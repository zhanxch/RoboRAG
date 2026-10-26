# 当前实验入口

按任务保存固定场景列表，运行器仍使用统一的 `roborag batch --task <任务名>`。

- `fold_clothes/all.json`：seed 1 的 50 个固定场景。
- `push_T/development.json`、`push_T/heldout.json`、`push_T/all.json`：分别为 6、44、50 例。
- `selected-versions.json`：所选策略来源、提示词哈希、控制器和数据位置。

逐个开发新任务时，在同名目录新增场景列表，再保存独立的运行记录和结果报告。新代码运行使用新的输出目录和冻结记录；当前所选原生成绩不能自动转移到重构后的代码。`history/` 保留整理前的实验声明，供解释历史报告，不是默认运行入口。
