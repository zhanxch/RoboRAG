# 基础版测试名单

2026-10-25 用户确认，版本 v1。后续提到“基础版测试名单”均指本页的 RoboDojo 仿真任务集合。
共 **10 个基础任务，五个类别各 2 个**。机器可读清单为
[robodojo-basic-test-suite-v1.json](../experiments/robodojo-basic-test-suite-v1.json)。

## 固定任务

- **Generalization**：`push_T`、`fold_clothes`。
- **Precision**：`pour_balls_into_vase`、`deposit_coin`。
- **Long-Horizon**：`put_bottles_into_dustbin`、`organize_table`。
- **Memory**：`cover_blocks`、`imitate_sorting_sequence`。
- **Open**：`general_pickup`、`pick_from_conveyor_by_image`。

## 变体与统计口径

Generalization 的两个基础任务均评测 standard 和 random：

- `push_T` 与 `push_T_random` 合计一个基础任务。
- `fold_clothes` 与 `fold_clothes_random` 合计一个基础任务。

其余八个基础任务各有一个运行配置，因此总计 **12 个运行配置**。
standard 和 random 分别报告结果，再等权汇总为对应基础任务的成绩。
每类两个基础任务等权汇总；十任务总体均分也按基础任务等权计算，
不因 Generalization 包含更多配置而增加它的权重。

完整任务成功率与原生进度分分别报告。这里的总体和类别成绩均是所选子集成绩，
与其他方法比较时使用同一任务集合和明确的评测协议。

## 选取依据

每类按“基线表现相对较高项＋挑战项”配对，以上列举顺序对应这两个角色。
依据是 HexaShell、WorldScape-R²SI、EmbodiedEvo 的公开完整任务成功率，
来源数据版本为 2026-10-24。
[原始提案及基线快照](../results/robodojo-ten-task-proposal-20261025.json)保留数值、来源与选择理由。

`fold_clothes` 是用户指定的可变形物体案例，并非各方法普遍低成功率的任务；
Precision 的较高项也只表示相对档位。原始提案保留历史状态，正式名单以本页及 v1 清单为准。

## 冻结范围

本次固定名称、版本、任务集合、分类及变体计数与汇总口径。
布局、种子、重复次数、模型与控制器、示范及检索库权限、运行预算和对照方法另行确定。
既有开发调优布局需明确标记，与未参与调优的评测布局区分。
后续若调整任务集合，记录为新版本并保留 v1。
