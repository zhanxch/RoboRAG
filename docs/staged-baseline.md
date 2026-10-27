# 阶段式架构改造的基线核验

2026-10-27 在实施阶段式 SOP、计划与执行改造前，先将既有成果保存为 `210fae415678444a9843e11c52f7e7ffc4eb1c5c`。该提交保存已验证的倒球 rim-v3 和此前重构回归记录，不包含新架构的效果声明。[机器可读审计](../results/staged-baseline-audit-20261027.json)记录源报告哈希、核验范围、成本与配对用例路径。

## 已核实的依据

- fold_clothes：冻结 `a8a3ee6` 的 [50 例回归](../results/refactor-fold-20261026.json)，37 成功，标准 19/25、随机 18/25。
- push_T：冻结 `a8a3ee6` 的 [50 例回归](../results/refactor-push-t-20261026.json)，50 成功。该报告的单例记录位于 `finished[]`。
- pour_balls_into_vase：[rim-v3 正式 22 例](../results/pour-balls-rim-v3-heldout-20261026.json)，16 成功；开发 0–2 另行保留，包括一次超时未评分，不能拼接为 25 例成绩。

独立核对了 122 份原生结果的 SHA-256、单次评分与成功标记，以及用例身份、场景哈希和运行证据文件的存在性；全部通过。仅读取终局、运行配置和工具名称等元数据，没有分析所选回归场景的逐步动作或图像。此检查不重新证明每次历史经验采用是否合理。

倒球冻结清单的 138 项来源在新开发开始前与实际文件全部一致。落盘审计时新开发已经修改运行器，因此改用基线提交直接验证其中 95 项受 Git 管理的源文件，另外 43 项保留的图库或外部源码仍逐项核验。不能要求新候选工作树与旧冻结代码一致，也不能把旧成绩转移给新代码。

基线状态执行了 104 项 unittest，全部通过且无跳过；`ruff check src tests scripts` 和 doctor 通过。命令输出保留在本轮 Codex 工具记录中，没有另存日志文件或声称日志哈希。doctor 不启动仿真，不代替原生回合验证。

## 本轮配对范围与预算

全部使用 seed 1、`gpt-6-astra`、`xhigh`、delta 控制器和原 TRAIN 图库。开发先运行每任务 standard layout 0；实际问题需要修正时，可追加 standard layout 1，同一版本同一用例只尝试一次并保存所有结果。

冻结后的预定配对范围是：

- fold_clothes：standard 3、4 和 random 3、4，已有基线 2/4 成功。
- push_T：standard 3、4 和 random 3、4，已有基线 4/4 成功。
- pour_balls_into_vase：standard 3、4、5、6，已有基线 4/4 成功。

12 例均有完整、哈希一致的配对依据，具体路径见审计 JSON。用例由实施协议预先指定，再查询元数据；这是历史场景配对回归和本轮开发保留范围，不是新的泛化留出集。候选须完整评分，成功数不低于匹配基线、不丢失基线成功用例，并具备完整结构证据；不满足则保留 legacy 默认。四例小样本不能支持完整无回归或 RAG 因果收益结论。

fold/push 开发与回归每例墙钟均为 2160 秒；pour 开发 2160 秒、正式配对 3600 秒。原生动作步数上限保持 fold 500、push/pour 600。超时与未完成保持未知，不能当作通过；全部尝试留存，不重跑取最好结果。

## 成本口径

本次使用最新原生回归的成本，不混用更早历史版本的成本段：

- fold：平均 385.44 控制步、687.49 秒；未缓存输入 4,007,437 token，缓存输入占 95.75%，输出 532,244 token。50 例共记录 77 次计划、52 次检索和 241 次参考展开。
- push：平均 320.24 控制步、621.26 秒；未缓存输入 3,770,334 token，缓存输入占 95.41%，输出 482,632 token。50 例共记录 81 次计划、128 次检索和 82 次参考展开。
- pour：平均 502.36 控制步、1505.38 秒；未缓存输入 2,585,560 token，缓存输入占 97.33%，输出 546,569 token。22 例共记录 81 次计划、31 次检索和 111 次参考展开。

完整 input/cached/output/reasoning 数值保存在审计 JSON。缓存输入是输入的子集，reasoning 是输出的子集；实际账单与 API 请求次数未知。工具次数不能代表模型请求次数，计划次数也不能单独证明无效重规划。比较应保留失败和超时成本，并同时报告缓存命中、未缓存输入、输出、控制步数和墙钟。历史与候选在不同时间运行，即使模型别名相同，也不能证明服务端权重和基础设施负载完全相同。

## 复核入口与保留事项

从仓库根目录运行，保留 venv 解释器原路径：

```bash
export PYTHONPATH="$PWD/src:$PWD/.runtime/mcp"
task_python=/home/zxc/文档/ChatGPT/infiniDojo/reproduction/astra_robodojo/.venv/bin/python
"$task_python" -m unittest discover -s tests -v
"$task_python" -m ruff check src tests scripts
"$task_python" -m roborag doctor
```

这些命令会验证当前 checkout；要复现基线测试需使用上述基线提交。运行器 `run` 默认 seed 0、墙钟 5400 秒，`batch` 默认墙钟 1800 秒，因此比较运行必须显式填写 seed 和预算。模型、effort、图库与候选版本也应明确记录，并使用新的输出目录。

原始 runs、冻结源码快照、图库和外部 vendor 资产不随基线提交完整进入 Git，须另行保留。本机已有 cloth-grasp dashboard 在调查时仍运行，应避免误清理。当前原生策略保留单 CLI 历史与 compaction，关闭在线 multi_agent；改变在线调度方式时应视为另一个实验因素。
