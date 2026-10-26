# RoboICL 公开基线验证

本轮评测使用 RoboICL 项目公开的逐布局分数作为外部 Baseline。
主要指标是 RoboDojo 原生任务进度（0–100），完整任务成功率单独统计。
**进度均分不是成功率；模型意图文字、相机测试和未完成回合不能代替原生成绩。**

机器可读结果见 [本轮比较](../results/roboicl-public-comparison-20261020.json)，
完整配置见 [冻结计划](../experiments/roboicl-public-3task-20261020.json)。
`complete=false` 表示评测尚未完成，不能将已完成子集当作整轮成绩。

## 本轮停止时的结果

2026-10-20 按用户要求停止全部实验，调度器、仿真器、策略服务和本轮模型进程
均已退出。原计划九回合，实际完整结束五回合，中断一回合，三回合未启动；
停止记录见 [执行状态](../results/roboicl-evaluation-stop-20261020.json)，
原生结果和来源哈希见 [证据快照](../results/roboicl-native-evidence-20261020.json)。

- `cover_blocks` 布局 2、3：均为 100 分、完整成功；公开同布局均为 100 分。
- `deposit_coin` 布局 2、3：均为 100 分、完整成功；公开同布局均为 100 分。
- `put_bottles_into_dustbin` 布局 2：40 分，未完全成功；公开同布局同为 40 分。
- 瓶子入桶布局 3 由用户中断，不计分；三个任务的布局 4 均未启动。

五个完整回合有四个成功，所有进度分均与公开同布局结果持平。
当前证据表明系统可以完成所测的遮盖积木和投币任务，但没有显示相对于
RoboICL 公开基线的优势。各任务完成数不一致，不报告原九回合计划的总体均分，
也不将中断的瓶子入桶回合算作失败。

五个完整回合共完成 209 次模型调用，已记录总 token 为 22,400,579；
这不是费用金额，也不能直接与公开 trace 的决策条数作计费效率比较。
[检索审计](../results/roboicl-retrieval-audit-20261020.json) 未发现已检查的
示范来源、时间边界或片段配额违规；实际最多使用 49 张图像。
审计也包含中断回合的调用日志，该回合不进入任务成绩。

## 公开数据

- [RoboICL 论文](https://arxiv.org/html/2609.34261v1)，主表为 Table 2。
- [项目公开逐布局数据](https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/assets/data/benchmark-rollouts.json?v=20260928-rollouts-v9)，快照日期 2026-09-28。
- [本地规范化记录](../results/roboicl-public-baseline.json) 保留来源 URL、原始文件 SHA256、任务名映射和逐布局身份。

导入核验得到 30 个任务、880 条完整评测、49 条部分评测。
公开的 30 任务 Overall 是任务均分的非加权平均；重算为 50.635，
论文四舍五入为 50.64。七个没有完整评测的任务在论文中保留零分，
不能据此宣称这些任务存在完整失败回合。部分评测不进入本轮同布局对照。

网站 `swap_t` / `play_xylophone` 分别映射到运行器的 `swap_T` / `play_Xylophone`；
配对键包含任务、seed、standard/random 变体和 layout，避免错误配对。
公开 trace 的 `decisionCount` 不保证包含 API 重试，不将它当作计费调用次数。

## 首轮范围与配置

选取三个支持示范的类别，各一个任务，每任务固定布局 2、3、4，seed=0，
standard 变体，共九回合：

- Memory：`cover_blocks`。
- Precision：`deposit_coin`。
- Long-Horizon：`put_bottles_into_dustbin`。

这些布局未用于本项目此前的控制器调优。每项任务只使用 RoboICL 锁定的
episode 0 训练示范，原始 HDF5 的 SHA256 与参考清单逐一核对。
检索库分别有 34、38、37 个有效动作片段，运行期间只读。
每个评测回合重新初始化短期记忆，评测轨迹不进入长期库。

冻结 `scene-phase-transition-v3` 检索实现，使用 12 个长期片段、
12 个短期片段、50 图像上限、`gpt-6-astra`、`xhigh`，
动作 horizon 沿用上游任务配置。运行至原生步数或成功终止条件，
不使用人为缩短的动作/调用上限。

这属于**当前算法在对齐后的评测配置上的验证**，不是重跑早先的
三示范候选、8+4、medium 试点。原有试点仍单独保留。
`stack_blocks` 不在 RoboICL 的 30 任务主表中，因此其已有成功结果不纳入该 Baseline 比较。
Open 类别也不纳入本轮：当前 `zero` 模式直接使用上游策略，
不能据此证明 RoboRAG 的双库检索效果。

## 可比范围

同布局对照与全任务公开均分分开报告。本轮三个任务的公开全布局均分分别为
100、78、62.5；对应布局 2/3/4 的均分分别为 100、100、55。
这两个口径不同，不能混用。

九份布局文件与官方冻结资产 revision 完全一致，见
[布局哈希审计](../results/roboicl-eval-layout-audit.json)。
所选布局直接使用的刚体、几何物体共 43 份 USD/描述/元数据文件也全部匹配，
见 [物体资产审计](../results/roboicl-eval-object-audit.json)。
这不等于验证了房间、材质、生成的机器人 USD 等完整传递依赖。

模型使用项目既有的已登录 CLI 传输，区别于论文的直接/中转 API，
也无法取得不可变的服务端模型快照。序列化文本、缓存行为和真实 token
预算仍可能不同，因此本轮是外部公开基线比较，不能单独证明检索的因果收益。
三个任务、每项三回合只能支持小规模验证，不代表完整 30 任务结果。

## 复现

使用 `local.runtime.json` 中的策略 Python。冻结计划记录本机代理环境；
另一台主机需要相应网络通路，不应静默切换模型或推理强度。

```bash
python scripts/import_roboicl_baseline.py \
  --snapshot data/baselines/roboicl-public-2026-10-20/benchmark-rollouts.json \
  --output results/roboicl-public-baseline.json

# 对每个任务执行；已验证的源文件会复用，已有库文件不会覆盖。
python scripts/fetch_demonstrations.py --task cover_blocks --episodes 0
python scripts/build_visual_bank.py \
  --manifest data/runtime-data/cover_blocks/train/sources-0.json \
  --output data/corpora/cover_blocks/roboicl-eval-1shot.json

# 核验冻结文件并汇总，不启动模型。
python scripts/evaluate_roboicl_suite.py \
  --plan experiments/roboicl-public-3task-20261020.json

# 真正运行。已完成回合复用；存在未完成回合时停止，保留证据供检查。
python scripts/evaluate_roboicl_suite.py \
  --plan experiments/roboicl-public-3task-20261020.json --execute
```

导入时间戳是规范化文件的一部分；重新导入后应创建新的冻结计划，
不能悄悄替换本轮锁定的 Baseline。计划中的哈希需要与实际本地文件一致。

## 启动兼容性记录

第一次启动在返回机器人动作前遇到直连 TLS 握手失败，记录保存在
`runs/roboicl-public-3task-20261020/`，不计分。
之后配置本机已有代理，并兼容新版 CLI 在回合开始前输出的准确启动提示。
校验器仍拒绝真实工具调用、未知错误及未完成推理。
这一修复不改变检索、动作生成提示或任务评分。
修订前冻结计划与异常说明保存在原运行目录，修订后的哈希在当前计划内。

后续启动还发现旧校验器将重连后成功的响应及启用系统代理时的准确启动提示
误判为失败。已增加精确识别，并要求最终仅有一个完整推理回合、没有真实工具调用、
没有 `turn.failed` 或未知错误；相应回归测试已通过。
这些启动异常发生在动作被模拟器接受前，已记录在计划的
`excluded_infrastructure_attempts` 中，未用其生成的动作重放或计算分数。
当前执行目录为 `runs/roboicl-public-3task-20261020-r4/`，使用两个仿真进程，
不比较并行运行的墙钟时间。

检索与开销审计可单独生成：

```bash
python scripts/audit_suite_retrieval.py \
  --plan experiments/roboicl-public-3task-20261020.json \
  --output results/roboicl-retrieval-audit-20261020.json
```

该审计检查训练源、两路配额、过去片段起点、最新已执行反馈和实际图像数，
并记录 token、已完成调用的耗时及输入缓存比例。它验证日志中的边界条件，
不将“检索发生了”解释为“检索改善了任务表现”。
