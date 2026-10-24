# 叠衣匹配实验记录

本记录区分当前实现、开发试跑和正式对比。RoboRAG 的研究方向是阶段感知的经验检索与使用；
当前实验比较完整执行系统，不能单独归因于 RAG。

## 对照与口径

对照为本机已完成的 HexaShell calibrated-v2 复现，标准布局 0–9、杂物布局 0–9，seed 0，每例一次。
原始结果路径与 SHA256 存在 `results/hexashell-fold-reference.json`，汇总器逐个校验原生结果。
这组基线成功 8/20（40%），平均进度 51，平均控制步数 445.35，平均回合墙钟 272.642 秒。
输入 11,540,197 tokens，包含缓存输入 10,682,496；输出 60,516，包含 reasoning 28,091。
两种细分都不能再次加到总数中。美元账单、模型 API 请求次数没有公开到逐回合日志。

RoboRAG 使用相同 GPT-6 Astra / xhigh、原生场景与 500 控制步预算。每个回合独立进程与目录；
同一版本最多并行两个回合。累计回合墙钟和并行批次耗时分别记录，不将并行吞吐当单回合提速。
控制表示存在差异：RoboRAG 让 GPT 生成 world link6 delta chunks，HexaShell 使用其冻结的运动工具。
工具调用/动作段数不能当作模型 API 调用次数。

两例开发试跑使用标准布局 0、1；它们仍包含在公开 20 例匹配套件中，因此完整套件不能声称是全留出测试。
每一轮修改单独测试、单独统计，不挑选不同版本各自最好的布局拼成一组结果。
完整对比还报告双方都成功的交集，避免以失败早停的短轨迹冒充效率改善。
未完成原生评价的回合保留为 incomplete，不能当成成功率的零分完整案例或零成本。

## 经验来源与成本

经验来自官方 TRAIN 的三条 `fold_clothes` 示范（episode 0、1、2），数据 revision
`269457a3b42d8d0ae69898a601f2c1487c2f826e`，逐文件哈希对照现有下载清单。
它们生成 3 张图、24 个阶段，与标准/杂物测试共享任务经验。自动阶段边界没有人工 GT 认证。
一次性标注使用 medium，输入 57,567、输出 4,747 tokens；详见 `results/fold-clothes-bank-cost.json`。
在线执行使用 xhigh，离线建库开销不混入每回合推理消耗，也不隐去。

## 开发版本

- v1：提交 `8b383db`，沿用堆叠已跑通的 delta 接口，GPT 自行进行 RGB 几何计算。
  标准布局 0 未成功（20 分、500 步），布局 1 成功（100 分、399 步）；
  对应 HexaShell 也是布局 0 失败、布局 1 成功，但成功回合用 479 步。
  RoboRAG 平均 987.282 秒，对照 231.943 秒；两例未缓存输入分别 314,260 与 76,113，
  输出分别 41,083 与 4,637。成功轨迹短约 16.7%，但耗时与 token 消耗明显更高。
  原始目录 `runs/rag-fold-pilot-v1-standard-l{0,1}`，完整对照 `results/fold-pilot-v1-comparison.json`。
- v2：提交 `15ec4e2`，复用基线只读 RGB 测量，并采用重复动作段编码。
  已核对几何工具源码哈希与归档基线相同。初始帧离线测量约 0.7 秒，
  两例真实执行均失败（20 分、500 步）；平均耗时 952.707 秒，未缓存输入合计 194,670，
  输出 36,694。比 v1 少约 38% 未缓存输入、少约 4% 耗时，但成功率从 1/2 降到 0/2。
  两例不能作显著性推断，也不能把工具离线成功当作叠衣改善。完整结果在 `results/fold-pilot-v2-comparison.json`。
- v3：GPT 可直接提交自己选定的当前指尖目标、朝向、夹爪和步数，由确定性转换器生成
  同一 14D 增量序列；保留 24 步上限和反馈检查。减少每段前的 Python 坐标换算。
  初始参考缩为一个闭爪事件窗口，后续按信息缺口展开；这轮同时改变接口与上下文策略，
  不能声称单一因素造成改进。原始历史动作仍可按需读取。

v3 第一次批次启动 `runs/rag-fold-v3-matched-20261024` 在第一个仿真器启动约 5 秒后遇到调度器日志缩进错误，
读取了仍在运行的进程尚未产生的 `result` 字段。模型未启动；此次基础设施失败保留，未算作完整评分回合。
修复后增加实际短命子进程的批次生命周期测试，完整重跑使用独立目录 `runs/rag-fold-v3-matched-20261024-r2`。

## 运行和汇总

```bash
<policy_python> scripts/run_rag_batch.py --task fold_clothes --effort xhigh --parallel 2 \
  --cases-file experiments/rag-fold-matched-cases.json --bank <frozen-bank> --output runs/<revision>-matched
<policy_python> scripts/compare_fold_clothes.py --reference results/hexashell-fold-reference.json \
  --runs runs/<revision>-matched --planned 20 --output results/<revision>-matched.json
```

批次启动记录源文件与图库哈希，期间文件改变则拒绝混入新版本；每个回合还保存实际 prompt 和原生结果哈希。
汇总拒绝重复布局、不同模型/推理档位或混合代码/提示/图库配置。代码维护和实验结果分开提交。
