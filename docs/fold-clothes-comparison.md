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
  标准布局 0、1 试跑进行中，原始目录 `runs/rag-fold-pilot-v1-standard-l{0,1}`。
- v2：准备复用基线只读 RGB 测量，并采用重复动作段编码。初始帧离线测量约 0.7 秒，
  该工具测试不构成执行成功或成本改善证据。测量没有返回上游折叠方案。

## 运行和汇总

```bash
<policy_python> scripts/run_rag_batch.py --task fold_clothes --effort xhigh --parallel 2 \
  --cases-file experiments/rag-fold-matched-cases.json --bank <frozen-bank> --output runs/<revision>-matched
<policy_python> scripts/compare_fold_clothes.py --reference results/hexashell-fold-reference.json \
  --runs runs/<revision>-matched --planned 20 --output results/<revision>-matched.json
```

批次启动记录源文件与图库哈希，期间文件改变则拒绝混入新版本；每个回合还保存实际 prompt 和原生结果哈希。
汇总拒绝重复布局、不同模型/推理档位或混合代码/提示/图库配置。代码维护和实验结果分开提交。
