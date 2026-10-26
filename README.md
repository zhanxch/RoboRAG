# RoboRAG

**RoboRAG: Interaction-Aware Retrieval-Augmented Manipulation**

RoboRAG 是一个统一的机器人检索增强操作框架。模型观察当前 RGB、相机标定和机器人本体状态，从 TRAIN 示范中检索历史经验，生成当前场景的新计划，再通过短动作段执行并接收反馈。

**当前只维护一套 `roborag` Python 包、一套 delta 控制协议和一个任务注册表。** fold_clothes 与 push_T 共用检索、MCP、动作转换、进程管理和评估代码；任务差异集中在 `TaskSpec`。它复用 HexaShell 的机器人几何与视觉测量能力，以及 RoboICL/RoboDojo 的仿真和原生评价接口；目前不是脱离这些依赖即可运行的独立机器人平台，也尚未完成真机后端验证。

## 当前版本与成绩

- **fold_clothes：选择 v3 `1e9ffa0`，37/50**，标准 19/25、随机 18/25。C1 为 30/50，observation-reuse 为 35/50，均不作为默认版本。
- **push_T：选择 `62f538d`，50/50**，标准与随机各 25/25。其中 6 例属于开发复核，44 例属于该阶段新增场景，分别保留统计。
- 均为 seed 1 固定场景每例一次，使用原始 TRAIN 图库、`gpt-6-astra` / `xhigh`。这些是**重构前冻结策略的原生成绩**。当前代码通过离线回归与接口一致性检查，尚未重新跑原生全套仿真。

“最佳”指当前已完成对照中的工程默认选择，不代表统计意义上的全局最优。暂无完成的 no-RAG / interaction-bank 原生消融，不能把系统成功率归因于 RAG 的单独贡献。完整口径见 [结果说明](docs/results.md) 和 [机器可读汇总](results/current.json)。

## 开始使用

在 Python 3.10+ 的开发或策略环境中：

```bash
python -m pip install -e '.[dev]'
roborag --help
roborag tasks
```

`roborag` 与 `python -m roborag` 是同一个入口。安装开发依赖即可运行纯离线测试；原生仿真还需要本机 RoboDojo/Isaac Sim、HexaShell、数据和策略 CLI，配置见 [运行指南](docs/runtime.md)。

```bash
python -m roborag doctor
python -m unittest discover -s tests -v
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

已有本机环境将 MCP 2.3.0 放在 `.runtime/mcp` 时，使用配置中的 `policy_python`，并设置 `PYTHONPATH=src:.runtime/mcp`。普通完整安装无需此目录。

原生单例运行示例（会启动仿真并调用模型）：

```bash
python -m roborag run --task push_T --seed 1 --variant standard --layout 0 \
  --model gpt-6-astra --effort xhigh --output runs/push-t-refactor-dev
```

## 代码阅读顺序

```text
src/roborag/
  cli.py / config.py       命令入口与本机路径配置
  tasks.py                任务能力、提示词选择、随机变体
  bank/                   TRAIN 图结构、索引、检索、参考片段与浏览器
  mcp/                    模型工具、当前回合会话和调用审计
  control/                目标到 delta 转换、执行约束与关节归位
  backends/               RoboDojo 适配、cloth / planar 视觉测量
  runtime/                模型会话、仿真进程、批处理与来源冻结
  evaluation/             原生结果核验、成本统计、配对比较与离线回放
  resources/              唯一有效的提示词、工具说明和图库页面
```

建议从 `tasks.py` → `runtime/episode.py` → `mcp/server.py` → `mcp/evidence.py` / `mcp/motion.py` 阅读。详见 [架构与一次回合的流程](docs/architecture.md)、[图库说明](docs/graph-bank.md) 和 [开发与重构说明](docs/development.md)。

## 历史与证据

整理前的代码已提交为 **`866f7f2`**。旧控制器、旧原型和已被替代的说明可以从该提交恢复，不再作为当前源码入口。旧实验配置和结果保留在 `experiments/history/`、`results/history/`；冻结工作树、原始日志、TRAIN 数据和模型产物仍在本机，不纳入包分发。

`results/selected-evidence.json` 索引所选两个版本的 100 个回合及文件哈希。具备本机原始证据时，可离线复核：

```bash
python -m roborag verify-evidence --output results/refactor-evidence-check.json
```

这个命令读取历史记录并重放检索，不产生新的模型或机器人回合。
