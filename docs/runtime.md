# 安装与运行

## 环境分工

离线库、图库浏览和测试使用普通 Python 3.10+。原生回合在 Linux 上使用两个解释器：`policy_python` 启动 MCP 和模型 CLI，`sim_python` 启动本机 RoboDojo/Isaac Sim。GPU 驱动、Isaac Sim、机器人资产与第三方源码不由 `pip install roborag` 安装。

在开发环境中安装 `python -m pip install -e '.[dev]'`；部署策略环境可用 `'.[runtime]'`。MCP 固定为 2.3.0，因为工具 schema 是策略协议的一部分。基础包依赖 numpy、scipy、h5py 和 Pillow；HexaShell 集成测试还需要其环境中的 OpenCV 等依赖。

本机原有环境通过 `.runtime/mcp` 加载 MCP 及其依赖；运行命令前可设置：

```bash
export PYTHONPATH="$PWD/src:$PWD/.runtime/mcp"
```

随后用 `local.runtime.json` 中的 **policy_python 原路径**代替示例中的 `python`。不要把 venv Python 符号链接解析到基础解释器；包内配置已保留该路径语义。

## 本机配置

复制 `local.runtime.example.json` 为 `local.runtime.json` 并填写解释器、RoboICL、资产和仿真动态库路径。路径可以是绝对路径，或相对于仓库根目录。配置是本机信息，不提交到 Git。

仓库还需具备：

- `vendor/RoboICL`，包含其 RoboDojo 原生环境，以及 push_T 使用的 `ik_continuity.py`。
- `vendor/HexaShell`，包含机器人几何、cloth_view、table_plane、planar_contact，以及 pour_balls_into_vase 的 rim_geometry。圆口测量只依赖 RGB、标定与已有 numpy/OpenCV；实际工具源码进入运行来源哈希。
- `data/Assets` 指向配置的资产目录，TRAIN HDF5 位于图库记录的 `source.path`。
- `data/graph-bank/current` 为选定的原始 TRAIN 图库，详见 [图库来源](graph-bank.md)。
- 本机策略 CLI、bubblewrap 和 NVIDIA 工具。运行器沿用既有模型工作目录隔离配置。

运行 `python -m roborag doctor` 查看路径和策略依赖是否齐全。它不启动仿真或调用模型，也不能代替 GPU/仿真的真实启动验证。`--root /path/to/RoboRAG` 必须放在子命令之前；也可设置 `ROBORAG_ROOT`。完整原生运行当前面向配置好的仓库 checkout。

## 开发与批量验证

先用 `roborag tasks --task fold_clothes --json` 查看任务源码目录、配置和已记录结果。任务成绩注明对应策略提交；注册任务不表示已完成验证。

进入对应的 `src/roborag/tasks/<任务名>/README.md`，按单例 → 开发场景 → 固定套件的顺序推进。输出必须是尚不存在的新目录：

```bash
python -m roborag run --task fold_clothes --seed 1 --variant standard --layout 0 \
  --model gpt-6-astra --effort xhigh --output runs/fold-refactor-dev
```

只检查环境初始化可以加 `--probe-only`，它不调用模型、不得派发策略动作，不计入任务成绩。

固定套件示例：

```bash
python -m roborag batch --task push_T --cases-file experiments/push_T/all.json \
  --model gpt-6-astra --effort xhigh --parallel 2 --max-wall-seconds 2160 \
  --output runs/push-t-refactor-seed1
python -m roborag evaluate --task push_T --runs runs/push-t-refactor-seed1 \
  --cases experiments/push_T/all.json --output results/push-t-refactor-seed1.json
```

fold 使用 `experiments/fold_clothes/all.json`。push_T 的 development、heldout、all 列表分别对应 6、44、50 例。不得将重构后的新回合和旧冻结版本拼接成一份成绩。`--no-rag` 仅提供未来消融运行入口，不表示已有消融结果。

批次并发限制为 1 或 2，会记录 GPU 功率、内存和所属进程资源。总 GPU 功率可能包含其他工作负载；并发墙钟不同于每例墙钟之和。

## 输出与复核

每个回合保留 `run.json`、实际提示词、模型事件、MCP 调用、动作/命令记录、`episode_end.json`、退出状态和原生 `_result.json`。原生结果哈希、布局哈希和源码/图库哈希用于防止混用版本。

```bash
python -m roborag summarize runs/fold-refactor-dev
python -m roborag diagnostics --runs runs/push-t-refactor-seed1 \
  --output results/push-t-refactor-diagnostics.json
python -m roborag verify-evidence --output results/refactor-evidence-check.json
```

最后一条命令需要本机已保留的冻结回合；它仅验证历史结果和检索回放，不是原生策略重评。原始数据和冻结工作树被 Git 忽略，应另行备份；包的 wheel 不包含它们。

## 选择阶段式候选与回退

三个适配任务可显式选择新路径，输出仍必须是全新目录：

```bash
python -m roborag run --task push_T --seed 1 --variant standard --layout 0 \
  --architecture staged-v1 --model gpt-6-astra --effort xhigh \
  --max-wall-seconds 2160 --output runs/staged-push-development
```

`batch` 和 `mcp` 同样接受 `--architecture staged-v1`。省略选项或指定 `--architecture legacy` 即回退；旧提示词、工具合同和图库不需要迁移或恢复。新路径暂不支持 `--no-rag`，会明确拒绝，避免把未实现的消融记成有效对照。

每个新回合另存 `sop-index.json`、`gpt-plan.json`、不可覆盖的 `plan-versions/`、`episode-ledger.jsonl`、`execution-state.json` 和当前观测快照。SOP 和新代码进入来源哈希；图库或原提示词与 SOP 声明版本不符时拒绝运行，需发布新 SOP/索引并重新验证。不要通过修改冻结文件来消除版本错误。

若 `enter_stage` 的跨阶段校验失败，状态中的 `dispatch_gate` 会阻止后续动作。先补齐评估并成功进入就绪阶段；若需要在原阶段继续恢复，则用 `reason="revisit"` 或 `"exception"` 明确重新进入当前阶段并描述当前观察。两条路径都要求已有当前步的可见观测，普通重复 `entry` 不能清除限制。`revisit` 可复用已有参考，`exception` 会刷新准备；只有实际信息缺口才需要额外检索。

开发与正式比较预算、场景和采用门槛见 [阶段式协议](../experiments/staged_v1/protocol.json)。旧的 122 例成绩只属于旧路径；新路径必须单独报告原生成功、未完成、缓存与成本。持久状态用于同一回合的模型上下文恢复，不意味着支持崩溃后从任意仿真中点恢复动力学。

阶段式回合使用独立审计入口，可读取本轮基线索引中的原始回合：

```bash
python -m roborag evaluate-staged --task push_T --runs runs/your-frozen-push-suite \
  --cases experiments/staged_v1/push_T-regression.json \
  --baseline results/staged-baseline-audit-20261027.json \
  --output results/your-staged-push-report.json
```

该命令同时检查原生证据、SOP/图库版本、计划修订、动作归属和阶段证据；报告保存缺失、失败与未完成，不筛选最佳重试。退出码 0 表示报告成功写出，采用判断应读取 `gate`，不能把命令成功当作效果通过。开发单例可指定自己的单例 cases 文件并省略 `--baseline`，此时没有无回归结论。
