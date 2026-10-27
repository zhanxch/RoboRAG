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
