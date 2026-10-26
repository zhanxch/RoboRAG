# fold_clothes

本目录是该任务的策略开发入口。任务配置见 [task.json](task.json)，任务提示词见 [prompt.txt](prompt.txt)。公共运行器、RAG 和动作协议复用 `roborag` 底层；仿真环境与成功判定由 RoboDojo 同名任务实现。

## 修改位置

- [tools.py](tools.py)：模型可见的 `measure_scene` 工具、参数校验与回合请求。
- [measurements.py](measurements.py)：调用 HexaShell 的视觉几何能力，并返回测量结果。
- [专用测试](../../../../tests/tasks/fold_clothes/test_measurements.py)：测量输入、边界和输出验证。
- [固定场景](../../../../experiments/fold_clothes/all.json)：标准与随机各 25 例。

沿用 v3 的布料测量与动作行为，任务提示词后拼接公共 `robot.delta.txt`。所选历史策略 `1e9ffa0` 为 37/50（标准 19/25、随机 18/25）。

这些是冻结策略的原生成绩，不是本次目录调整后的新评测。完整口径见 [结果说明](../../../../docs/results.md)。

## 从单个任务开始

以下命令从仓库根目录运行，使用已配置的策略 Python；仿真依赖与环境配置见 [运行指南](../../../../docs/runtime.md)。

```bash
python -m roborag tasks --task fold_clothes --json
python -m unittest discover -s tests/tasks/fold_clothes -v
python -m roborag run --task fold_clothes --seed 1 --variant standard --layout 0 \
  --probe-only --output runs/fold_clothes-init-dev
python -m roborag run --task fold_clothes --seed 1 --variant standard --layout 0 \
  --model gpt-6-astra --effort xhigh --output runs/fold_clothes-single-dev
```

输出目录每次必须是新目录。初始化探针不调用模型、不派发策略动作、不计成绩；先确认单个开发场景，再冻结策略、执行固定套件。不要覆盖旧结果。新增任务与共享模块的边界见 [开发指南](../../../../docs/development.md)。
