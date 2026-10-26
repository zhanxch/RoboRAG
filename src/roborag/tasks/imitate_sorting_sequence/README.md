# imitate_sorting_sequence

本目录是该任务的策略开发入口。任务配置见 [task.json](task.json)，任务提示词见 [prompt.txt](prompt.txt)。公共运行器、RAG 和动作协议复用 `roborag` 底层；仿真环境与成功判定由 RoboDojo 同名任务实现。

当前保留接入配置和提示词，未列入当前所选完整套件；验证状态以 [results/current.json](../../../../results/current.json) 为准。需要专用工具时，在本目录添加实现及 `tests/tasks/imitate_sorting_sequence/` 测试；需要固定套件时，在 `experiments/imitate_sorting_sequence/` 保存场景列表。

## 从单个任务开始

以下命令从仓库根目录运行，使用已配置的策略 Python；仿真依赖与环境配置见 [运行指南](../../../../docs/runtime.md)。

```bash
python -m roborag tasks --task imitate_sorting_sequence --json
python -m unittest discover -s tests -p test_task_registry.py -v
python -m roborag run --task imitate_sorting_sequence --seed 1 --variant standard --layout 0 \
  --probe-only --output runs/imitate_sorting_sequence-init-dev
python -m roborag run --task imitate_sorting_sequence --seed 1 --variant standard --layout 0 \
  --model gpt-6-astra --effort xhigh --output runs/imitate_sorting_sequence-single-dev
```

输出目录每次必须是新目录。初始化探针不调用模型、不派发策略动作、不计成绩；先确认单个开发场景，再冻结策略、执行固定套件。不要覆盖旧结果。新增任务与共享模块的边界见 [开发指南](../../../../docs/development.md)。
