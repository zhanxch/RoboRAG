# 开发与维护

## 当前维护规则

按任务逐个开发：先完成单任务的观察、工具、提示词和固定场景验证，再处理下一个任务。参考 HexaShell 的任务归属方式，专用代码留在任务目录；有真实复用需要时再提取共享模块。

```text
src/roborag/tasks/<任务名>/
  __init__.py       任务包，不在导入时启动环境
  task.json         任务标识、公共协议、能力和运行选项
  prompt.txt        本任务策略提示词
  tools.py          可选：任务专用 MCP 工具
  measurements.py   可选：任务专用测量实现
  README.md         开发入口、运行命令和结果来源
tests/tasks/<任务名>/          任务专用回归测试
experiments/<任务名>/         固定场景列表
```

共享的 `bank/` 负责 RAG，`control/` 负责动作协议，`runtime/` 和 `evaluation/` 负责运行与核验。这些流程不复制到任务目录。`tasks/spec.py` 只定义配置结构，`registry.py` 只发现和校验任务，不承载各任务的策略分支。

## 新增一个任务

1. 在 `src/roborag/tasks/` 新建与 RoboDojo 原生任务同名的目录，加入 `__init__.py`、`task.json`、`prompt.txt` 和 `README.md`。先确认仿真端已实现该任务及其评价器。
2. 编写最小配置，目录名必须与 `name` 一致。公共机器人协议和随机变体按实际支持情况选择。例如：

   ```json
   {
     "name": "new_task",
     "prompt_file": "prompt.txt",
     "protocol_prompt": "robot.basic.delta.txt"
   }
   ```

   默认只启用共享观察、检索、计划和动作工具，并传递原生公共任务指令；`random_variant` 默认关闭。`TaskSpec` 的字段说明以 [配置定义](../src/roborag/tasks/spec.py) 为准。注册表自动发现 `*/task.json`，无需增加中央任务列表。未知字段、重复标识、无效类型、缺失提示词和未知工具能力会在启动运行器之前报错。
3. 需要新测量时，在任务目录实现 `tools.py` 和 `measurements.py`，在 `capabilities.py` 登记名称、请求键和实现路径，再用 `task.json` 的 `capabilities` 启用。工具只读取允许的视觉/标定信息，不能用评价器状态代替感知；测量请求与返回值须有测试。已经满足需求的能力可以直接复用，无需修改共享运行器或给仿真派发增加任务名判断。
4. 在 `tests/tasks/<任务名>/` 添加测试与 `__init__.py`，在 `experiments/<任务名>/` 保存开发和评测场景。先跑离线测试、初始化探针、单个开发场景，再冻结策略并评估完整套件。不要把调试回合拼进正式成绩。
5. 更新该任务 README 和独立的结果报告；只有选为当前正式版本后才更新 `results/current.json` 与 `experiments/selected-versions.json`。任务配置不存成绩，`roborag tasks --task <任务名> --json` 从结果文件读取已记录成绩及其策略提交。

当前保留 11 个已有任务，没有创建 42 个空实现。注册表测试用 42 份合成配置验证扩展能力，不代表已完成 42 个机器人任务。

模型可见的自然语言协议在任务目录的 `prompt.txt` 和 `resources/` 中；Python 注释与格式化不应隐式更改它。若有意修改提示词、工具说明、schema、动作限制或参考内容，应使用新的策略版本和回合输出目录，保留此前结果。

新增仿真后端实现 `control/backend.py` 定义的派发协议，并对坐标、夹爪、缓存和原生终止行为做集成验证。

## 检查

```bash
python -m unittest discover -s tests -v
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir /tmp/roborag-wheel
```

测试使用合成 HDF5 图库，无需模型调用或原生机器人回合。HexaShell 几何集成测试在缺少其 checkout 时明确跳过；本机具备依赖时一并执行。黄金协议固定于已评估版本，不能为了让测试通过而直接用当前实现重新生成基准。

有本机历史数据时，再运行 `roborag verify-evidence`，验证实际检索返回文本与历史统计。原生行为验证仍需新的固定场景批次。

## 这次整理做了什么

整理前的主目录已有多代原型、多个控制入口、空 README，以及散落在工作树中的较新实现。当前源码以 push_T 冻结策略及其保留的 fold v3 行为为基础，形成一个分层包。

- `graph_bank / graph_chunks / graph_retrieval / graph_server` 拆为 `bank/`；存储缓存与 HTTP 展示分离。
- `rag_mcp` 拆为 `mcp/` 的服务组装、会话、schema 和工具模块；旧 `EpisodeMemory` 改称 `EpisodeSession`，准确表达用途。
- `action_targets / rag_execution` 归入 `control/`；仅保留 delta 执行和明确的关节归位路径。执行器通过后端协议派发，避免访问原生环境私有字段。
- `hexashell_adapter / rag_sim` 归入 `backends/`；cloth / planar 测量及其 MCP 包装归入各自任务目录。移除未被当前链路使用的旧 CLI Registry。
- `rag_runner / episode_batch / hexashell_agent` 归入 `runtime/`；`evaluation/` 统一原生结果与用量口径，suite 支持明确的任务参数。
- 提示词、模型工具说明和图库页面成为包资源；提供 `roborag` CLI、依赖声明、doctor 及 wheel 分发。

旧 `memory/retrieval` 双记忆原型、旧视觉库、独立布料学习/抓取试验、旧 dashboard 和备选控制器不再在当前源码中并列维护。它们已包含在整理前提交 `866f7f2`，需要研究历史实现时从 Git 查看。当前脚本目录仅是兼容旧命令的薄包装，不放业务逻辑。

旧文档以 Git 快照为准；当前文档以主 README、五份主题说明和各任务 README 为入口。实验配置与结果的历史证据独立归档，不删除失败记录，也不从不同版本挑选成功回合。

## 仍需继续改进的边界

本次是代码组织与接口整理，不声称所有架构问题已解决。原生运行仍依赖本机 checkout、文件轮询和外部仿真环境；fold 几何工具的能力边界可进一步收紧；检索的本体状态/阶段条件尚未进入模型工具请求；原始库的部分检索限制未呈现给模型。后续应每次选择一项行为改动，配套独立评估，避免与性能优化混在同一冻结批次。
