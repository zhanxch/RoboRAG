# 开发与维护

## 当前维护规则

新增任务先修改 `tasks.py` 的 `TaskSpec`，再补提示词及必要的独立测量模块。共享的检索、动作、会话和评估流程不复制到新的任务 runner。新增仿真后端实现 `control/backend.py` 定义的派发协议，并对坐标、夹爪、缓存和原生终止行为做集成验证。

模型可见的自然语言协议在 `resources/` 中；Python 注释与格式化不应隐式更改它。若有意修改提示词、工具说明、schema、动作限制或参考内容，应使用新的策略版本和回合输出目录，保留此前结果。

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
- `hexashell_adapter / rag_sim / scene_measurements / planar_measurements` 归入 `backends/`；移除未被当前链路使用的旧 CLI Registry。
- `rag_runner / episode_batch / hexashell_agent` 归入 `runtime/`；`evaluation/` 统一原生结果与用量口径，suite 支持明确的任务参数。
- 提示词、模型工具说明和图库页面成为包资源；提供 `roborag` CLI、依赖声明、doctor 及 wheel 分发。

旧 `memory/retrieval` 双记忆原型、旧视觉库、独立布料学习/抓取试验、旧 dashboard 和备选控制器不再在当前源码中并列维护。它们已包含在整理前提交 `866f7f2`，需要研究历史实现时从 Git 查看。当前脚本目录仅是兼容旧命令的薄包装，不放业务逻辑。

旧文档以 Git 快照为准；当前文档只保留本 README 和五份主题说明。实验配置与结果的历史证据独立归档，不删除失败记录，也不从不同版本挑选成功回合。

## 仍需继续改进的边界

本次是代码组织与接口整理，不声称所有架构问题已解决。原生运行仍依赖本机 checkout、文件轮询和外部仿真环境；fold 几何工具的能力边界可进一步收紧；检索的本体状态/阶段条件尚未进入模型工具请求；原始库的部分检索限制未呈现给模型。后续应每次选择一项行为改动，配套独立评估，避免与性能优化混在同一冻结批次。
