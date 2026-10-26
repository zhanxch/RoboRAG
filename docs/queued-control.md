# Queued 控制第一阶段

2026-10-23：已实现可选的单模型同步运动队列，对应 [控制 TODO](control-todo.md)
中 P0/P1 的执行账本、动作展开、分窗口续接和基础观测统计部分。
它还没有 Astra/Luna 监督、结构化任务阶段授权、异步控制、目标跟踪或独立急停。
当前通过确定性测试与真实上游 policy 的假模型集成测试；尚未获得该新配置的完整仿真任务成功证据。

## 运行入口

使用 [runtime](runtime.md) 中的策略 Python 环境：

```bash
python scripts/run_episode.py \
  --task put_bottles_into_dustbin --profile queued \
  --model gpt-6-astra --effort xhigh --mode zero \
  --layout 0 --seed 0 --role development \
  --output runs/bottle-queued-development
```

`queued` 当前仅允许瓶子入桶任务，默认 `zero`、`xhigh`，输出目录必须不存在。
显式指定 `--mode fixed/long_only/dual` 可接原有检索条件，但需要对应训练库；
这些组合尚未取得新队列的任务收益证据。
原有 `standard` / `hybrid` / `continuous_fast` 默认仍为 `dual`、`medium`。
历史 `hybrid` 的截断实现保留作对照，完整队列通过 `queued` 单独启用。
训练片段转换现已绕过实时控制 mixin，避免把录制的动作行重新展开或截断。

模型名从启动器、policy、Responses bridge 一直传到实际 CLI `--model`。
每次实际调用的模型、档位和命令均记录在 `decisions/call-*/invocation.json`；
缺失的 token/cache 计数保留为 null。上游预算检查会在必需的输入/输出计数未知时拒绝动作。

## 一个决策如何完成多个执行窗口

模型的 `act` 必须明确 `control_mode`：

- `transit`：1–6 个稀疏位移行，每臂每行不超过 0.30 m / 0.60 rad；两夹爪保持最新 commanded 开度。
- `precision`：1–8 行，每臂每行不超过 0.01 m / 0.15 rad；可以显式改变夹爪指令。

一次请求先验证所有行，再完整展开为不超过 0.03 m / 0.15 rad 的动作。
超过 64 步容量则整段拒绝，不执行前缀。模式、夹爪和行数限制由代码检查。
几何分段并不是速度/加速度限制，也不提供碰撞检测。

增量相对请求时的最新 EE 状态仅转换一次，生成绝对目标并存入队列。
例如四行各 0.30 m 的同向位移生成 40 步，以 32＋8 步派发；
第二窗口继续原绝对目标，不重新基于新状态累计相同增量，也不再次请求模型。
这是软件语义例子，不代表这条长直线在机器人工作空间中可达。

一个模型决策对应一个 execution 和一个最终工具回执；中间窗口不会生成新的模型调用。
原有执行回执的 predicted/dispatched/executed/unexecuted 计数覆盖整个段。
模型调用预算由 policy 维护，仿真端独立 `sim-harness.json` 不把派发窗口数当成模型调用数；
原生任务步数上限仍然有效。

## 中断、身份和证据

队列核心在 [execution.py](../src/roborag/execution.py)，上游接入在
[queued_policy.py](../src/roborag/queued_policy.py)。核心保存：已接受、已派发、
已确认执行、在途、尚未派发、已作废、结果未知等计数。
同 episode/request 和同参数提交幂等，不同参数拒绝；计划版本、队尾与窗口 ID 都需匹配。

只有逐步 `executed_action` 与派发绝对目标一致，才推进执行账本。
缺少回执、动作不一致或重复取动作都不能触发重放。
第二窗口的 IK 拒绝索引会转换为整个模型段内的索引。
跟踪中断作废剩余段；窗口末尾尚未消退的一次大残差也会阻止下一窗口，
避免上游在窗口间重建 TrackingGuard 而漏掉连续异常。

取消/故障时，尚未派发的动作可明确作废；已派发但未确认的动作先标为 unknown，
锁住新派发。`confirm_discarded` 只能由知道旧窗口已经停止且后缀未执行的适配器调用。
当前接入仅在同步 rollout 返回并核对执行前缀后提供此证据。
它不能作为异步机器人已经停下的替代证明。

队列本身提供 pause/resume/cancel/fault 状态操作，不下发物理停止。
当前仿真内的中断仍依赖既有执行侧 IK/TrackingGuard；网络请求之外的独立看门狗尚未实现。
夹爪状态仍是指令值，队列 `succeeded` 只表示所有命令获得执行回执，`task_success` 始终为空。

每次派发/收尾写入 `policy/<episode>/executions.jsonl` 与原子替换的 `queue-state.json`。
快照恢复接口将未完成执行锁为 unknown；运行器不自动恢复旧轨迹。
episode reset 使用新 ID，隔离旧请求和回执；真实本体停止/场景重置由上层执行端负责。

## 验证与下一步

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

策略环境运行全部测试；无机器人依赖的普通 Python 会跳过对应运行时测试，
仍运行纯 Python 队列与边界验证。重点覆盖：超过 32 步不丢失、一次推理跨窗口、
第二窗口 IK 拒绝、边界残差、未完成窗口重复取动作、终止前缀、未知结果锁定、
快照损坏和重置隔离。测试使用假模型和合成观察，不替代真实运动/安全验证。

下一步先完成 `queued` 的开发布局实测，再按同一执行器比较 Astra 单模型与
Astra/Luna 分工。异步调度、任务阶段授权和感知扩展仍按控制 TODO 继续，
不能据本阶段测试结果宣布三层架构或机器人实时保护已完成。
