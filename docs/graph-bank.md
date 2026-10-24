# 交互经验图库 v0.2

这是 RoboRAG: Interaction-Aware Retrieval-Augmented Manipulation 的首版离线图库与本地浏览器。
重点是把现有示范构造成可浏览、可检索的图。节点使用自然语言，不要求预先归入固定交互类别。

## 打开前端

```bash
python3 scripts/graph_bank.py serve --port 8772
```

打开 <http://127.0.0.1:8772/>。启动器使用 `local.runtime.json` 中的 `policy_python`；
无此配置时使用当前 Python，需要 h5py、numpy、Pillow、jsonschema。构图还需要现有已登录的 Codex CLI。

前端的入口是 RAG 图库：任务筛选、经验图概览、阶段内容检索。
打开一张图后可以点选阶段或局部指导节点、折叠指导树、缩放、查看节点 JSON 和导出整图。
节点的「示范片段」页支持三视角回放、关键帧跳转和下载对应窗口的原始动作 / 本体状态。

「检索与动作参考」入口：<http://127.0.0.1:8772/#/retrieve>。
可切换规划参考 / 执行参考查询，检查匹配理由、前后图像、14 维动作行，以及供 Agent 使用的精简上下文。
每张图的阶段详情都有「以此阶段查相似经验」按钮；它用该阶段的起始观测查询，并排除源图，方便查看跨示范匹配。
这是检索浏览器，没有在页面背后调用生成模型或控制机器人。

## 图结构

借鉴 TG-RAG 的 chain-of-trees：时间顺序上的阶段构成主链，每个阶段包含场景线索、操作、反馈三种局部指导内容。
这是机器人示范上的轻量改编，并非 TG-RAG 完整算法复现。

**图的角色是历史交互经验与证据；当前任务的新 plan 由 GPT 生成。**
graph 的 `semantics.role=recorded_experience` 明确这一点。`next` 边仅表示这条示范发生过的时间顺序，
不表示所有场景都必须遵守的规划依赖。自动描述也不等于已经验证的前置条件或因果规律。
因此不需要把每条示范改成唯一标准 plan，也不应将相似示范强行合成一个“正确答案”。

- `phase` 节点：自然语言名称、局部目标、对象、时间窗口、关键帧。
- `cue / action / feedback` 子节点：该阶段的场景线索、示范操作和可见反馈。
- `next` 边：已记录的阶段顺序。
- `detail` 边：阶段与局部指导内容的关联。

时间窗口使用 `[start_frame, end_frame)`。图的来源和 Codex 提取配置保存在 JSON 中。
当前图是自动提取的参考经验，阶段边界是估计值；尚未作为人工标注 GT。
当前提供库侧检索与上下文接口，尚未接入在线执行器，也没有从初始图像直接生成 plan 的功能。

## 构图库

```bash
python3 scripts/graph_bank.py build \
  --manifest data/runtime-data/stack_blocks/train/sources-0-1-2.json \
    data/runtime-data/cover_blocks/train/sources-0.json \
    data/runtime-data/put_bottles_into_dustbin/train/sources-0.json \
    data/runtime-data/deposit_coin/train/sources-0.json \
    data/runtime-data/stack_blocks/train/sources-3-4-5.json \
  --output data/graph-bank/v1
```

构图器从原始 HDF5 生成带帧编号的多视角接触表，将夹爪变化和本体状态摘要交给 Codex，
提取自由命名的阶段与局部指导树。每条示范完成后立即更新图库和索引。
已有图自动跳过，已有标注会被复用；更换模型或提示后应指定新的输出目录。
`--prepare-only` 只生成接触表，`--limit` 限制本次处理条数。
默认使用 `gpt-6-astra / medium`，可用 `--model` 和 `--effort` 修改。

已有图只需本地重建索引，不必重新调用 Codex：

```bash
python3 scripts/graph_bank.py reindex --horizon 10 --stride 5
```

构图脚本结束时也会自动更新执行索引。重建读取原始 HDF5，需配置的 Python 环境含 h5py、numpy、scipy。

```text
data/graph-bank/v1/
  bank.json              图库目录与图摘要
  graphs/*.json          每条示范的完整 graph
  search-index.json       以阶段及其指导子树为单位的 BM25 索引
  execution-index.json    任务描述、阶段前置情境、短动作窗口与起始状态
  build/<graph-id>/       接触表、标注、Codex 调用记录
```

原始示范不复制；前端按 graph 的 source 引用读取对应 HDF5。
这些生成数据保存在项目已忽略的 `data/` 下，不随源码提交。

## 检索设计：完整图、阶段和动作分别负责什么

1. **规划参考检索**：任务过滤后，分别比较目标和初始场景描述，返回少量历史阶段链、初始情境、历史结果与来源。
   返回字段为 `episode_trace`，不输出当前任务的可执行 plan。GPT 用当前初始帧、目标和这些示例生成自己的 plan。
2. **阶段级检索**：将当前观察匹配到历史 `cue`；将子目标匹配到阶段目标与示范操作。
   历史 `feedback` 作为返回证据，不能用于匹配当前观察，以免把示范的未来结果误当成当前前置条件。
3. **阶段内动作参考**：阶段下面关联 `(observation_before, actions, observation_after)` 窗口。
   提供兼容本体状态时，按窗口起始位姿与夹爪命令值距离选窗；未提供时取阶段起点，并明确标注。

实现使用分字段 BM25、少量中英文词汇归一化、明确持物/支撑物条件、手臂过滤，以及结果多样性选择。
持物和支撑物的自动识别目前是有限中文模式及白/黑/紫/木纹描述，不能当作通用视觉对象绑定器；可在请求中显式给出条件。
节点标题仍是自由文本，不需要强制归入统一阶段类别。尚无学习型文本/图像向量索引、自动物体定位或通用语义重排器。
当前只有 9 张图，使用本地小索引；是否引入向量检索应由真实漏检例子决定，没必要先部署图数据库。

相似 graph 保留各自来源、顺序与动作证据。检索输出每条示范至多一个参考，并对重复操作描述施加轻量惩罚，
避免多个重叠窗口占满上下文。不同条件的示范不会被平均为一条轨迹，也没有被合并成一张虚构的大图。
如果输入是抓取失败而库里没有明确匹配的失败前置情境，返回缺失提示，不将普通成功示范包装成恢复策略。

这个分字段设计借鉴了 [Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval) 的保留局部上下文思路；
不是对该方法的完整复现，也没有把其检索指标当作本项目结果。

## 动作示例的含义

当前 9 条示范产生 **567 个重叠窗口**，默认最多 10 步、步长 5 帧；25 Hz 下最长 0.4 秒。
窗口不跨越阶段边界，尾部可能不足 10 步；必须有真实的 `state[end]`，因此没有伪造示范最后一个动作之后的状态。
这些窗口是参考样本，不等同于 567 个技能或成功标签。

动作转换与 `vendor/RoboICL/roboicl/policy/reference_data.py` 一致：每行左右臂各 7 维，
依次为 `dx,dy,dz,rx,ry,rz,gripper_target`。平移是世界坐标系米；旋转是世界坐标系旋转向量弧度。
首行从窗口起始实测末端位姿计算到控制目标的增量，后续行从上一个控制目标计算到下一个目标的增量。
源末端位姿使用 `xyz + quaternion(wxyz)`。夹爪状态是命令侧值，不能据此认定夹住了物体。

完整动作参考包括：

- 原始示范的哈希、图、阶段、时间窗口与动作接口。
- 动作前后本体状态，以及头部和双腕图像引用。
- 连续数值动作行；实际末端位移、最大指令平移量、夹爪命令起止值。
- 历史阶段文字反馈；它可能发生在短窗口之后，与窗口末端观测分别保存。

它能给 Agent 提供局部运动形状、位移尺度、左右臂配合与夹爪时序的参考。
当前对象位置、支撑面高度、接触与抓取状态，仍需当前观测来确定；不会自动对齐或重定向历史轨迹。
控制器限幅仍以当前执行接口为准，历史动作不会在入库时静默裁剪。
这是一项待验证的 action chunk 生成支持方案，尚未证明执行误差或任务成功率得到改善。

## Agent 接口与成本控制

```python
import requests

result = requests.post("http://127.0.0.1:8772/api/retrieve", json={
    "mode": "action",  # plan 表示为 GPT 规划检索历史参考，不是生成 plan
    "task": "stack_blocks",
    "goal": "把白色块放到黑色底块上",
    "observation": "右爪夹着白色块，黑色底块已摆在中央",
    "arm": "right",
    "held_object": "白色块",
    "target_object": "黑色底块",
    "k": 2,
    "max_context_chars": 6000,
    "include_actions": False,
}).json()

context_for_model = result["model_context"]
# 只将这一字段加入 Agent 上下文，避免再次发送 results 中的展示字段。
# 需要数值和画面时，读取 results[i]["chunk"]["url"]。
```

`include_actions=true` 会将完整数值 transition 放进上下文；预算不足时退回引用，不截断动作数组。
如果当前已有本体状态，传 `state`（如 `right_ee_pose`、`right_ee_joint_state`）和一致的 `action_space`。
只有双方机器人、世界坐标原点和接口确实可比时才应声明相同接口；当前不做坐标系转换。
返回的图片 URL 是本地资源引用，调用方需将选中的图像加载为模型的多模态输入；文本 URL 本身不代表模型看过图像。

索引、图文件按文件版本缓存，查询本身不调用 LLM。原始视频/动作保留在 HDF5 中，默认只返回引用；
少量代表经验按需展开，避免每次把所有 graph 和数百行控制数据发送给模型。
页面报告字符数、字节数和本地查询耗时，不将字符数冒充 token 或实际费用。
离线 Codex 构图和下游 Agent 的推理仍有成本；当前没有端到端成本/收益实验。

## GPT 规划环节：当前已支持与尚未接入的部分

预期流程是：当前初始图像 + 任务 → GPT 形成检索问题 → RAG 提供经验 → GPT 生成新 plan；
执行时：当前图像/本体状态 + plan 中的子目标 → RAG 提供局部动作证据 → GPT 生成新的 action chunk → 执行后再观察。
这里的检索问题可以由同一个 GPT 在正常决策过程中构造，不要求额外调用一次模型只做查询改写。

当前实现的是中间的经验索引、检索、证据读取、上下文封装和可视化；尚未实现两端的 GPT 规划器与在线闭环。
返回 `model_context.purpose` 区分 `evidence_for_gpt_planning` 与 `evidence_for_gpt_action_generation`，
并明确要求根据当前目标/图像/状态生成新计划或动作、绑定当前物体、引用采用的来源。
索引中的自动文字、源数据中的数值测量、GPT 对当前场景的判断应分别标明，不能混成一种“事实”。

未来的 GPT plan 可先用简单的 `steps` 列表，每步保留 `subgoal`、当前对象绑定、`completion_observation` 和 `evidence_refs`。
这些字段是规划接口建议，不是当前图库已经生成的计划。GPT 可以调整顺序、跳过已完成步骤或组合不同经验的建议；
不能将不同坐标系的历史数值轨迹直接拼接。
经验 graph 与生成的 plan 可以共享阶段结构，但前者记载历史，后者表达待执行意图；节点无需一一对应。
示范中提取的阶段链也只是已演示的一条路径，不能自动成为所有初始场景的唯一 GT plan。

## 浏览接口

保留全文 BM25 用于查库；Agent 使用上面的分字段 `/api/retrieve`。
全文查找包含阶段及完整指导文本，返回 graph ID 和节点 ID，可回原图查看前后阶段。
分数是相关性排序值，不是成功概率。

- `GET /api/bank`：图库目录。
- `GET /api/graph?id=...`：完整图。
- `GET /api/search?q=抓取&task=stack_blocks`：阶段检索，task 可省略。
- `GET /api/export?id=...`：下载图 JSON。
- `GET /api/frame?id=...&frame=0&camera=cam_head`：原始示范画面。
- `GET /api/trace?id=...&node=p01`：节点对应的动作与本体状态窗口。
- `POST /api/retrieve`：任务/阶段检索、动作引用与预算内模型上下文。
- `GET /api/retrieval-info`：执行索引规模和方法说明。
- `GET /api/chunk?id=...`：一个短窗口的完整实测参考，`download=1` 可下载。
- `GET /api/query-example?id=...&node=p01`：从阶段起始观测生成查询，排除该源图。

## 验证

```bash
PYTHONPATH=src <policy_python> -m unittest discover -s tests -p 'test_graph*.py' -v
node --check web/graph-bank/app.js
node --check web/graph-bank/retrieval.js
```

12 项测试覆盖图/API、角色颠倒、未来反馈排除、同接口状态选窗、去重、预算，以及积分恢复原始控制目标和真实末端观测。
本地示例结果保存在 `results/graph-bank-retrieval-check-20261024.json`：
初始场景查询得到两个历史阶段链参考（2,064 字符）；白块/黑色底块查询得到 `stack_blocks-0003:p05`（1,304 字符，默认只含动作引用）；
缺少恢复数据的失败查询返回空参考。数值仅对应这些示例，不能作为留出集检索准确率或机器人成功率。

下一步研究验证应在相同输入和上下文预算下比较：无经验、仅阶段文字、阶段 + 实测动作转移。
先观察正确阶段召回与条件冲突，再比较执行误差、成功率和实际 tokens/延迟，才能判断 RAG 是否值得其成本。
