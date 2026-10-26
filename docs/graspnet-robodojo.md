# GraspNet 接入 RoboDojo：兼容性检查与接口方案

检查日期：2026-10-23。结论：**不能把 `graspnetAPI` 直接作为 RoboDojo 的 action 策略；可以把独立的 GraspNet 系列检测模型接成抓取候选生成器。**

## 本次实际测到了什么

复跑入口：`scripts/probe_graspnet_compatibility.py`；结果：`results/graspnet-compatibility-20261023.json`。

- 在当前策略解释器实际尝试导入依赖。Python 3.11.16、PyTorch 2.7.0+cu128、RTX 4090 可用；`graspnetAPI`、`pointnet2_utils`、`knn_pytorch`、`MinkowskiEngine`、`gsnet` 导入失败。后两项用于 GSNet/AnyGrasp，并非原版 baseline 都需要。PATH 中没有 `nvcc`。这描述当前解释器与 PATH，不代表整机所有环境都没有这些依赖。
- 对本地固定版本 RoboDojo 的真实 `ObsManager.get_obs` 方法执行了三组**合成传感器输入**检查，三组均通过：默认 RGB 不输出深度；仅启用导出 flags 仍不输出深度；同时提供深度 annotator 与导出 flags 后输出 float32 深度、K、T，深度数值原样保留。
- 实际读取 `stack_blocks/train/episode_0000000.hdf5`：25 Hz、三路 RGB、内外参，**没有深度**。这只覆盖该记录，不概括全部数据集。
- 保存了 RoboDojo revision、实际读取文件的 SHA256、导入错误和观测字段，便于复核。

**本次没有加载抓取网络权重、启动新的仿真抓取回合，也没有测量网络延迟或抓取成功率。** JSON 中延迟与成功率为 null，物理回合数为 0。合成观测检查不是实际渲染深度、标定或抓取可行性的验证；硬件性能仍待测试。

复跑命令（使用 `local.runtime.json` 中的 `policy_python`）：

```bash
/home/zxc/文档/ChatGPT/infiniDojo/reproduction/astra_robodojo/.venv/bin/python \
  scripts/probe_graspnet_compatibility.py \
  --output results/graspnet-compatibility-20261023.json
```

## 输入与输出的区别

[`graspnetAPI`](https://github.com/graspnet/graspnetAPI) 是数据集、抓取表示和评估工具，不是动作网络。当前源码版本为 1.2.11，README 提醒 PyPI 版本可能落后；其 `GraspNet` 类和 baseline 中的神经网络同名，不应混淆。

真正做预测的是例如 [`graspnet-baseline`](https://github.com/graspnet/graspnet-baseline)。其官方 demo 默认 `input_feature_dim=0`，将约 20,000 个 XYZ 点输入模型。深度图配合相机内参可生成点云；已有米制点云时不必再输入深度图。RGB 用于展示、目标识别或筛选等，默认网络的几何输入不要求颜色。只有 RGB 截图不能直接运行这个预训练检测器；额外预测深度会引入尺度和几何误差。[demo 源码](https://github.com/graspnet/graspnet-baseline/blob/main/demo.py)

输出是多个抓取候选，而非机器人关节速度或连续动作序列。每条 Grasp 表示为 17 个数：评分、开口宽度、高度、抓取深度、9 个旋转矩阵元素、3 个平移量、object_id。几何量使用米；抓取深度是夹爪几何/接近参数，不是输入深度图。`object_id` 字段本身不等于语义识别能力，也不能把评分当成经过校准的实际成功概率。[表示定义](https://github.com/graspnet/graspnetAPI/blob/master/graspnetAPI/grasp.py)

如果 action space 特意定义为“执行这个抓取位姿”，它可以成为高层 action 的主要参数；在本项目里仍要经过末端映射、IK、轨迹与夹爪控制，才能变成 RoboDojo 可执行的逐步 action。

## 需要增加的接入层

推荐流程：目标区域 → RGB-D/点云 → 抓取候选 → 目标与碰撞过滤 → 坐标/TCP 变换 → IK/轨迹 → 接近、闭爪、抬升 → 观测验收。

### 1. 开启实际深度采集

当前 `env_cfg/arx_x5.yml` 的 `observation.vision.depth`、`intrinsic_matrix`、`extrinsic_matrix` 为 false；`env_cfg/camera/camera_config.yml` 的深度 annotator 被注释。应在独立实验的运行时配置中同时启用：

```yaml
# 合并进 observation.vision
depth: true
intrinsic_matrix: true
extrinsic_matrix: true
approximate_depth: false

# 合并进 camera.annotator.cam_head，保留既有 RGB 配置
distance_to_image_plane_capture:
  type: distance_to_image_plane
  device: cpu
```

只开启头部相机即可开始单视角实验。必须在环境/捕获器初始化前合并；`depth: true` 本身不会创建渲染通道。继续核验采集帧与机器人状态的时间戳、RGB/深度对齐、有效深度比例和桌面尺度。

深度使用到图像平面的 Z 距离，已是米制时 `factor_depth=1`，不能重复除以 1000。输入点云需排除 NaN、Inf、零深度与范围外点；裁剪目标时仍需保留环境几何用于碰撞检查。公式为 `X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy`。使用与深度分辨率匹配的 K，不使用为 LLM 缩略图缩放过的 K。

### 2. 明确坐标系与夹爪几何

本地已存在 `roboicl/policy/head_calibration_capture.py`，使用实时 Camera API 读取 K 和 USD 相机姿态，并减去 environment origin，可复用。相比只按配置推导 K，它更适合校验当前渲染画面。腕部相机必须按观测帧更新外参。

本项目标定约定为 USD 相机轴：X 向右、Y 向上、-Z 向前；深度反投影的 OpenCV 轴为 X 向右、Y 向下、Z 向前。令 `T_A_B` 将 B 坐标转换到 A 坐标，则：

```text
F = diag(1, -1, -1, 1)
T_env_G = T_env_Cusd @ F @ T_Ccv_G
T_env_link6 = T_env_G @ T_G_link6
```

其中 `T_G_link6` 是针对实际夹爪几何标定的映射，需要考虑 GraspNet 抓取原点、手指方向和抓取深度；不能默认为单位矩阵。现有局部布料代码把 TCP 定为 `link6 + R*[0.145,0,0]`，说明至少存在 14.5 cm 的参考点差异，但这不是 GraspNet 到本机夹爪的完整标定。

同时处理四元数顺序：RoboDojo 相关姿态使用 wxyz，SciPy 默认 xyzw。开口宽度是米制，RoboDojo 的夹爪 command 是归一化开度，需要按实际夹爪映射，不可直接把 width 当 command。

### 3. 将目标姿态变成动作

先选择任务指定物体/区域，再在候选中检查夹爪最大开度、桌面碰撞、双臂冲突、可达性与关节连续性。网络的局部点云碰撞过滤不能代替整臂路径碰撞检查。

控制阶段应包括预抓取位姿、沿接近轴靠近、闭爪、等待接触稳定、抬升和保持验证。复用本项目的 IK 与执行回执接口；动作反馈决定重试或终止，不能以预测分数或闭爪命令直接记成功。当前模型入口仍是 dialogue policy，尚未新增 GraspNet policy server 或上述执行器。

## 实时性与维护状态

**推理时间：本机未测。** 不把 GSNet 或 AnyGrasp 的数据当成原版 GraspNet baseline 的速度。

- [GSNet 论文 §5.5 Table 6](https://arxiv.org/html/2406.11142v1#S5.SS5) 报告 RealSense/Kinect 输入的模型推理约 0.10/0.12 秒。这是论文条件下的模型时间，不含本项目的完整采图、传输、碰撞筛选、IK 与执行成本。
- [AnyGrasp 官方项目页](https://graspnet.net/anygrasp.html) 描述亚秒级密集抓取候选生成；这仍不是本机端到端延迟。
- RoboDojo 当前控制/观测频率为 25 Hz，即每个仿真步 40 ms。适合让抓取网络在准备抓取或需要重规划时更新目标，让本地控制器每步执行。不能仅凭约 100 ms 的检测结果声称整条闭环能达到 25 Hz。
- 当前 rollout 同步请求 policy action，再推进动作，因此“推理超过 40 ms”不意味着离线仿真不能跑，只意味着墙钟速度与真实机器人实时控制是不同指标。

正式计时应分别记录：采集、点云处理/传输、GPU 推理、NMS/碰撞、IK/规划、观测到命令总延迟。模型常驻并预热，GPU 计时同步，报告 p50/p95、有效候选率与无候选率，同时注明 Isaac Sim 是否共享 GPU。抓取实验另外报告成功率、滑落、误抓目标与桌面碰撞；不能混用模型 FPS 和任务吞吐量。

**维护与选型：** baseline 是 CVPR 2020 工作，README 仍给出 PyTorch 1.6 参考环境，需要编译 PointNet2/KNN 扩展；requirements 没有锁定同一套版本，不能据此保证本机 PyTorch 2.7/CUDA 12.8 即装即用。建议独立模型环境或容器，不改动当前仿真依赖。

若关注维护情况，可以评估 [AnyGrasp SDK](https://github.com/graspnet/anygrasp_sdk)：截至检查日，官方 README 列出 2026-07 的区域/接近方向控制及 Python 3.14、CUDA 13 支持更新。但它需要机器许可证，仓库说明通常五个工作日回复申请，不能视作无需授权的开源网络。若需要可改训练代码，可评估组织下的 [`graspness_unofficial`](https://github.com/graspnet/graspness_unofficial)，这是第三方实现的 fork，也需要另行处理旧依赖和 CUDA 扩展。此处未安装、加载或横向评测这些后端。

## 对当前叠衣服任务的适用性

建议先用 `stack_blocks` 的刚体局部拾取验证深度、坐标、夹爪映射和执行链。最小验证可预先固定 20 次场景/种子与抬升保持判据，再扩展到不同物体；这是后续实验建议，本次尚未执行。

对于 `fold_clothes`，抓取候选可以作为感知辅助，但没有证据说明直接迁移即可成功：袖口/衣摆需要语义定位；贴桌薄布的几何、摩擦和单层夹取不同于一般刚体；折叠还需要材料点跟踪、双臂配合与闭环恢复。AnyGrasp 展示过可变形物体抓取，也不能直接推导出它已验证 RoboDojo 的衣物单层夹取或完整折叠能力。

本次产物是可复跑前置检查与接入说明，不是已训练、已接通或已验证成功的抓取策略。
