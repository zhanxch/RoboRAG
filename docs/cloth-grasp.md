# 局部布料抓取实验与模型接口

本实验训练一个小策略：高层先把夹爪移到袖口或衣摆附近，策略再完成对齐、闭爪、抬升和保持。运行的是 RoboDojo ARX X5 双臂与原生 PhysX particle-cloth，未把粒子绑定到夹爪。结果衡量局部抓取，完整折叠评分不适用。

训练方法借鉴 [SuperMimic 论文 §3](https://arxiv.org/pdf/2610.28480) 的“采集仿真转移 → 多步动力学拟合 → 冻结世界模型并通过想象轨迹反传策略梯度”循环。本项目使用自己的小型 MLP、布料状态、奖励、BC anchor 和二值夹爪，属于改编实验。原论文的数据效率不能直接推导出一条叠衣示范足以学会通用抓取。

## 首轮实验结论（2026-10-23）

正式 48 次物理运行完成，24 个预先冻结的案例均有 BC 与 RL 结果。BC 成功 **9/24（37.5%）**，最终 BC+RL 成功 **2/24（8.3%）**，下降 29.2 个百分点。7 个案例只有 BC 成功，没有只有 RL 成功的案例。该配置的 RL 改进没有得到支持，当前模型不应被称为通用抓取模型。

训练交互衣物 09：BC 8/12，RL 2/12；仿真训练留出衣物 04：BC 1/12，RL 0/12。BC 在左袖口、右袖口、左衣摆、右衣摆分别为 1/6、1/6、3/6、4/6，RL 对应为 0/6、0/6、1/6、1/6。只有一个训练种子和两种衣物，不能将这些数字外推到所有布料或 RoboDojo 任务。

从原始逐步状态复算，BC 有 12 次抬升后没有保持、3 次未达到抬升阈值；RL 分别为 10 次和 12 次。动力学建模误差和闭环分布偏移是后续要验证的解释，现有证据不足以把退化归结为某一个原因。下一阶段将在 TRAIN 和新的开发集上验证增加示范、稳定策略更新等改进。

全部 48 个模型—案例组合及其实际加载衣物已核对，成功判据已从原始物理状态重新计算；训练与测试种子分离，冻结协议及模型哈希一致，RL TorchScript 与原模型的最大绝对输出差为 0。另完成 14 项接口、学习器、数据隔离和导出测试。记录的训练与测试局部控制共 5440 步，不含预定位与仿真复位。

模型包同时提供 `policy.ts`（最终 RL）与 `bc-policy.ts`（表现较好的 BC 对照），以及完整 checkpoint、逐案例 CSV、科学图和审计记录。二者原始 TorchScript 接受 `B × 39` 输入；`ClothPolicy` 包装器同时支持单条 `39` 维输入和批量输入。两者都需要特权几何观测，尚未验证真机可用性。

## 正式实验协议

预先冻结的协议在 `experiments/cloth-grasp-benchmark-v1.json`，实际运行目录为 `runs/cloth-grasp-bench-v1/`。协议 SHA-256：`c7642f738bc0a206e20007004acc4ba1cc689ca78fe9e5e2df235cf21eb83be6`。测试案例、判据和训练预算在本次模型训练开始前确定；最终模型固定取最后一次训练更新，不根据测试结果选权重。

原始 TRAIN HDF5 未提供衣物资产编号，因此“留出衣物”仅指未参与本轮仿真训练交互，不保证该款式从未出现在原始示范中。

一个父任务 `fold_clothes`，四个局部子任务：左袖口、右袖口、左衣摆、右衣摆。每个部位测试三种初始 TCP 偏移，单位米：`[0,0,0.065]`、`[0.02,-0.02,0.065]`、`[-0.02,0.02,0.09]`。每组分别使用训练见过的 `Top_Long/00009` 和未参与本轮仿真训练交互的 `Top_Long/00004`，共 24 个案例。

BC 初始化与最终 BC+RL 模型各测试 24 次，共 48 次。对应案例使用相同的布局、起始偏移和种子，并交替执行模型顺序。每次测试前重新复位真实布料。模型测试期间不更新，测试轨迹不写入训练 replay。预定位不可达记为失败，不从分母中移除；基础设施故障会中断实验。

成功要求目标材料点相对初始位置抬升至少 8 cm，目标到 TCP 距离小于 5.5 cm，夹爪命令开度小于 0.35，并在回合末尾连续满足至少 10 个控制步（0.4 s）。中途抬升后滑落算失败。闭爪命令不是接触力测量；这是本实验对稳定局部拾取的操作化判据。

## TRAIN 数据如何参与训练

只使用一个原始 TRAIN HDF5：`fold_clothes/train/episode_0000000.hdf5`，300 帧、25 Hz、三个 RGB 相机。SHA-256 为 `5a1fbba99a9f13900ec4439ba503674627335e28aeba6840223beb3e17cc93ac`。

从该记录提取四个抓取窗口：每个窗口从闭爪帧前 20 帧开始，到闭爪后第 5 帧结束，共 26 个机械臂动作帧，四个窗口共 104 个臂动作帧。使用原始 action 中的末端姿态和夹爪指令，将 TCP 轨迹平移，使示范闭爪位置对齐仿真目标点上方 6 mm。动作经每轴 8 mm 限制，并将夹爪开度阈值化。之后接 54 步局部等待、竖直抬升和保持；这段是显式任务扩展，不声称来自原始示范。

训练包含 20 个局部物理回合：8 个示范动作重定向回合、4 个几何教师回合、8 个当前策略探索回合。第 5、6 个完整回合只用于世界模型误差评估，始终不进入动力学训练或 BC/actor 采样。其余数据参与训练。BC 的 800 个实际观测－动作对中，156 个命令由 TRAIN 动作重定向得到，644 个命令来自抬升保持扩展或几何教师；四个窗口的 104 个独立臂动作帧均被使用。局部采样共 1,600 个控制步，其中 1,440 个训练转移、160 个动力学留出转移。环境复位、预定位、渲染以及此前开发试跑的成本另计。

原始 HDF5 没有布料粒子状态。重定向动作在仿真中执行后，布料观测与动作配对形成 BC 数据，不向原始 HDF5 伪造布料标签。这不是 RGB 端到端行为克隆，也不是单靠一条示范、不再采样仿真就训练成功。

BC 预热 1,600 次梯度更新。世界模型用 8 步自回归误差拟合；actor 用 16 步想象轨迹、critic 自举、BC 权重 20 进行 9 轮、每轮 5 次更新。二值夹爪使用前向硬阈值、反向 straight-through 梯度估计，因此它不等同于原 SuperMimic 的全连续解析梯度。想象回报不计入物理成功率。

## 模型及部署边界

Actor 为 MLP `(256, 256, 128)`，109,444 个参数；世界模型为残差 MLP `(256, 256, 256)`；critic 为 MLP `(128, 128)`。部署只需要 actor。导出 `policy.ts` 可由 PyTorch 在 CPU 或 CUDA 加载，不需要加载世界模型或 Isaac Sim。

输入是 39 维归一化浮点状态，**需要特权布料几何或另行训练的状态估计器**，不能输入 RGB 图像：

- `0:3`：目标点减 TCP，除以 0.1 m。
- `3`：目标相对初始目标的高度，除以 0.1 m。
- `4`：上一时刻夹爪命令开度，范围 0 到 1。
- `5:8`：TCP 减初始目标，除以 0.1 m。
- `8:11`：TCP 单步位移，除以 0.008 m。
- `11:14`：目标单步位移，除以 0.008 m。
- `14`：局部技能步数除以 80。
- `15:39`：8 个固定材料点相对目标的 xyz，除以 0.1 m 后展平。

目标是语义部位标注的材料点集合均值；附近 8 个点在复位后选定，整个回合固定材料身份，不逐帧重新找最近点。前端展示同一组记录点与 TCP 的 X–Z 轨迹。

输出为 4 维：前三维乘 0.008 m 得到每轴位置增量；最后一维为 `-1` 闭爪或 `+1` 开爪。姿态固定使用 `demo.json` 中对应部位的四元数，不由网络预测。TCP 约定为 `link6 + R(q) [0.145,0,0] m`。机器人适配器执行 IK 与以初始目标为原点的工作空间限制：下界 `[-.07,-.07,-.005]`，上界 `[.07,.07,.19]` m。控制频率 25 Hz，每次技能至多 80 步。

```python
from roborag.cloth_grasp.inference import ClothPolicy, encode_state

policy = ClothPolicy('runs/cloth-grasp-bench-v1/policy.ts', device='cpu')
state = encode_state(
    tcp=tcp_xyz, target=target_xyz, initial_target=initial_target_xyz,
    previous_tcp=previous_tcp_xyz, previous_target=previous_target_xyz,
    gripper_command=last_open_command, step=local_step,
    patch=eight_persistent_material_points,
)
command = policy.local_command(state)
# command['delta_xyz_m'], command['gripper_open']
# 仍需机器人适配器执行姿态、IK、工作空间边界及技能终止。
```

本轮不训练语义目标识别、最近可抓点选择、RGB 感知或完整叠衣策略。测试只有两件衣物、单个训练种子和结构化的 24 个案例，不能据此证明 grasp anything、跨相机、跨摩擦、任意褶皱或真机泛化。

## 运行与审计

依赖复用 `local.runtime.json` 中的仿真与策略 Python。训练需要 torch、numpy、scipy、Pillow、h5py 以及已有 RoboDojo 资产；前端只用标准库 HTTP 服务。正式训练在第 20 回合复位时遇到共享 GPU 显存不足，未记录该未完成回合。已完成的 19 回合及第 8 次更新权重保留，通过恢复权重、三个优化器和完整 replay、按固定采样日程重建随机数状态后补跑第 20 回合。恢复逻辑另有源码快照和采样器一致性单元测试；详情记录于 `experiment.json` 的 `recoveries`。

正式管线顺序启动一个仿真进程，按 4 个案例分批测试以释放渲染资源。

```bash
python3 scripts/cloth_grasp.py prepare --run runs/cloth-grasp-new \
  --demo /absolute/path/to/TRAIN/episode_0000000.hdf5
python3 scripts/run_cloth_benchmark.py --run runs/cloth-grasp-new --freeze-only
python3 scripts/run_cloth_benchmark.py --run runs/cloth-grasp-new
python3 scripts/cloth_grasp.py serve --run runs/cloth-grasp-new --port 8768
PYTHONPATH=src /path/to/policy/python scripts/summarize_cloth_benchmark.py \
  --run runs/cloth-grasp-new
```

浏览器打开 `http://127.0.0.1:8768`。可查看示范、三个相机视角、现场、逐回合回放、测试案例筛选、BC/RL 对照、实际抬升和世界模型留出误差。服务只监听回环地址，页面不能提交训练命令。

训练与测试保存原始 `transitions.npz`、相机 JPEG、`episodes.jsonl`、`metrics.jsonl`。`benchmark-results.json` / `benchmark.csv` 包含全部配对结果；汇总脚本从逐步状态重新计算成功判据，验证测试完整性、种子分离、冻结权重哈希与 TorchScript 推理一致性，写入 `audit.json`。`model-card.json` 包含完整输入定义、数据来源、配置与评估结果。

模型与结论打包在 `cloth-grasp-model-and-results.zip`，包含部署 actor、完整最终训练 checkpoint、BC 对照、测试清单、结果、审计信息和训练源码快照。原始大数据和模拟器资产保留在本机，不包含在模型包内。`runs/` 已被 Git 忽略，模型包需要随实验单独保管。

```bash
PYTHONPATH=src /path/to/policy/python -m unittest discover -s tests \
  -p 'test_cloth_grasp*.py' -v
```

单元测试检验梯度、回放隔离、协议完整性和输入单位，不替代真实物理测试。`cloth-grasp-v1` 是无效的早期接口调试；`cloth-grasp-v2` 是开发试跑，包含渲染内存故障后的不完整验证。它们不进入本轮正式训练数据或正式成绩。未启动的 v3 队列已取消。

## 72 小时扩大量实验（已被单任务持续训练替代）

用户已收回跨任务扩展目标。当前执行以 [叠衣持续 RL 说明](cloth-focus.md) 为准，旧批次队列和其他任务下载均已停止；以下保留原计划记录。

用户授权本机 RTX 4090 连续运行最多 72 小时。该批次从北京时间 2026-10-23 16:56:59 开始，到 2026-10-26 16:56:59 截止；重启或恢复不能刷新截止时间。配置、进度和进程归属在 `runs/grasp-scale-72h/`，`plan.json` 为时间预算依据。进程只清理自己创建的进程组，不中止其他实验。

首批公开数据固定在 `RoboDojo-Benchmark/RoboDojo` 的版本 `269457a3b42d8d0ae69898a601f2c1487c2f826e`，清单共 15 条、6.91 GB：10 条 `fold_clothes`，以及 `stack_blocks`、`classify_objects`、`fill_pen_holder`、`hang_mugs`、`pack_objects_into_box` 各一条。既有相同内容文件直接复用。每个文件核对 LFS SHA-256 和字节数，再提取动作。`general_pickup` 在该版本公开示范目录中缺失，已记录在计划修订中；不能把任务配置存在解释为有 TRAIN 数据。

这是将公开示范分配为本研究 TRAIN 的数据清单，不声称官方公开目录本身有严格的 TRAIN/TEST 资产分割。示范没有衣物资产编号，衣物留出仍只针对仿真交互。新增叠衣记录必须每个手臂恰有两次闭爪、25 Hz 控制且完整覆盖抓取窗口，才能沿用第一处袖口、第二处衣摆的任务顺序；该顺序不适用于其他任务。

第一阶段预先固定 88 个采集回合：10 条示范 × 4 个部位 × 2 个起始扰动，共 80 个 TRAIN 回合，另加 8 个完整世界模型留出回合。比较使用第 1 条和全部 10 条示范的嵌套数据子集，各训练三个种子 `31001/31002/31003`。BC 预热 4800 步、世界模型拟合 1200 步，RL 更新 50 步、学习率 `2e-5`、BC 权重 20。这里先评估增加数据后的一次离线世界模型策略改进，尚未加入下一轮在策略物理采样。

开发集使用衣物 09 与 07、4 个部位、3 个不同夹爪位置，共 24 个案例。六个训练配置均进行 BC/RL 配对，共 144 组、288 次物理运行。所有模型固定取最终更新，开发集仅用于决定改进方向和是否扩大；不与上一轮独立测试混称。衣物 01、06、11 预留给后续冻结的独立测试；原先测试过的 04 不用于新增训练或最终独立验证。

工程扩展门槛：10 条示范条件下，RL 三种子平均开发成功率至少 70%，比对应 BC 平均高至少 10 个百分点，并且每个种子都不低于 BC。这是扩展决策门槛，不是统计显著性判据。若失败，保留所有结果并先诊断；若通过，再冻结新测试、扩到 50 条示范和更多任务适配。刚体等其他任务首批只是数据准备，不算模型覆盖，也不算完整任务成功。

```bash
/path/to/policy/python scripts/prepare_grasp_scale.py --run runs/grasp-scale-72h
python3 scripts/run_grasp_scale.py --run runs/grasp-scale-72h
```

运行器等待首轮审计及数据完成后，自动按小批次采集、训练、验证；工作进程限时 30 分钟，基础设施错误保留进度并退出以便诊断，不重复覆盖已有模型和回合。开发协议生成后进行哈希冻结。模型拟合测试验证了数据量子集、世界模型留出隔离、拒绝 TEST 轨迹以及实际 TorchScript 导出。前端展示本轮结果与后续开发批次的状态，二者分开统计。


## SuperMimic paper implementation (2026-10-23 method correction)

The active experiment is `runs/supermimic-cloth-v1`, driven by
`scripts/run_supermimic_cloth.py`. The custom BC+RL focus/scale runners are
superseded and must not be restarted. The user accepts the original method's
analytic policy gradient and explicitly rejects the earlier BC+RL recipe.
The new learner is in `src/roborag/supermimic_cloth`; it does not import the old
learner or its weights. Shared simulator helpers are limited to reset, scene
registration, camera capture and owned-process management.

The implementation follows [SuperMimic](https://arxiv.org/pdf/2610.28480):
10 future reference frames, 10 observation-history frames, a 32D tanh latent,
separate encoder/decoder, paper MLP widths, 8-step autoregressive world fitting,
32-step differentiable imagination, soft survival and bootstrap, and a critic
with fixed lambda-return targets. There is no BC action loss, BC initialization,
PPO update, target critic, or extra lift reward. Analytic tracking rewards use
physical reference states. Resource differences and the cloth tracking adapter
are enumerated in the frozen effective plan, not presented as official code.

The current grasp criterion follows the user's correction: no 8cm requirement.
It checks physical elevation above the initial support position (5mm measurement
tolerance), co-motion with the TCP, closed physical fingers, and stable relative
position during the final second. This is a trajectory proxy; images and raw
trajectories must also be reviewed. `verification-protocol.json` adds two seen
starting positions, 16 development trials, and a 32-trial confirmation with
3 seconds of final retention. Reference feasibility is reported separately from
learned policy success. Neither score is a complete folding-task benchmark.

Reference extraction crops the grasp segment before later folding transport.
Recorded orientation and continuous grip are retained, with translation to the
semantic cloth point and time resampling; no height scaling is used. Only native
cloth simulation supplies reference particle states. Ten official TRAIN files
are hash-verified; the first stage uses one of them for reference construction.
The old pilot TEST and development trajectories remain excluded from replay.

Run `python3 scripts/run_supermimic_cloth.py` after checking its owned process
and shared simulation lock. A healthy process must never be duplicated.
`--references-only` stops at the physical reference gate. The fixed hard deadline
remains 2026-10-26 16:56:59 Asia/Shanghai. The runner atomically saves all
optimizers, replay and RNG after complete joint updates and resumes simulation
from complete episodes. `scripts/audit_supermimic_cloth.py --run <run> --package`
recomputes trajectory outcomes and verifies scene, reference, checkpoint and
frozen pilot hashes before producing a loadable model bundle. Failed reference
attempts, amendments and earlier source snapshots are retained.
