# RoboDojo pilot 工作记录

## 2026-10-16

- 仓库：`zhanxch/RoboRAG`（private），初始提交 `a248a3a`。
- 初版纯 Python 双库原型 11 项单元测试通过。
- 用户已授权使用所有可用资源、必要时停止占用 GPU 的其他实验；重点转向在少量 RoboDojo 任务上实现并验证有效性。
- 已向经过进程身份核验的 InfiniDojo scheduler PID 329553 发送 SIGTERM，其 OpenWAM/仿真子进程退出；GPU 已释放。原始命令和停止原因保存在 ignored `runs/resource-handoff/`。未修改其他项目源码或删除其结果。
- 本机为 RTX 4090 24 GB、约 62 GiB 内存，根分区初查剩余 36 GB。复用现有 Isaac Sim 5.1 环境与资产，避免重复下载。
- 首个开发任务候选：`deposit_coin`。源训练轨迹已通过上游 SHA256 校验：`b9093beb1142f1e15ad38003a82abdc85f96417425193b0b5c7d9128ae90cd29`。
- 已构建上游 12 段 horizon=5 参考；后续 RoboRAG 将建立独立的密集片段长期库。
- `vendor/RoboICL` 是独立依赖 checkout，四个上游依赖通过只读复用已有同 commit、clean 的 checkout 连接；不修改共享依赖。
- 模型调用计划复用本机 Codex 账号桥接，不读取/导出凭据。它与论文 API 传输不同，必须明确记录；比较方法保持相同桥接和推理档位。
- 当前尚无本项目的真实控制成绩。

## 下一步

1. 自有启动器记录依赖、资产、训练来源及运行配置，完成相机 smoke test。
2. 从训练 HDF5 构建密集多视角片段，使用冻结视觉特征与本体状态检索。
3. 在上游控制接口上接入独立长期/短期检索、保留最新反馈。
4. 先跑开发 layout 0 的固定选段与双库检索，依据真实失败改进；随后使用未参与调优的布局验证。
