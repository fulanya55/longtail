# 长线目标当前证据状态（2026-08-04）

这是 P1n long-goal 的只读状态索引，不改变任何冻结模型、阈值、数据角色或结论，也不读取任何 consumed/locked test。它补充而不改写 `p1n_goal_requirement_evidence_audit_20260803.md`。

| 目标 | 当前可用证据 | 当前边界 | 下一允许动作 |
| --- | --- | --- | --- |
| MLCNN + IQ-Grouped 异构路由 | P4c/P4d 对旧 global head 的 full-12 改进是 reference-only 经验事实 | P7a 等 MAC/等参数 generic I/Q control 未证实 IQ grouping 因果；P6a-r1、P6b 亦停止 | 仅接受新的信息源与新的因果 control；不得调宽/调参已关闭路由 |
| SawFM/VFT 特异专家 | P2h 在受限弱分量条件下有 target-specific 中间证据 | SawFM OOD 与 VFT oracle sufficiency 均失败；P2w 禁止同源 VFT Doppler/phase 深化 | 需要独立物理观测契约后先做 oracle screen |
| ISRJ 低置信/功率校准 | P1n/P2a 是 Level-3 的 bootstrap + powerwise calibration 证据；P3o-S 诊断到 MFT 诱发 ISRJ-FP 的 oracle 因果 | P3n、P3o-T、P3o-U 是三种独立 deployable 单通道机制并均失败；P3o-U 的 U2-U3 约为零、recall 约降 3pp | 不得再建单通道 MFT-to-ISRJ router/residual/loss/calibration；须先取得新 acquisition observation |
| 分类到全 12 参数转化 | P1n in-range 与 P2h weak-component 有 predicted-gate 正向结果；P5e 五模型确认了 MFT/VFT/ISRJ JNR-only attachment | 不是通用 full-12 或 composition safety 证明；P5e 最差组合仍有局部风险 | 任何新候选必须同时报告 predicted-gate full-12、per-target/cell harm 与 matched control |
| 多轴泛化 | P2a 九功率、P2h 弱分量/63组合为分离的正向证据 | OOD、strict worst-combination、四轴 long-tail 没有单一安全候选；P8b/P8c 已停止 | 不可合并不同数据角色为“最终系统”；需新候选分轴预注册 |
| 惰性调用与共享前端成本 | P4j 有 compact-route CUDA-Graph 控制；P5e 有 dense attachment 的激活覆盖 | 覆盖率不是端到端加速；P5g--P5o 无有效计时结论 | P5p 正在全新三池、guarded、output-equivalent 成本开发；完成前不声明速度收益 |

## 新增停止：P3o-U

`P3o-U local phase dynamics` 在三组 base/residual seeds 上未通过预注册门：U2 相对 U0 的 source-FPR 降幅为 1.34%、1.70%、1.50%（门为 2%）；相对同容量 U1 的增益并非一致，相对 time-bin-shuffled U3 约为零；同时 ISRJ recall 每种子下降约 3 个百分点（允许 -2pp）。这属于“抑制响应而非提高可分性”，确认身份不生成。详见 `configs/p3ou_local_phase_dynamics_outcome_freeze_20260804.json`。

## 正在执行：P5p

P5p 不改变 P5e candidate、权重、特征、batch、模型、输出等价性检查或计时门；唯一新轴是崩溃可审计性。它已在数据生成前通过协议与无 CUDA receipt-recovery 自检，随后用三个全新 train-only paired-power 身份池完成零重叠审计。运行时先写入 `launch_receipt.json`、`environment_receipt.json` 与逐池 start/completion receipt，并禁止中断后恢复同一身份池。其结论只能是狭义的 output-equivalent per-batch 计算可行性，不是分类创新。

截至本索引编写时 P5p 正在第 1 个开发池运行；不生成 confirmation/test，也不对任何已消费 P5 池重跑。

## Locked-test gate

仍不满足。只有在一个单一候选分别冻结：matched-control 五模型与新身份确认、目标/组合/功率安全、OOD/组合不相交/长尾分轴证据，以及输出等价 CUDA-isolated 成本证据后，才可提交一次性新 locked-test 预注册供明确授权。既有 locked 或独立测试均不得读取或复用。
