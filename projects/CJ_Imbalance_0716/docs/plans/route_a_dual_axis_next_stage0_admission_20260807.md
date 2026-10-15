# 路线 A 双轴长尾下一阶段：Stage 0 观测准入审计

## 1. 本阶段目的

本阶段只回答一个问题：当前 E 盘本地数据、生成器和可参考程序中，是否已经存在一种在部署时可获得、非标签泄漏、非 oracle、且与当前单路复数 I/Q 具有独立物理信息的第二观测。

Stage 0 是只读元审计，不生成新数据、不训练模型、不拟合校准器、不选择阈值，也不读取 sealed test-v3 的样本、标签、预测或指标。test-v3 只允许读取 `TEST_V3_DO_NOT_OPEN.json` 和 `test_v3_sealed_manifest.json` 两个封存元数据文件。

## 2. 准入定义

候选第二观测必须同时满足：

1. 推理部署时可获得，而非仅训练期 teacher；
2. 不是由当前 I/Q、标签、功率真值或完整样本反演出的 oracle 特征；
3. 具有独立的接收、重复测量、探测斜率或外部标定合同；
4. 能以身份级键与当前样本配对，并可执行 train/development/test 身份隔离；
5. 新增观测的生成或采集方法可复现、可冻结、可哈希。

## 3. 审计对象

- 当前路线 A 的 P1/P2、MFT、VFT 主要数据生成器；
- `CJ_Inbalance_0724` 中的 `iq_reference`、功率配对数据集和训练逻辑；
- 0724 独立观测合同审计提出的 M1 双接收机、M2 独立重复 PRI、M3 双斜率探测、M4 外部时延参考；
- B0、RawSaw、Base-DB、IP-CVE P0 及 post-P0 审计的权威收据；
- test-v3 封存元数据。

## 4. 判定

Stage 0 的预期机器判定字段为：

```text
decision = no_admissible_non_oracle_observation
selected_branch = R_sawfm_scoped_dual_axis
```

主要依据如下：

- 当前六个直接生成器均只输出单路复数 I/Q，没有第二接收通道或外部参考；
- 0724 的 `iq_reference` 是同一身份的固定 30 dB 高可见度功率视图，仅作为训练期排序 teacher；单视图部署时不存在，因此不能作为第二物理观测；
- M1–M4 目前只是可建立的新合同，项目内没有已经可执行且可部署的正式数据源；
- IP-CVE P0 已否定现有单路局部候选，不能把相同信息的再编码包装成独立观测。

因此自动转入路线 R：限定 SawFM，使用全新身份构造“标签频次 × 分量可见度”二维实验，而不是继续搜索当前单路 I/Q 的局部变换。

## 5. 一致性修正

B0 权威 `b0_summary.json` 和正式结果报告记录的计算量均为 `211,857,536 conv/linear MACs/sample`。2026-08-07 两份活动汇总中的 `11,857,536` 属于转录错误；只修正活动汇总和 Prompt，历史正式结果、JSON 与收据保持不变。

## 6. 产物与终止条件

执行：

```powershell
D:\anaconda3\envs\PY10\python.exe experiments\audit_route_a_dual_axis_next_stage0.py
```

必须生成：

- `outputs/route_a_dual_axis_next/stage0_admission/admission_audit.json`
- `outputs/route_a_dual_axis_next/stage0_admission/run_receipt.json`

只有审计、测试和哈希收据完整，才允许写路线 R1 预注册。Stage 0 不授权生成正式数据或启动训练。
