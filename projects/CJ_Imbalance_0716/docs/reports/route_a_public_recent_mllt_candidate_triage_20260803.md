# Route A：公开近期多标签长尾候选的准入审计

日期：2026-08-03  
性质：只读外部源码/元数据审计；未读取 0716 I/Q、标签、预测或任何测试数据。

## 结论

本次新发现的两个公开仓库均**不准入**为“近期同行评审多标签长尾方法的同协议数值基线”。这不是对其代码或研究价值的否定，而是避免在缺少可追溯方法身份、精确接口或原始协议的条件下，将定制移植错误写成公平复现。

## 准入标准

一个候选必须同时具备：

1. 可由论文 DOI 或等价主出版物核验的作者/方法身份；
2. 固定公开修订、许可证和可审计训练实现；
3. 与当前 F2 全局特征/六标签 I/Q 协议可精确对接，或者有可预注册、同参数量控制的完整适配协议；
4. 不依赖测试集或本项目的可见性真值来构建图、采样、频次分组、门或阈值。

仅 README 自称“official”、仅有 `CITATION.cff`，或只显示一个可移植的损失函数，都不足以满足第 1--3 项。

## 结果

| 候选 | 可核验的公开观察 | 不准入原因 | 决定 |
|---|---|---|---|
| `ZhuJiaYucode/TEC-Graph` | README、视觉锚点、VOC 数据加载和 typed signed graph 源码可读；创建/更新于 2026-07-28 | 无 `CITATION.cff`、无 SPDX 许可证、无 DOI；其主线依赖冻结视觉锚点、图像数据/缓存与多阶段图训练，不能视为当前 F2 全局头的原样损失 | 不运行、不改名移植 |
| `JiajunChen223/RuralTail-MLR`（TALC） | MIT；`ASL + frequency-grouped SoftF1 + tail-recall` 代码可读，且有 CFF | CFF 无 DOI、可追溯作者论文或正式出版物；原协议是图像数据与 H/M/T 频次分组、图像载体训练和组阈值流程。仅移植其 SoftF1 会成为 TALC-inspired 自定义损失，而不是 TALC 的等协议复现 | 不运行、不改名移植 |

审计脚本将仓库默认分支、创建/更新时间、许可证、README/CFF blob SHA 与 SHA-256、路径标记和准入裁决写入机器收据。TALC 的 `data/processed/images_index.csv` 是 README 所描述的外部数据接口，不是仓库跟踪文件，因而其存在性检查为 `false`；这表示不能由该仓库的文件树证明当前 I/Q 接口兼容，并不声称原方法没有该数据输入。

## 与 GEDFD、PGL、MLC-NC 的关系

- GEDFD（TMM 2025）：题名、DOI、缩写和作者组合的四种 GitHub 查询均为零，仍无全文或可审计作者代码；不能近似实现。
- PGL（AAAI 2023）：官方固定 commit 有语法错误、未定义变量和不可决定的设备/类别依赖；不能修补后再称 PGL。
- MLC-NC（AAAI 2025）：有公开实现，但其 spatial-token、标签查询、专属投影和 ETF classifier 不是 F2 全局特征接口；移植属于新架构而非同主干基线。

因此，当前完成的 DR/RAL 与经典 BCE、pos-weight BCE、Focal、ASL、DB、CB、LA、两阶段等公平比较保持有效，但“**近期同行评审多标签长尾方法的精确等协议数值基线**”仍不可标记完成。

## 证据边界与后续动作

本轮不产生模型、训练、阈值、校准或性能比较，也不更改 H5o 的关闭结论。若后续获得上述候选的 DOI/正式论文、作者归属和适配许可，应先写独立预注册：固定上游 revision、清楚定义 F2 接口映射、增加同容量/同优化预算控制，并在新 identity development 源运行五种子；否则不得把新结果计为外部最新 MLLT 基线。

## 工件

- 机器收据：`outputs/tapc/route_a_public_mllt_candidate_triage/public_candidate_triage.json`
- 审计脚本：`experiments/audit_route_a_public_mllt_candidate_triage.py`
- GEDFD 多查询收据：`outputs/tapc/route_a_latest_mllt_reproducibility_audit/public_code_refresh_20260803.json`
- GEDFD、PGL、MLC-NC 的既有审计：
  `docs/reports/route_a_latest_mllt_reproducibility_audit_20260801.md`、
  `docs/reports/route_a_pgl_official_code_compatibility_audit_20260803.md`、
  `docs/reports/route_a_mllt_reproducible_baseline_candidate_audit_20260801.md`
