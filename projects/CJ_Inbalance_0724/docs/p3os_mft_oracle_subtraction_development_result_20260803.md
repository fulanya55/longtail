# P3o-S 开发结果：MFT oracle 去除的因果诊断

## 结论

P3o-S **未通过晋级门**，因此不得接回主路线或申请 locked test。

但诊断假设得到强而一致的支持：在完全冻结的六类分类器上，按生成器真值精确重建并去除 MFT 后，`MFT+ / ISRJ-` 子集的 ISRJ 假阳性率在全部 3 个模型种子 × 2 个独立数据种子中下降；等能量、非 MFT 的相位扰动对照没有复现这种下降。故此前 ISRJ 误报中存在可归因于 MFT 波形分量的混淆，而不只是“减去任意能量”造成的阈值效应。

同时，直接去除 MFT 会使模型失去 MFT 的可识别证据，导致宏 AUPR 稳定下降约 0.075。这是诊断性 oracle 操作而不是可部署的输入变换，不能据此宣称整体识别提升。

## 冻结设计与证据边界

- 条件：C0 原始复合 I/Q；C1 精确、按生成器噪声功率标定的 MFT 去除；C2 等能量确定性非 MFT 相位扰动去除。
- 数据：两个新 receiver-noise identity 集，种子 `1920000000`、`1930000000`；5 个干扰结构 × 3 个功率点，每视图 2,079 样本。
- 模型：冻结的 `mlcnn_iq_grouped_task_routed` 主分类器，模型种子 `2041/2042/2043`；冻结 safe thresholds；无训练、无阈值调优、无路由变化。
- 推断：每个模型×数据复制对 MFT+/ISRJ- 身份簇进行 2,000 次 paired bootstrap。
- 身份审计：两个新种子互不重叠；相对 2,942 个可发现历史训练 MAT 文件的最大重叠为 0。
- `test_generated=false`、`test_opened=false`、`locked_test_used=false`。

重分块缓存按设计将 `data_legacy` 转为 float32 且不复制 MATLAB `config` 元数据。首个无预测预检据此发现 P3o-S 读取噪声功率的缓存接口缺口；在任何开发预测前，实施仅元数据访问范围的修复：缓存没有该字段时，沿其不可变 `source_path` 回执读取同一原始 MAT 的 `config/radar/noise_power`。假设、数据、条件、标定、控制、检查点、阈值、bootstrap 与门规则均未改变，修订已记录在 method freeze。

## 结果

| 指标（C1 相对 C0） | 6 个复制范围 | 均值 | 预注册门 |
| --- | ---: | ---: | --- |
| MFT+/ISRJ- ISRJ FPR 降低 | 0.0867–0.1100 | 0.0972 | 通过（每复制 >= 0.02） |
| C1 相对 C2 的 FPR 降低 | 0.0909–0.1001 | 0.0942 | 通过（每复制 >= 0.01） |
| ISRJ 召回变化 | -0.0017–+0.0084 | +0.0045 | 通过（不低于 -0.02） |
| 宏 AUPR 变化 | -0.0770–-0.0721 | -0.0749 | **失败**（要求 >= -0.005） |

- 全部 C0-C1 与 C2-C1 paired-bootstrap 95% CI 下界均大于 0；C0-C1 下界范围为 0.0734–0.0944。
- MFT 的平均 AP 变化为 -0.4414，是宏 AUPR 失败的主因；SinFM 亦为 -0.0605。相反，ISRJ、VFT、SawFM 的平均 AP 分别为 +0.0204、+0.0241、+0.0087。
- 因此，C1 的误报下降不是普适性能改进，而是以移除 MFT 类证据为代价的、机制验证意义上的消歧结果。

## 产物与复现

- 方法冻结：[p3os_mft_oracle_subtraction_method_freeze_20260803.json](../configs/p3os_mft_oracle_subtraction_method_freeze_20260803.json)，SHA-256 `358c60662f32e1694828eeb6b2bfdd60976161bfbd865b0e1b5955a50ceb939a`。
- 汇总：[summary.json](../outputs/p3os_mft_oracle_subtraction_development_20260803/summary.json)，SHA-256 `62292a6ff546fbaa18740ed324f10a66f187982ffdb162b523561a35ff428f4f`。
- 逐样本预测：[predictions.npz](../outputs/p3os_mft_oracle_subtraction_development_20260803/predictions.npz)，SHA-256 `9706b80788b05436fadf26ba7b6339fc5b425fc14763115f22fb72840887fe80`。

## 下一步门

不要把 C1 作为推理前处理直接推广。若继续该机制线，下一实验应检验“保留原始 MFT 证据、仅向 ISRJ 判别引入 MFT 条件残差”的可部署表征/双支路，而不是再调整 C1 的去除强度。新方案须重新预注册，并在新的身份隔离开发集上同时满足：FPR 降低、MFT AP/宏 AUPR 不退化、预测标签门控端到端指标改善，之后才可考虑 locked test。
