# Route A RawSawCrossSeedFPR15 一次性 locked-test 结果（2026-07-26）

## 1. 最终结论

冻结候选 `RawSawCrossSeedFPR15+H2Others` 已完成唯一一次 locked-test 评估，10项预注册门槛全部通过，最终决策为：

`locked_test_passed`

本次运行没有训练、拟合、阈值搜索、校准器重拟合、seed选择或ensemble。测试完成后禁止依据本测试集继续修改模型、校准器和阈值。

## 2. 一次性审计

- locked-test 样本数：25,200；
- 六类顺序：FMB、SawFM、SinFM、MFT、VFT、ISRJ；
- 弱分量条目数：17,363；
- 测试文件 SHA256：`261650367d30ed182c37e3bd430136e2a8190aa407b523d6825114fe2366f105`；
- 冻结推理包 SHA256：`14d3667dd545f5eaa8d1d69881050fa05da529ddec7b2a48d7af724c65fb3cea`；
- 授权配置 SHA256：`10aa86841d1ceba79db7978ee44aee1ca44102cf8ce866a636632b3fb3a10710`；
- 正式 summary SHA256：`efe82851160ede64411d89e5080aae4c1f7fecf722dce3d0cc9ccf7c55f49f2d`；
- `fit_or_selection_on_locked_test=false`；
- `rerun_or_retuning_allowed=false`。

## 3. 主要结果

五个 seed 独立推理，不做概率平均。

| 指标 | H2 baseline | RawSawCrossSeedFPR15 | 变化 |
|---|---:|---:|---:|
| macro-AUPR | 0.8619 ± 0.0038 | **0.8934 ± 0.0046** | +0.0315 |
| micro-AUPR | 0.8658 | **0.8988** | +0.0329 |
| macro-F1 | 0.7555 ± 0.0045 | **0.7961 ± 0.0055** | +0.0406 |
| Hamming loss | 0.2315 | **0.1998** | -0.0316 |
| Exact match | 0.2241 ± 0.0101 | **0.2680 ± 0.0136** | +0.0439 |
| 全标签弱分量召回 | 0.3473 | **0.4066** | +0.0593 |

### SawFM

| 指标 | H2 baseline | 冻结候选 | 变化 |
|---|---:|---:|---:|
| AUPR | 0.7598 | **0.9487** | +0.1889 |
| Recall | 0.5151 | **0.8352** | +0.3201 |
| FPR | 0.1907 | **0.1352** | -0.0555 |
| 弱分量召回 | 0.1553 | **0.5122** | +0.3569 |
| 强分量召回 | 0.6197 | **0.9291** | +0.3094 |
| ECE-10 | 0.1362 | **0.0281** | -0.1081 |

逐 seed SawFM：

| seed | FPR | Recall | 弱分量召回 | macro-F1 | Exact match |
|---:|---:|---:|---:|---:|---:|
| 2039 | 0.1425 | 0.8622 | 0.5265 | 0.7981 | 0.2623 |
| 2040 | 0.1432 | 0.8452 | 0.5314 | 0.7993 | 0.2689 |
| 2041 | 0.0862 | 0.7895 | 0.4443 | 0.7914 | 0.2593 |
| 2042 | 0.1435 | 0.8230 | 0.5050 | 0.8025 | 0.2911 |
| 2043 | 0.1605 | 0.8560 | 0.5536 | 0.7894 | 0.2583 |

五个 seed 的 SawFM FPR 均低于0.20。

## 4. 对失败原因的最终验证

开发阶段 seed 2039 的高 FPR 不是主干表征或 Raw 专家排序能力不足，而是逐 seed 阈值贴近经验 FPR 上限，缺少跨 role、跨 seed 的安全裕量。locked-test 结果支持这一判断：

- AUPR保持0.9487，说明Raw专家排序稳定；
- LOSO阈值使五seed平均FPR为0.1352；
- locked-test recall 0.8352与开发阶段0.8448接近；
- locked-test macro-F1 0.7961与开发阶段0.8026接近；
- seed 2039 FPR为0.1425，没有复现原ordinary-early的0.2806。

因此本轮有效改进来自“跨 seed 稳健决策边界”，而不是增加网络宽度或重新训练主干。

## 5. 当前仍然存在的问题

locked-test通过不代表六类任务已经完全解决。冻结候选逐标签均值为：

| 标签 | AUPR | Recall | FPR | 弱召回 |
|---|---:|---:|---:|---:|
| FMB | 0.8810 | 0.8611 | 0.1884 | 0.5168 |
| SawFM | **0.9487** | **0.8352** | **0.1352** | **0.5122** |
| SinFM | 0.9299 | 0.8553 | 0.1830 | 0.4904 |
| MFT | 0.8984 | 0.7398 | 0.1444 | 0.2970 |
| VFT | 0.8543 | 0.6935 | 0.1930 | 0.3088 |
| ISRJ | 0.8479 | 0.6750 | 0.2092 | 0.3177 |

主要剩余瓶颈已经从SawFM转移到：

1. MFT、VFT、ISRJ弱分量召回仅约0.30；
2. ISRJ平均FPR为0.2092；
3. exact match仅0.2680，多标签错误累积仍然明显；
4. 最差组合主要包含SinFM及高基数组合，部分组合exact match接近0；这不意味着SinFM排序差，而是组合中任意一个标签漏检或误报都会令整条样本失败；
5. 五个Raw checkpoint仍来自两种训练抽样协议，虽然本次测试稳定，但方法一致性仍可进一步提高。

## 6. 后续工作计划

### 当前论文版本

1. 将本次 frozen candidate 和一次性 locked-test 结果作为路线A最终版本；
2. 不再使用本locked test选择阈值、结构、损失或模型；
3. 补充五seed置信区间、逐标签结果、组合结果、计算量和Base-DB/H2消融表；
4. 论文创新证据重点表述为：标签级专家路由、Raw物理证据保留和跨seed受限阈值共同改善SawFM弱分量识别。

### 下一代方法

下一代模型必须使用新的development与独立test数据，不能继续把本locked test作为验证集。建议顺序：

1. 为MFT/VFT/ISRJ分别建立独立可辨识性筛选，先证明冻结特征上存在可恢复局部证据；
2. 对通过筛选的标签增加标签条件池化、局部top-k旁路或独立物理专家；
3. 阈值继续采用LOSO跨seed选择，并同时约束弱召回与FPR；
4. 统一全部seed的frontend训练和角色抽样协议；
5. 使用全新receiver-noise seed、参数分布或新采集数据建立development-v2和test-v2；
6. 只有在development-v2预注册门槛通过后，才允许一次性使用test-v2。

## 7. 正式产物

- 配置冻结：`configs/shared/route_a_raw_cross_seed_locked_test_authorized_20260726.json`
- access marker：`outputs/tapc/route_a_raw_cross_seed_locked_test_once_20260726/locked_test_access_started.json`
- 正式总结：`outputs/tapc/route_a_raw_cross_seed_locked_test_once_20260726/route_a_raw_cross_seed_locked_test_summary.json`
- 最终receipt：`outputs/tapc/route_a_raw_cross_seed_locked_test_once_20260726/run_receipt.json`
- 各seed预测：`outputs/tapc/route_a_raw_cross_seed_locked_test_once_20260726/seed_*/predictions.npz`
- 各seed指标：`outputs/tapc/route_a_raw_cross_seed_locked_test_once_20260726/seed_*/metrics.json`

