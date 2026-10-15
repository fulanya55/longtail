# Route-A 六类候选组合比较报告（2026-08-26）

## 1. 目的与边界

本轮按用户指定的六个候选集合，验证哪些六类组合适合作为后续 Route-A 开发对象。该分支使用此前特征提取能力已通过结构筛选的 `F2_DFMS_Full`，属于 development evidence；不读取、不调节 locked test，也不宣称已经解决九类特征充分性问题。九类整体门控仍按原九类审计结果单独保留。

## 2. 候选组合

| 编号 | 六类标签 |
|---|---|
| C0 | FMB、SawFM、SinFM、MFT、VFT、ISRJ |
| C1 | FMB、AMA、SawFM、SinFM、MFT、ISRJ（VFT→AMA） |
| C2 | FMB、SawFM、SinFM、SN、MFT、ISRJ（VFT→SN） |
| C3 | FMB、SawFM、SinFM、MFT、ISRJ、RangeFakeTarget（VFT→RangeFakeTarget） |
| C4 | FMB、AMA、SawFM、SinFM、MFT、VFT（ISRJ→AMA） |
| C5 | FMB、SawFM、SinFM、SN、MFT、VFT（ISRJ→SN） |

标签顺序由九类全局索引投影得到，组合编码、逐标签指标和组合级指标均在六类标签空间内重新计算。

## 3. 统一模型与训练协议

- 主干：`F2_DFMS_Full` / `dfms_iq`，width=16，真实复数 I/Q 输入 `[B,2,3000]`。
- 结构：复数 I/Q 延迟融合、四级多尺度复数编码、Cartesian/幅度/多滞后相位（lag=1/4/16）晚期融合、全局 mean/std 统计池化和六维线性分类头；当前实现约 257,511 个参数。
- 初始化：每个 seed 使用相应的 Route-A F2 checkpoint `20260731_route_a_visibility_balanced_fusion_v2/seed_x/backbone/best_model.pth`，仅替换六类分类头。
- 损失：`BCEWithLogitsLoss`；优化器：AdamW；`lr=1e-4`，`weight_decay=1e-4`；梯度裁剪 5.0。
- 训练：30 dB train MAT 中固定 split_seed=260826 抽取 1,024 个拟合样本，2 epochs，batch=16。
- 开发评估：10、15、20、25、30、35、40、45、50 dB 普通 test MAT，每个功率固定抽取 256 个样本；batch=32；阈值固定 0.5。
- seeds：首轮 2039/2040/2041；晋级确认 2039–2043。训练/开发 receiver_noise_seed 不重叠（每个功率 overlap=0）。
- 成本：三种候选结构完全共享 F2 主干，均为 257,511 参数、4,449,008,256 个单样本卷积/线性 MAC；单次 batch-1 CUDA 延迟测量受 GPU 状态影响（约 4.1–7.3 ms），因此不据此宣称速度差异。

## 4. 三种子筛选结果

指标为五种子均值；`std30` 是 30 dB Macro-AUPR 的样本标准差，`worst` 是每个 seed 在九个功率点的最弱功率 Macro-AUPR 再取均值，`comb30` 是 30 dB 六类组合完全匹配率。

| 候选 | 30 dB Macro-AUPR | std30 | 最弱功率 Macro-AUPR | 30 dB 组合准确率 | 10 dB / 50 dB Macro-AUPR |
|---|---:|---:|---:|---:|---:|
| C1 VFT→AMA | **0.9200** | 0.0072 | 0.7103 | **0.4508** | 0.8381 / 0.7103 |
| C0 原六类 | 0.9089 | **0.0049** | **0.7332** | 0.4211 | 0.7480 / **0.7370** |
| C4 ISRJ→AMA | 0.9085 | 0.0092 | 0.6866 | 0.4102 | 0.8223 / 0.6866 |

C2、C3、C5 在三种子首轮筛选中已经明显落后于上述三者（30 dB Macro-AUPR 分别约为 0.8652、0.8613、0.8515），因此本轮五种子确认不再重复消耗计算资源，但结果保留为负向筛选证据。

### 逐标签观察（五种子、30 dB）

- C1：AMA 0.9987、FMB 0.9948、SinFM 0.9907、SawFM 0.9729；MFT 0.7307、ISRJ 0.8323，是主要瓶颈。
- C0：FMB 0.9939、SinFM 0.9988、SawFM 0.9252；MFT 0.8598、VFT 0.8300、ISRJ 0.8457，整体较均衡。
- C4：AMA 0.9966、FMB 0.9919、SinFM 0.9919、SawFM 0.9778；MFT 0.7962，VFT 0.6966，VFT 成为最明显弱项。

## 5. 结论与下一步门控

1. **C1（VFT→AMA）是当前开发指标的首选候选**：30 dB Macro-AUPR 和组合完全匹配率最高，且五种子波动可接受。
2. **C0 是可靠性基线**：最弱功率和 50 dB 表现最好，波动最小，适合作为任何改进的连续性对照。
3. **C4 可作为第二替换对照**：高功率/30 dB不占优，且 VFT 的逐标签性能明显拖累其最弱功率表现。
4. C2/C3/C5 暂不晋级，不删除其结果；后续若研究 SN 或 RangeFakeTarget，应单独设计针对性数据/损失，而不能把本轮低分解释为类别本身不可识别。
5. C0、C1、C4 的五种子确认已经完成。以同一 seed 配对的 30 dB Macro-AUPR 比较，C1 相对 C0 的 paired-t p=0.0133、相对 C4 的 p=0.0309；由于只有五个 seed，Wilcoxon 双侧 p 均为 0.0625，不能写成强统计显著结论。组合完全匹配率方面，C1 相对 C4 的 paired-t p=0.0134，但相对 C0 的 p=0.1738。因此 C1 只能作为当前开发优先候选，仍需逐标签 recall/FPR、九功率曲线和成本审计后再冻结；locked test 仍需另行预注册并获得授权。

五种子均值的 seed-bootstrap 95% CI（仅描述稳定性，不替代独立样本显著性检验）为：C0 30 dB Macro-AUPR 0.9089 [0.9051, 0.9127]、最弱功率 0.7332 [0.7273, 0.7380]；C1 分别为 0.9200 [0.9139, 0.9258]、0.7103 [0.6947, 0.7260]；C4 分别为 0.9085 [0.9015, 0.9158]、0.6866 [0.6741, 0.6991]。

## 6. 可复核工件

- 三种子筛选汇总：`outputs/route_a_sixclass_combo_screen_20260826/sixclass_combo_summary.json`
- 五种子确认汇总：`outputs/route_a_sixclass_combo_confirm_20260826/sixclass_combo_summary.json`
- 五种子角色清单：`outputs/route_a_sixclass_combo_confirm_20260826/role_manifest.json`
- 五种子终端回执：`outputs/route_a_sixclass_combo_confirm_20260826/terminal_receipt.json`
- 执行脚本：`experiments/run_route_a_sixclass_combo_screen.py`
