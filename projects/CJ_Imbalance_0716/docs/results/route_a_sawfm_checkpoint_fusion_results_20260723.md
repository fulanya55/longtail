# 路线 A：SawFM 早期—后期 Base-DB 检查点融合验证结果（2026-07-23）

## 1. 本轮结论

静态检查点融合验证成功。Logit 融合与参数权重插值均通过预注册 AUPR 晋级门槛，说明 SawFM 的早期能力与六类总体后期能力不是完全不可兼容的，可以通过沿同一 Base-DB 训练轨迹进行融合而部分恢复。

推荐将 **WeightFusion-0.2** 作为下一阶段主要候选：它保持单模型和原始部署参数量，在独立 fusion-validation role 上基本不损失 macro-AUPR，同时稳定提高 SawFM AUPR。**LogitFusion-0.6** 保留为互补性上界和双模型消融，不作为首选部署方案。

## 2. 协议

| 项目 | 设置 |
|---|---|
| 数据 | `C:\LJQ_code\干扰数据\复合干扰数据0716` |
| 环境 | `D:\anaconda3\envs\PY10\python.exe` |
| seeds | 2041、2042、2043 |
| 后期端点 | DB-Macro，第 20、19、20 轮 |
| 早期端点 | DB-Tail，第 3、2、3 轮 |
| alpha 网格 | 0.0、0.1、...、1.0 |
| alpha 选择 | 三个 calibration role 选择一个共享 alpha |
| 独立验证 | threshold-selection role 改作 fusion validation |
| 阈值 | 仅报告温度校准后的固定 0.5，不在验证角色选阈值 |
| locked test | 未使用 |

每个 seed 的 calibration 和 fusion-validation 均为 300 样本，两个角色的 sample ID 交集为 0。

## 3. Alpha 选择结果

### 3.1 Logit 融合

选定共享 `alpha=0.6`。在 calibration role 上相对 DB-Macro：

- macro-AUPR：-0.002031；
- SawFM AUPR：+0.029219。

### 3.2 参数权重插值

选定共享 `alpha=0.2`。在 calibration role 上相对 DB-Macro：

- macro-AUPR：-0.000975；
- SawFM AUPR：+0.008804。

参数插值只需向早期检查点移动 20%，表明后期模型附近存在可恢复部分尾类能力的低损失方向。

## 4. 独立 Fusion Validation 结果

### 4.1 总体结果

| 方法 | alpha | Δ macro-AUPR | Δ SawFM AUPR | Δ macro-F1 | Δ Hamming | Δ SawFM F1 | 通过 |
|---|---:|---:|---:|---:|---:|---:|---|
| LogitFusion | 0.6 | -0.000586 | **+0.038916** | -0.028988 | +0.002407 | +0.061594 | 是 |
| **WeightFusion** | **0.2** | **-0.000202** | **+0.011442** | **-0.001795** | +0.002963 | +0.017574 | **是** |

置信区间：

- LogitFusion：Δ macro-AUPR 95% CI `[-0.006116, 0.004125]`；Δ SawFM AUPR `[0.027298, 0.058660]`；
- WeightFusion：Δ macro-AUPR `[-0.002283, 0.003056]`；Δ SawFM AUPR `[0.000768, 0.025423]`。

### 4.2 LogitFusion-0.6 分种子

| Seed | Δ macro-AUPR | Δ SawFM AUPR |
|---:|---:|---:|
| 2041 | +0.000231 | +0.058660 |
| 2042 | +0.004125 | +0.030790 |
| 2043 | -0.006116 | +0.027298 |

SawFM 在 3/3 seeds 上提高，但 seed 2043 的 macro-AUPR 略超出 -0.005 非劣界；总体仍满足预注册的 2/3 seeds 非劣要求。其 macro-F1 平均下降约 0.029，且部署需要同时运行两个模型，因此更适合作为“两个端点输出确实互补”的机制上界。

### 4.3 WeightFusion-0.2 分种子

| Seed | Δ macro-AUPR | Δ SawFM AUPR |
|---:|---:|---:|
| 2041 | -0.001380 | +0.025423 |
| 2042 | +0.003056 | +0.000768 |
| 2043 | -0.002283 | +0.008134 |

WeightFusion 在 3/3 seeds 上都满足 macro-AUPR 非劣并提高 SawFM AUPR。macro-F1 基本不变，仍为单一 47,718 参数模型，不增加推理次数。

## 5. 机制解释

### 5.1 早期和后期能力可以部分兼容

如果两个检查点位于完全不兼容的表征盆地，参数插值通常会明显破坏总体性能。本轮 `alpha=0.2` 能保持 macro-AUPR 并恢复 SawFM，说明同一 DB 轨迹的早期—后期模型之间至少存在局部可连接区域。

### 5.2 Logit 上界高于参数融合

LogitFusion 的 SawFM 增益约为 WeightFusion 的三倍，说明两个端点的决策函数具有更强的输出互补性；但内部参数空间不能无损表达全部互补信息。后续若需要追求更高上限，可设计轻量双分支或早期教师残差，而不是直接部署两个完整模型。

### 5.3 静态融合修复的是排序能力，不完全修复固定阈值决策

LogitFusion 虽显著提高 SawFM AUPR，却使 macro-F1 下降约 0.029。WeightFusion 的固定阈值指标更稳定，因此当前论文主候选应优先选择 WeightFusion，LogitFusion 作为 AUPR 上界和机制消融。

## 6. 当前决策

1. Base-DB-Macro 继续作为主要基线；
2. WeightFusion-0.2 晋级为下一阶段主要候选，alpha 固定为 0.2，不再重新调参；
3. LogitFusion-0.6 作为双模型上界保留，alpha 固定为 0.6；
4. Mix75/Mix90 与 ASL 分类头路线继续停止；
5. locked test 仍不启用。

## 7. 下一阶段验证建议

下一阶段应做固定超参数的确认性验证：

1. 新增 seeds 2039、2040，分别训练 DB-Macro 与 DB-Tail 端点；
2. 不再访问 calibration 网格，直接使用 WeightFusion-0.2 和 LogitFusion-0.6；
3. 聚合为 5-seed SawFM 结果，重点确认 WeightFusion 的 SawFM AUPR 是否仍为正；
4. 若 5-seed 通过，再在 VFT、ISRJ 上分别重新产生早期/后期端点，但融合 alpha 保持固定，不按类别重调；
5. 最终比较参数量、单样本推理时间和 GPU 显存，形成论文效率证据链。

## 8. 审计与复现

- 三个 seeds 的早期/后期端点 split SHA256、manifest SHA256 一致；
- calibration 与 fusion-validation 每 seed 各 300 样本，交集为 0；
- 参数插值 alpha=0 和 alpha=1 均以 `1e-6` 容差精确复现对应端点 logits；
- 未执行 threshold selection，验证角色没有用于选择 alpha；
- 每个 seed 均保存 WeightFusion 单模型检查点、LogitFusion 部署清单和两种预测文件；
- locked test 未使用；
- `py_compile` 通过；
- 全量测试：123 passed，36 个现有 AMP 弃用警告。

正式摘要 SHA256：`B4047081822364592B61988CE8AE48D11EB39CD74DB07BFA4A8A88FFDAE4D60E`

## 9. 产物

- 技术方案：`docs/implementation/route_a_sawfm_checkpoint_fusion_plan_20260723.md`
- 执行脚本：`experiments/run_route_a_sawfm_checkpoint_fusion.py`
- 正式摘要：`outputs/tapc/route_a_sawfm_checkpoint_fusion/route_a_sawfm_checkpoint_fusion_summary.json`
- 融合模型与预测：`outputs/tapc/route_a_sawfm_checkpoint_fusion/runs/seed_<seed>/`
