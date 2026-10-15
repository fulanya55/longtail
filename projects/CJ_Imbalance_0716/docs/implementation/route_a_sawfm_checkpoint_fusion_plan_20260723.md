# 路线 A：SawFM 早期—后期 Base-DB 检查点融合验证方案（2026-07-23）

## 1. 已确认的问题

2×2 因子试验已经证明：Mix90 没有独立收益，SawFM 的约 +0.069 AUPR 增益来自选择第 1～3 轮的早期检查点；macro-AUPR 最优点则稳定位于第 19～20 轮。

因此，下一步不再修改 ASL/DB 混合权重，而验证两个能力是否可以融合：

- 早期 DB-Tail：保留 SawFM 排序能力；
- 后期 DB-Macro：保留六类总体判别能力。

## 2. 验证问题

1. 两个检查点的输出是否具有可融合互补性？
2. 两个检查点是否位于可通过参数线性连接的低损失区域？
3. 融合能否在 macro-AUPR 基本不下降的前提下恢复至少一部分 SawFM AUPR？

## 3. 两条验证路径

### 3.1 Logit 融合

对同 seed 两检查点的 logits 做：

`z(alpha) = (1-alpha) * z_macro + alpha * z_tail`

其中 alpha 网格为 0.0、0.1、...、1.0。

该方法不改变模型内部参数，用于验证两个决策函数是否互补。

### 3.2 参数权重插值

对两个检查点的浮点 state tensor 做：

`theta(alpha) = (1-alpha) * theta_macro + alpha * theta_tail`

非浮点缓冲区采用最近端点。该方法用于验证早期和后期模型是否位于可连接区域；若 logit 融合有效但权重插值失败，说明两个内部表征盆地不兼容。

## 4. 数据角色与防泄漏

| 角色 | 用途 |
|---|---|
| early-stop | 已用于选择 DB-Macro/DB-Tail 两个端点，不再作为本轮主验证 |
| calibration | 选择单一共享 alpha，并拟合温度 |
| threshold-selection | 本轮改作独立 fusion validation，只报告 AUPR 和温度校准后的固定 0.5 指标，不在该角色上重新选择阈值 |
| locked test | 禁止读取 |

三个 seeds 使用同一个共享 alpha，不允许逐 seed 选择最优权重。

## 5. Alpha 选择规则

分别对 logit 融合和权重插值：

1. 在三个 calibration role 上计算相对 alpha=0（DB-Macro）的配对均值；
2. 候选必须满足平均 macro-AUPR 增量不低于 -0.005，且至少 2/3 seeds 满足该非劣界；
3. 在合格候选中选择平均 SawFM AUPR 增量最大的 alpha；
4. 若并列，依次选择 macro-AUPR 更高、alpha 更小者；
5. 选定后在 threshold-selection role 上一次性验证。

## 6. 晋级门槛

相对 DB-Macro，独立 fusion validation 同时满足：

1. 平均 macro-AUPR ≥ -0.005；
2. 平均 SawFM AUPR ≥ +0.005；
3. 至少 2/3 seeds 的 macro-AUPR ≥ -0.005；
4. 至少 2/3 seeds 的 SawFM AUPR > 0；
5. locked test 未使用。

若 logit 融合通过而参数插值失败，下一阶段优先设计双分支/残差融合；若两者都通过，优先选择部署成本更低的参数融合；若均失败，则早期尾类能力与后期总体能力不可通过静态线性融合兼容，应转向训练期抗遗忘约束。

## 7. 固定设置

- 环境：`D:\anaconda3\envs\PY10\python.exe`；
- 数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`；
- 配置：`configs/student_a/six_class.yaml`；
- seeds：2041、2042、2043；
- 每个 development role 最多 300 样本；
- 端点必须具有相同 split SHA256、manifest SHA256 和模型结构；
- locked test 保持冻结。
