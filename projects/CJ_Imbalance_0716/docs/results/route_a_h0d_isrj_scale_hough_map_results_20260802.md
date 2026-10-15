# Route A H0d：无 oracle ISRJ scale-Hough 复制图

## 问题与边界

H0b 已证明：若以真实 `sample_width` 将延时轴归一化，ISRJ 重复次数可被稳定回归；H0c 又证明冻结宽度预测的点估计不足以替代该坐标。H0d 检验一个新的、推理可用的替代方案：不输入真实或预测宽度，而是在 I/Q 上固定计算全部物理可行宽度假设的局部复相关库。

该试验只使用新的 train-only 数据，未生成、打开、分析或使用 test-v3、test-v2 或历史 locked test。它是物理可识别性诊断，不是六类识别或融合结果。

## 固定方案

- 数据：`C:\LJQ_code\干扰数据\复合干扰数据0802_isrj_scale_hough_h0d\closedset_6class_train.mat`；SHA-256 `65fc53ca8fd69486bfd7947018c065c48e19aa8b41caaebb5a8dd6270538597c`。
- 5,040 个唯一 `receiver_noise_seed`（928677743--928682782）；ISRJ 正例 2,560；fit / early-stop / evaluation 为 1,536 / 512 / 512，所有历史训练源身份交集为零。
- 固定表示：对宽度假设 100, 120, ..., 300 samples、复制倍数 1--8、12 个重叠块计算归一化复相关，得到无可训练参数的 `2 x 88 x 12` map。
- 回归头：`ScaleHoughReplicaProbe`，2,497 参数；AdamW（lr `3e-4`，wd `1e-4`），SmoothL1，30 epochs；五个固定 seed `2039--2043`。
- 对照：同一 map、容量、初始化、角色与优化器，仅置乱 fit/early-stop 的 `num_replicas` 目标；evaluation 目标不置乱。

## 结果

| 指标 | H_real（五 seed 均值） | H_real - H_perm |
|---|---:|---:|
| R2 | 0.0288 | +0.0422 |
| Spearman | 0.2326 | +0.2845 |
| 配对 R2 95% CI | — | [+0.0286, +0.0552] |
| 正向 R2 差异 | — | 5 / 5 |

五个 H_real R2 分别为 `0.0045, 0.0532, 0.0124, 0.0481, 0.0260`；H_perm 分别为 `-0.0423, -0.0114, -0.0052, 0.0021, -0.0099`。因此真实表示携带统计上稳定、但量级很小的目标信息。

## 预注册判定

H0d 要求：真实 R2 >= `0.20`、Spearman >= `0.40`、真实减置乱 R2 >= `+0.20`、配对 CI 下界为正、至少 4/5 正向、容量相同且 test-v3 封存。容量、CI、5/5 方向和封存条件通过；前三个效果量门槛失败。

**结论：停止 `scale-Hough` 路线，不能进入 H1 ISRJ 分类器或与主干融合。** 禁止在这一来源或同一机制上继续搜索宽度网格、间隔、乘数范围、池化策略或 CNN 容量。

## 机制解释与后续约束

H0b 的 oracle R2 约 `0.717` 与 H0c/H0d 的失败共同构成更清楚的边界：复制结构不是不存在，而是当前复相干统计对延时坐标误差高度敏感；“点预测宽度”和“固定离散宽度库”都无法在未知宽度下让该信息可部署地恢复。后续若研究 ISRJ，只能采用不同的、在生成物理上可解释的无相位对齐证据，而不是 scale-Hough 的变体；首先必须用新的身份隔离来源完成同等的 H0 可识别性门。

机器可读产物：`outputs/tapc/route_a_h0d_isrj_scale_hough_map_formal/route_a_h0d_isrj_scale_hough_map_summary.json`、身份审计 `outputs/tapc/route_a_h0d_isrj_scale_hough_generation/identity_audit.json`、特征缓存 SHA `f56ec3c6f5fc601bb49e2e9646be41b0d4ce9bb220d0d3047fcd9dec0c362cd6`。
