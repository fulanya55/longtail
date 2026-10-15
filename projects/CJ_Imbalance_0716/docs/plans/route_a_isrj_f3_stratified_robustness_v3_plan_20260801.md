# Route A：冻结 F3 的 ISRJ 分层稳健性与统计验证预注册

## 单一机制假设

若 F3 的收益确实来自可泛化的 ISRJ 长周期证据，则对齐周期融合头相对 C4 和 block-shuffle 的增益不应只由一个随机种子、单一 JNR、采样宽度或复合标签组合驱动。

## 冻结边界

- 只读取 `route_a_isrj_frozen_score_fusion_head_v3` 已保存的 F0、F3、F4 early predictions；
- 模型、融合头、校准器、阈值和代码推理输出均不改变；
- development-v3、种子 2039--2043；
- 弱分量阈值固定为相对最强共现分量低至少 10 dB；
- 不访问 test-v3。

## 预注册分层

1. ISRJ 相对功率差：`<=-15`、`(-15,-10]`、`(-10,-5]`、`(-5,0]`、`>0 dB`；
2. ISRJ nominal JNR：`[10,15)`、`[15,20)`、`[20,25)`、`[25,30.1] dB`；
3. ISRJ sample width：`[5,10)`、`[10,15)`、`[15,20.1] us`；
4. 复合阶数：1--6；
5. 与 FMB、MFT、VFT、SawFM、SinFM 分别共现/不共现；
6. exact label combination 是否在 fit role 出现；
7. receiver_noise_seed 按整数模 5 固定分成五个 cohort。

参数分层只选择该区间的 ISRJ 正样本，并使用当期 early role 全部 ISRJ 负样本作为共同负对照。其他分层在子组内部同时计算正负。

## 指标与最低支持度

- AUPR、全正召回、弱召回、强召回、负样本 FPR、弱正 vs 负样本 AUROC；
- 至少 20 个目标正样本、50 个负样本；弱 AUROC另要求至少 10 个弱正样本；
- 不满足支持度的格子保留计数但不纳入门槛；
- 五种子配对 bootstrap 95% CI；
- 预注册方向性 Wilcoxon signed-rank（alternative=`greater`）；
- 配对效应量 Cohen dz。

## 稳健性门槛

1. F3 相对 F0：所有受支持格子中至少 80% 的弱 AUROC为正，至少 80% 的弱召回不下降；
2. F3 相对 F4：至少 70% 的受支持格子弱 AUROC为正；
3. 两个弱功率层合并后，F3 相对 F0 的弱召回增益至少 +0.030；
4. 三个 sample-width 层中至少 2/3 的弱 AUROC相对 F0 为正；
5. 四个 JNR 层中至少 3/4 的弱 AUROC相对 F0 为正；
6. 五个 receiver cohort 中至少 4/5 的弱 AUROC相对 F0 为正；
7. 受支持复合阶数的最坏 F3 FPR不超过 0.25；
8. 非 ISRJ 五标签 logits、概率和预测与 F0 完全一致；
9. test-v3 未使用。

若门槛失败，不修改 F3；只记录失效区间并决定该候选是否还能进入基线/外部验证。

## 产物

保存预注册配置、每种子分层 JSON、五种子汇总、统计检验、最坏组收据、输入 predictions/data/role-indices 哈希和正式结果报告。

