# Route A ISRJ 参数范围外外部验证预注册

## 科学问题

development-v3 上的 `H1 + 对齐周期专家` 增益是否能泛化到训练范围之外的 ISRJ 长周期结构，而不是记忆 `[5,20] us`、`1--5` 次副本的生成参数分布？

## 外部数据

- 新生成种子：`64000000`；
- train-only外部评价集，不生成 test split；
- 63种六类非空组合，每组合80条，共5,040条；
- 其余雷达体制和五类干扰参数保持0716合同；
- ISRJ `sample_width_s=[22e-6,30e-6]`，训练范围为 `[5e-6,20e-6]`；
- ISRJ `num_replicas=[6,8]`，训练范围为 `[1,5]`；
- `(replicas+1)*width <=270 us < 300 us PRI`，满足生成器物理边界；
- receiver_noise_seed必须与 development-v2/v3/v4/v5/v6及所有保留测试合同完全不相交；
- 保存生成receipt、文件SHA256、身份审计和参数范围审计。

该数据是同一受控生成器下的“参数范围外”证据，不宣称新噪声类型或真实数据。

## 冻结评价

每个 seed完全冻结 development-v3 的：

- Focal-36表征；
- H1 BCE线性头；
- RawSaw；
- 对齐周期HGB；
- B2低容量融合头及其归一化；
- Platt校准器；
- 六标签阈值。

外部数据不重新训练、不重新校准、不重新选阈值。比较 B0 H1 与 B2 H1+周期；二者除 ISRJ logit外完全一致。

## 指标与门槛

- 主要：ISRJ AUPR、弱召回、强召回、弱-vs-negative AUROC、FPR；
- 次要：六标签macro-AUPR/F1、SawFM AUPR/弱召回；
- 分层：sample width 22--26/26--30 us、replicas 6/7/8、JNR、组合阶数、共现标签、receiver cohort；
- B2相对B0：ISRJ AUPR和弱AUROC均为正，弱召回 `>=+0.030`，至少4/5种子主要方向为正；
- B2绝对ISRJ FPR `<=0.25`；
- macro-AUPR不得下降，SawFM逐元素必须完全一致；
- 若失败，不修改已冻结B2；如实判定周期专家只在训练参数范围内成立。

## 禁止事项

- 禁止访问 test-v3及任何保留 test-v4/v5/v6；
- 禁止用外部结果调整B2的C、弱正权重、校准器、阈值或专家特征；
- 禁止将5,040个样本伪装成五个独立随机种子；统计单位仍为五个冻结模型seed和receiver identity。
