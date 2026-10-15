# Route A 预测可见度条件融合 development-v3 预注册

## 1. 单一假设

D2 visibility head 已经学到相对功率差，但普通辅助学习没有把该变量送入 presence 决策。
冻结 D2 后，用“预测可见度”而非真实功率显式控制有界残差，应提高弱正排序；若样本对齐
的可见度不优于打乱对照，则物理预测只是一项旁路任务，不能构成分类创新。

## 2. 冻结项

冻结 F2、D2 adapter/presence/visibility 两头、RawSaw、全部角色、Platt 校准、FPR<=0.20
阈值选择、残差上限 0.50 和五随机种子。训练和推理都不读取真实 gap；真实 gap 只用于
development 分层评估。test-v3 禁止访问。

## 3. 融合器与同容量对照

对每个 MFT/VFT/ISRJ 标签计算：presence 标准化证据 `p`，预测 gap 对应的严重度 `s`，以及
交互项 `p*s`。每标签使用 `residual=0.5*tanh(a*p+b*s+c*p*s+d)`，共 12 个标量。

- E1 aligned：`s` 与当前样本保持对齐。
- E0 shuffled：按预注册固定种子在每个数据角色内部独立置换 `s`，保持边际分布、参数量、
  优化器和选择规则完全相同，只破坏样本级物理对应关系。

## 4. 晋级门槛

E1 相对 E0 必须同时达到：弱召回 `>=+0.010`、弱正-vs-负 AUROC `>=+0.010`、macro AUPR
不劣于 `-0.003`、强召回不劣于 `-0.010`、FPR `<=0.20`；相对 C0 弱召回 `>=+0.030`、
AUROC `>=+0.015`，至少 4/5 种子弱召回为正。若失败，停止 final-temporal 后处理路线，
转向 development-v3 同 receiver identity 多功率数据生成与 stage2 配对 adapter。

