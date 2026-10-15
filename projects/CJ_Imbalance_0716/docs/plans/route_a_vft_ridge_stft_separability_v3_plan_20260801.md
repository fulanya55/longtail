# Route A VFT Ridge-STFT development-v3 预注册

单一假设：VFT-STFT 的 validation→early 落差主要来自 validation 早停选择，而不是特征本身。固定恢复 16×16 STFT 特征，使用 `RidgeClassifier(alpha=10)` 在 fit 上一次拟合，按类别反频率样本权重平衡弱正和匹配负；validation 仅报告，不参与选择。

与 V0 相比，网络输入、弱正/匹配负定义和标准化不变；唯一变化是 AdamW+validation early-stop 替换为固定凸 Ridge。晋级门：early AUROC `>=0.62`、相对 final-temporal `>=+0.03`、至少 4/5 种子正增益、validation→early gap `<0.15`、相对 V0 early AUROC不低于 `-0.01`。不搜索 alpha。
