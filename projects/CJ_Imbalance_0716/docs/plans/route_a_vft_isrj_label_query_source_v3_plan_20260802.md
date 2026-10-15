# VFT 标签条件 token 查询：source-v3 预注册

唯一假设：F2 final-temporal token 中的顺序信息只能由 VFT 专属查询选择，固定 12-bin 池化因此不稳定。冻结 F2；仅训练 VFT 二元 query-head：64 维可学习 query 与每时刻 token 点积、softmax 时间权重、加权 token 加 mean 安全旁路，接线性分类器。对照为同参数无条件 query pooling 与每记录 token 时间置乱 query。无主干训练、无阈值、无校准、无候选融合。

数据为新鲜 source-v3，seed `182000000`，63×80 train-only；五个身份隔离 fit/validation/early 角色。主要 strict view 为 weak VFT 正 vs VFT-absent/ISRJ-present 且其他四标签精确匹配。固定 AdamW `1e-3`、weight decay `1e-4`、最多 30 epoch、patience 5，以 validation AUROC 早停；参数量与无条件对照匹配。

early 门槛：query 相对 Global128 AUROC `>=+0.010` 且 4/5 正；相对无条件 query `>=+0.015` 且 4/5 正；相对置乱 `>=+0.020` 且 4/5 正；validation--early gap `<0.15`。失败停止标签条件 token 机制，不调整 query 温度、宽度、epoch 或正则；通过仅授权另一份新鲜 development 的受限残差验证，禁止 test-v3。
