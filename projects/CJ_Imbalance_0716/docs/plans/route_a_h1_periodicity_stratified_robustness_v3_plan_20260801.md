# Route A H1+B2 development-v3 分层稳健性预注册

冻结读取 B0 H1、B2对齐周期和B3 block-shuffle在development-v3保存的early预测，复用既有F3分层定义和最低支持度，不重新训练、校准或选择阈值。

检查相对功率、JNR、sample width、组合阶数、共现标签、seen/unseen组合和receiver cohort；主要门槛沿用既有分层门，额外如实报告最坏一般子组FPR。该分析用于确认外部OOD观察到的FMB-present和三阶组合风险在development-v3是否已有对应迹象。

若最坏子组FPR明显高于0.25或关键层弱AUROC为负，不修改本轮门槛；进入新的安全机制development和新的外部数据验证，禁止在已查看OOD-v1上调阈值。
