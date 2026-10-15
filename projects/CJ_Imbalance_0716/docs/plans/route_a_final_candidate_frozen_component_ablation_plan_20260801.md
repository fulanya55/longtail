# Route A 最终候选冻结组件消融预注册

## 目的

当前候选为 `H1 + B2 + MFT shared-decision + VFT Ridge-STFT`。本消融不训练、不拟合、不选阈值，只在已保存的 development-v12 逐角色 logit 中作单组件替换，直接量化每个模块的必要性。

## 固定变体

- `A0`: 完整候选 Q6；
- `A1`: 仅将 VFT logit 换为 Q3 同容量 base-only 分类头；
- `A2`: 仅将 VFT logit 换为 Q5 role-wise shuffled Ridge-STFT 分类头；
- `A3`: 仅将 MFT logit 换回 Q1 H1+B2 父模型，保留 Q1 的 MFT 校准与阈值；
- `A4`: 仅将 ISRJ logit、概率和决策换回 Q0 H1 父模型，保留其余五标签完整候选输出。

所有未被删除标签保持完整候选的 frozen logits 或 frozen 决策。A1/A2 用各自 construction-v3 已冻结 Platt 校准与 FPR 阈值；A3 用共享父级决策契约；A4 用 Q0 ISRJ 决策契约。没有再拟合、没有阈值搜索。

## 指标与边界

报告完整候选相对每个 lesion 的 macro-AUPR、macro-F1、三目标平均弱召回、弱 AUROC、FPR，以及被移除标签的 AUPR/弱召回/弱 AUROC/FPR。A3 额外报告 MFT hard-FPR 与 order-5 FPR。

development-v12 已被使用，只作为冻结后的机制诊断，不能当作新模型选择或项目全局新鲜确认。test-v3 不读取、不授权、不使用。
