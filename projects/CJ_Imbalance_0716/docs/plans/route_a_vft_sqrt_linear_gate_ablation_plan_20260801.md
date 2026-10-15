# Route A VFT 平方根门与线性门消融预注册

## 机制问题

VFT 融合头当前使用 `sqrt(4p(1-p)) * z`。该形状强调边界不确定区域；若换成 `4p(1-p) * z` 后性能相同，则平方根门不是必要设计。

## 固定设计

- 使用 development-v3、五个既有角色与已经冻结的 Q2 H1+B2+MFTGuard logit；
- 同一固定 Ridge-STFT 分数、同一 fit 统计标准化、同一 `C=0.1` 四输入逻辑回归、同一弱 VFT 正样本权重 4；
- `G0`：`[base, z, sqrt(4p(1-p))*z, tanh(base)*z]`；
- `G1`：`[base, z, 4p(1-p)*z, tanh(base)*z]`；
- 两者均重新执行相同的 fit/calibration/threshold 角色协议，FPR 上限为 0.20；不搜索参数；
- G0 必须与已保存 Q4 的 VFT logit 复现一致，作为实现校验。

## 判定

报告 G0-G1 的 VFT AUPR、弱召回、弱 AUROC 与 FPR。该消融只回答门函数形状，不用于更换已冻结候选；test-v3 禁止访问。
