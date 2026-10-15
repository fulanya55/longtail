# Route A Backbone M2/CSFP 预注册

日期：2026-08-06  
候选：`B2 = B0 + Cross-Stage Complex Feature Pyramid (CSFP)`  
状态：在任何 M2 效果训练之前冻结。

## 唯一允许的结构变化

从原 B0 complex encoder 同时读取：Stage2 `[B,32,1500]`、Stage3 `[B,48,750]`、Stage4 `[B,64,375]` 的实部和虚部。每层构造 `[real, imag, magnitude]`：

- Stage2：96→64，`Conv1d(1x1,bias=False)+GroupNorm(8)+SiLU`；
- Stage3：144→64，同上；
- Stage4：192→64，同上；
- Stage2 用两级固定 `[1,4,6,4,1]/16` 低通加 AvgPool2 对齐至 375；Stage3 用一级；Stage4 不降采样；
- 三层拼接为 192 通道，再经 `Conv1d(192,128,1,bias=False)+GroupNorm(8)+SiLU`；
- 现有 Stage4 late-fusion 输出为主路，CSFP 以 `0.25*tanh(a)` 的有界残差相加，`a=0` 初始化；随后进入原 temporal backbone、global mean/std、128 维投影和线性分类头。

禁止同时启用 M1。除 CSFP 外，B0 全部模块保持不变。预计新增 52,865 个可训练参数，总参数固定为 310,376；参数增幅约 20.5%，仍低于总体预算。

## 固定训练与评价

完全继承 B0：development-v1 数据及 SHA、五 seed 与 role hash、从 scratch 初始化规则、batch order、36 epoch/patience 8、Focal 损失、AdamW、两组学习率、Platt、FPR≤0.20 阈值和 early-only 报告。不得加载 B0/M1 权重，不得根据 smoke 或 early 改结构、学习率或残差上限。

执行顺序：结构/梯度/参数测试 → 真实 MAT/CUDA 2-epoch smoke（仅工程证据）→ 五 seed 正式训练 → 固定比较脚本配对统计。

## 晋级规则

严格使用 Goal 的全部 Stage-2 门槛。对“FMB、SawFM、SinFM 不得系统性退化”的固定操作定义为：任一安全标签 AUPR 平均差 `< -0.01`，或该标签至少 4/5 配对 seed 的 AUPR 差为负，即判定失败。该定义在 M2 正式结果产生前冻结，并继续用于 M3/M4。

只有全部 Stage-2 门槛通过，才执行 equal-capacity control 与 Stage2/Stage3 时间对齐破坏控制；否则关闭 M2，不允许在同源上修改通道数、残差上限、对齐方法、训练轮数或阈值后重试。

test-v2、locked test、test-v3 全部禁止访问；M2 仍然只是 development-only 候选证据。

## 预实现代码快照

- `phase_multiscale.py` SHA256：`a2429a9b2d42b63bbaad23f88923bcc19e1615e3772effca7063f22934c77cc`
- `backbone_adapter.py` SHA256：`6493cb9f1e2434819cfb7cdea10410deb9341f2dc258481387fee7399f487038`
- 训练 runner SHA256：`c761a845dd94dd1f982db44e82b6a5819c7980a3723296e726aa69a9365ffccc`
- 固定比较脚本 SHA256：`52fc17880ab209a514fa66d61f62224382eda7ce7e4b9f5fe98546060c5342d3`
