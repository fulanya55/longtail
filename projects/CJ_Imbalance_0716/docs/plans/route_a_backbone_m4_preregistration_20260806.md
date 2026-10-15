# Route A Backbone M4/Local Statistics Bypass 预注册

日期：2026-08-06  
候选：`B4 = B0 + Local Evidence Statistics Bypass`  
状态：在任何 M4 选型或正式训练前冻结。

## 结构

保留 B0 temporal `[B,128,375]`、global mean/std 和现有 global 128 维投影。新增：

- 每通道 temporal max `[B,128]`；
- 每通道 top-k temporal mean `[B,128]`；
- 拼接为 256，经现有风格的 `Linear(256,128)+LayerNorm+GELU+Dropout(0.2)`；
- `z = z_global + 0.25*tanh(b)*z_local`，`b=0` 初始化；
- 最终仍是 128 维特征和原六标签线性分类器。

只比较 `k=8` 与 `k=16`，预计两者参数均为 290,664。M1/M2/M3 均不启用。候选模块在全部 B0 模块之后实例化，确保共同参数同 seed 逐字节一致。

## 一次性 k 选择

1. 对 k=8 和 k=16 分别只运行 seed 2039、36 epoch/patience 8；两者共享相同数据、角色、初始化和 batch order。
2. 选择只读取 `history.csv` 的 validation 指标：先比较各自最佳 `validation_common_macro_aupr`，完全相同再比较同一行 `validation_target_weak_auroc`。
3. 禁止读取或使用两次选型运行的 early metrics、校准后指标、阈值指标决定 k；runner 虽会为工程完整性生成这些工件，但其数值不进入选择。
4. 选中 k 后写 selection receipt 和 hash；未选 k 永久关闭。
5. 用选中 k 从 scratch 独立执行五 seed 正式训练，再按固定比较脚本与 B0 配对。

若最终候选未通过全部硬门槛，不允许改 k、top-k 定义、残差上限或 pooling 统计后重试；若通过，才运行随机位置 top-k 和同参数 MLP 控制。

test-v2、locked test、test-v3 禁止访问。

## 预实现代码快照

- `phase_multiscale.py` SHA256：`e504c89265a7248f8c1a62e93122b09a36a9fa950633a3b76e43023efe03cf4d`
- `backbone_adapter.py` SHA256：`ed7acdd041ad164e889bd030371f1703d670f1736b2e77cb1944c178d0b06e44`
- 训练 runner SHA256：`c34ca64cc6a3474d0e1d10098e91fedcd7ec9b888712bc76f85b617a2babe4b7`
