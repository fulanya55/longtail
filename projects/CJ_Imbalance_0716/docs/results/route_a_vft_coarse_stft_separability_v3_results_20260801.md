# Route A VFT 粗粒度 STFT development-v3 结果

8×8 粗粒度 STFT 未通过晋级门。它保持了 early AUROC `0.64101`，相对 final-temporal `+0.08086` 且 5/5 种子为正，相对 16×16 仅 `-0.00768`，满足性能非劣；但 validation→early gap 为 `0.19133`，没有下降，较 V0 的 gap 变化反而为 `-0.00661`。

结论：VFT 的泛化落差不是由 16×16 特征维数过高导致，而更可能来自使用 validation AUROC 进行 epoch 选择。下一轮固定回到信息更完整的 16×16 特征，使用不访问 validation 的强正则凸 Ridge 分类器。test-v3 未访问。

主要产物：`outputs/tapc/route_a_vft_coarse_stft_separability_v3/route_a_vft_coarse_stft_separability_v3_summary.json`。
