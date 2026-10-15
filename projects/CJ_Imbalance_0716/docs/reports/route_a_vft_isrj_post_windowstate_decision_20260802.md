# VFT--ISRJ：window-state 后续机制裁决

## 当前证据

- 配对干预证明：在其他分量、噪声和 receiver identity 固定时，ISRJ 窗宽增大显著提高冻结 Q4 的 VFT hard-negative FPR；副本数增大方向相反。
- 新鲜 window-state 屏幕否定了固定低维相位重置统计：AUROC `0.4678`，低于相位置乱控制 `0.5323`。
- 既有证据还排除了：普通/高分辨率 STFT 网格、浅层 STFT-CNN、Raw+STFT 低维补充和 VFT--ISRJ 低维 logit cross-head。
- 0724 只读参考的 P3a 记录表明，学习型 ISRJ 时域波形分离器也未能把 exact-component oracle 转移到复合 I/Q；因此不得把它移植为本项目的下一轮“新”方案。

## 下一唯一假设

PAM-IQ 的 `forward_temporal_features` 已保留 final-temporal `[B,C,T]`；现有 F2 路径将其 global pooling 后投影为共享 128 维向量。假设是：VFT--ISRJ 的可用高维时序证据存在于 token 序列中，但被全局池化消除；标签条件的 token 读取能在严格 composition-matched weak-VFT vs VFT-absent/ISRJ-present 视图上超过当前 global/STFT probes，并对每记录时间置乱敏感。

这不是增加一个任意专家，也不是重试已否证的 STFT/分离器。下一实施前必须生成新的、未被上述失败结果使用的 source-v2 identity 数据；先做冻结 F2 token 可识别性屏幕（global-vector、token-readout、per-record time-shuffle token-readout 的等参数对照），不接回候选、不训练主干、不访问 test-v3。只有该屏幕通过，才允许在另一份新鲜 development 数据上设计标签条件 token residual。

## 禁止事项

不再调 phase-reset 阈值、STFT 尺寸、RawSource lags、VFT--ISRJ logit head，亦不移植 0724 已失败的时域 separator。当前 `H1_B2_MFTSharedDecision_VFTRidgeFusion_v2` 仍不具备 test-v3 资格。
