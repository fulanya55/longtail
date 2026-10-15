# Route A Backbone M3/EPB 预注册

日期：2026-08-06  
候选：`B3 = B0 + Evidence-Preserving Multi-Scale Block (EPB)`  
状态：在任何 M3 效果训练之前冻结。

## 唯一结构改动

在 B0 原 temporal backbone 输出 `[B,128,375]` 与 global mean/std 之间加入一个 EPB：

- 三个并行 depthwise Conv1d 分支，通道均为 128；
- `(kernel,dilation)` 固定为 `(3,1), (15,2), (31,4)`，保持长度 375；
- 三分支各自仅做 depthwise convolution，不加入注意力或 sigmoid 门控；
- 拼接 384 通道，经 `Conv1d(384,128,1,bias=False)+GroupNorm(8)+SiLU`；
- 输出以普通 `x + encoded` 残差相加，不使用可学习残差比例；
- global mean/std、128 维 projection、dropout 与六标签线性分类器不变。

预计新增 55,680 参数，总参数固定为 313,191。EPB 在所有原 B0 模块初始化完成后再实例化，保证同 seed 下 B0 参数子集与 B0 逐字节一致。M1、M2 均不启用。

## 固定训练、统计与关闭规则

数据、五 seed、角色、初始化、batch order、损失、优化器、LR、36 epoch/patience 8、checkpoint key、Platt、FPR≤0.20 阈值与 early 报告完全继承 B0。先进行结构/梯度/参数与共同初始化测试，再进行真实 MAT/CUDA 2-epoch smoke，之后才能启动正式五 seed。

使用固定 `compare_route_a_backbone_candidate.py` 与 B0 配对比较，全部硬门槛不变；安全标签系统性退化操作定义继续使用 M2 前冻结版本。若失败，不允许改 kernel、dilation、加门控、改变残差比例或延长局部训练后重试；若通过，才执行匹配参数固定单尺度和 dilation 置换控制。

test-v2、locked test、test-v3 均禁止访问。

## 预实现代码快照

- `phase_multiscale.py` SHA256：`f8b0bbaff348b61b6919791cd0099507c5a4b92ed8ebb84070f52268d530e670`
- `backbone_adapter.py` SHA256：`4eaf20a0169abbab7c52b327fa8b092f57f4a234aac1cc4483a437a58ddd4825`
- 训练 runner SHA256：`b8d321239b537337c837c28b135f8ecb26c90a9ff68b019984f6a2c3076c19e8`
