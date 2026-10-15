# 可复现最新多标签长尾基线候选审计：MLC-NC

## 目的

此前的 GEDFD（TMM 2025）没有可审计的完整算法或公开代码，不能以近似实现作为数值基线。本审计寻找一个**不同的、可公开审计**的近期多标签长尾方法，供后续在六类 I/Q 协议中作公平复现；本文件不报告模型结果，也不改变当前 VTSE-VBF 候选。

## 只读来源核查（2026-08-01）

- 论文：AAAI 2025, *MLC-NC: Long-Tailed Multi-label Image Classification through the Lens of Neural Collapse*。
- 公开仓库：`t1neo666/AAAI25_MLC-NC`；其 README 明确将代码关联到该论文。
- 仓库树包含训练入口 `train.py`、推理入口 `infer.py`、多标签损失、类别原型/分类器和采样模块，而不是只有论文说明或模型权重。
- 这只证明代码可审计，不证明原仓库可直接用于雷达 I/Q 数据；原始实现依赖图像数据集、ResNet/TResNet 和其 VOC/COCO/VG 标注协议。

来源：

- [AAAI 2025 paper page](https://ojs.aaai.org/index.php/AAAI/article/view/34298)
- [authors' public repository](https://github.com/t1neo666/AAAI25_MLC-NC)

## 源码级适配结论

完整源码审计后，MLC-NC **不能作为当前 F2/PAM-IQ 的原样同主干基线**。其训练入口固定使用：

- 图像 backbone 的空间 token，随后 `2048 -> 768` 映射；
- 每标签查询向量和 8-head cross-attention；
- 每标签 `768 -> 20` 专属投影；
- 将六/二十个投影拼接后与 ETF prototype classifier 相乘；
- 前 5 epoch 只用 feature-label loss，随后使用
  `0.1 * contrastive-prototype + 1.0 * TwoWay + 0.5 * feature-label`；
- class-aware image sampler、EMA 与 VOC/COCO/VG 的类别元数据。

当前 F2 的可复用公开接口是一个全局 128 维向量，既没有空间 token，也没有源码所需的标签查询注意力。把原代码的 attention、投影或 classifier 改造成 128 维版本会改变方法的表征和容量；这种实现最多能叫“受 MLC-NC 启发的自定义头”，不能在论文中作为 MLC-NC 的公平复现。

## 方法边界和可迁移单元

MLC-NC 的核心是用 neural-collapse 相关的表征/分类器约束处理多标签长尾，而非显式的标签内弱正—强正可见度分层。因此它可以作为文献层面的强比较对象，但不能替代 VTSE 的机制消融。由于上述结构不相容，当前项目不能把它列为同主干数值基线；不能仅移植一部分 loss 后仍使用 MLC-NC 名称。

## 进入实现前必须完成的静态适配审计

1. 若要做**方法移植研究**，必须先注册新的 token-level PAM-IQ backbone 和同参数 token-attention control；这属于新的候选架构，不属于 E 阶段同主干基线。
2. 若要满足 E 阶段的“同主干最新方法”要求，应继续寻找定义在 logits/global-feature level、且有公开官方代码的多标签长尾方法。
3. 对任何新公开实现，锁定上游 commit SHA-256/commit id、许可、`requirements.txt` 与全部训练入口文件哈希；不得引用浮动分支。
4. 不得从 validation、early、test-v2 或 test-v3 估计类别频率、原型、采样权重或任意超参数。

## 当前裁决

MLC-NC 被登记为“**公开可审计但与当前同主干协议不兼容的近期方法**”。它解决了 GEDFD 无公开代码的问题，却不能解决公平数值比较的结构兼容性问题。因此 `latest_multilabel_longtail_equal_protocol_baseline` 仍为未完成状态，test-v3 继续封存。
