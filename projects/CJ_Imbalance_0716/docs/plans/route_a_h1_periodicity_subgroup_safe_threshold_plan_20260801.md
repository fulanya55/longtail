# Route A H1+B2 子组安全阈值预注册

## 假设

当前B2风险来自ISRJ阈值只约束threshold role总体FPR，没有约束不同组合阶数和共现背景的最坏负样本FPR。网络、H1、周期专家、融合头和Platt校准全部冻结，仅改变ISRJ全局阈值选择规则。

## 选择规则

在development-v3 threshold role上，从冻结B2校准概率的唯一取值中选择最低可行阈值，从而最大化弱召回，同时满足：

- 总体ISRJ负样本FPR `<=0.20`；
- 每个具备至少50个ISRJ负样本的组合阶数子组FPR `<=0.22`；
- 每个具备至少50个负样本的其他标签present/absent子组FPR `<=0.22`。

阈值仍为单个可部署全局ISRJ阈值；推理时不读取真实组合阶数或共现标签。其他五标签阈值保持B2原值。

## development-v3晋级门

- early最坏受支持一般子组FPR `<=0.25`；
- ISRJ总体FPR `<=0.20`；
- 保留B2相对H1至少80%的弱召回增益；
- 强召回相对原B2下降不超过0.03；
- macro-F1相对原B2下降不超过0.005；
- AUPR和弱AUROC因概率不变必须逐元素一致；
- 5个种子阈值均只由threshold role选择；
- test-v3未使用。

通过后不能在已查看OOD-v1上晋级，必须生成相同参数合同但全新receiver身份的OOD-v2冻结验证。
