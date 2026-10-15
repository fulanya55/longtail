# RAAT-H0/v1.1 阶段 0 安全修订

日期：2026-08-02  
状态：**在任何正式数据生成、训练或 H0 指标之前作出的预注册修订。**

## 触发原因

`RAAT-H0/v1` 曾将 weak-positive mask 放在每个角色的 matched view 构造环节。即便该 mask 不输入网络，该选择也会从真实 component power 派生训练样本集合，不能满足“训练不依赖可见性真值”的部署和因果边界。因此 v1 不再可执行；它保留为历史冻结文件，v1.1 的唯一候选仍为 RAAT，但采用以下更严格的样本角色。

## 唯一修改

1. 对 fit、validation 和 early 的训练/完整性 view，先**仅用**目标标签全部正负、其他五标签组合、总标签数、I/Q 计算的全 PRI 接收能量五分位，按每 strata 较小类数下采样。不得读取 weak mask、component power、parameter 或 receiver seed 来构造 map、标准化、训练、validation 或 batch。
2. 所有臂均在该 label-only view 上训练 30 epoch。没有 early stopping、没有 checkpoint 选择。
3. 只在 epoch-30 的 early 预测已经写出后，由隔离的离线评估函数读取 component power 计算 weak mask，组成 weak-positive vs matched-negative 主端点。这个数组不得返回给 extractor、标准化、训练、模型或样本选择函数。
4. primary endpoint 的样本量门改为：early role 内 target weak positives 和 target negatives 均不少于 160；不满足即停止，不补样本、不改组合规则。

## 不变部分

RAAT 的数学、LFM/延迟/Doppler 网格、三种 map、错误参考控制、网络 `7425` 参数、BCE、AdamW、30 epoch、五 seeds、身份范围 `978677743--978682782`、统计门、两标签晋级条件、测试封存和永久禁止项均逐字继承 v1，未作改变。

## 实施影响

在 v1 实现草稿中已经编写的代码不能被视为已冻结实现。只有 runner 被人工确认遵循此 amendment、静态审计隔离 metadata 读取、单元测试和内存 smoke 再通过，并形成新的实现 manifest 后，才允许 materialize train-only source。
