# P3o-J：组件条件参数残差的执行封存

## 决定

P3o-J 在 seed 2091 的训练前冻结校验阶段抛出异常，按预注册“任一 exception 即冻结”的规则停止。该分支没有产生 checkpoint、预测、开发指标、确认数据、测试数据或 locked-test 访问；因此它既不是正结果，也不是该组件条件机制的性能反证。

## 已完成但不可用于性能主张的准备工作

- 预注册定义的 J1 是对 detached 六标签 posterior 的 6×12 有界残差修正，直接以同容量共享残差 J0 为对照；J0 不训练 component table。
- 实现与单元测试完成：posterior detach、F-base state/logits 保持、J0/J1 component 梯度隔离、inactive NaN 不进入有效 Smooth-L1。开发前全套回归为 `270 passed, 9 warnings`。
- 1450M/1460M 新身份数据各 5 regime × 3 power × 2,079 identities 已被生成、缓存和审计；缓存仅含 train split，训练/开发身份为零交集。
- 因 C 盘容量不足，本轮数据和输出使用 D 盘路径保存，并保留项目内 C 路径 junction；该存储映射不改变方法或样本内容。

## 触发条件和证据

P3o-J method freeze 记录的 `matlab/generate_p3o_low_power_attribution_data.m` SHA-256 为：

`48c1666d7c62df74c78f39df4e1e55d8244afbd74698df45d5566a271f0b4d20`

seed 2091 实际启动时，该文件 SHA-256 已为：

`ebace24ed18583760027e3a1e95c1801447a402fe09b091a039dafa671b59ceb`

底层 runner 因 generator hash mismatch 正确拒绝继续执行。异常发生在任何训练 loop 前：

- `checkpoint_count = 0`
- `prediction_count = 0`
- `test_generated=false`
- `test_opened=false`
- `locked_test_used=false`

异常回执：`outputs/p3oj_component_conditioned_residual_development_20260802/seed_2091/exception_receipt.json`。

## 不允许的后续动作

- 不更新 P3o-J freeze hash 后重试 2091；
- 不在 1450M/1460M 身份上启动 2092/2093、调参、选 checkpoint 或生成确认；
- 不把现有数据/缓存用于 P3o-J 的性能选择，也不生成或访问测试集。

如要重新研究 posterior-component-conditioned residual，必须用新的预注册、固定的当前生成器哈希、全新的未消费身份及完整 preflight 后另立实验编号。

