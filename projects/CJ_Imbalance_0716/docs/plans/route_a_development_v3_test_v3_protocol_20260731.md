# Route A development-v3 / test-v3 数据协议预注册

日期：2026-07-31  
证据角色：阶段 B；先于六标签统一模型的进一步选择完成。  
冻结边界：历史 locked test 与已消耗 test-v2 均不得访问。

## 1. 目的

建立新的身份隔离数据协议。development-v3 仅承担训练、早停、校准、阈值和模型
选择；test-v3 在生成后立即封存，在阶段 A-G 全部完成且模型、校准器、阈值、融合
规则、代码和依赖全部冻结前，不允许授权、读取信号、计算预测或查看指标。

## 2. 固定生成配置

- 生成器：`C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成`；
- 输出根：`C:\LJQ_code\干扰数据\复合干扰数据0731_v3`；
- 基础种子：`24000000`；
- development-v3 样本种子区间：`[24000001, 24025200]`；
- test-v3 样本种子区间：`[27000001, 27025200]`；
- 每个角色均为 63 个非空六标签组合，每组合 400 条，共 25,200 条；
- 接收机噪声身份由样本种子经固定双射映射产生。

## 3. test-v3 封存规则

1. 生成 test split 后，只允许读取文件字节用于 SHA-256，以及读取
   `receiver_noise_seed` 用于身份泄漏审计；
2. 禁止读取 I/Q、标签、参数、弱分量 mask、类别统计和任何性能相关字段；
3. 验证 development-v3 与 test-v3 身份各自唯一且交集为空；
4. 写入不可覆盖的 `test_v3_sealed_manifest.json` 和
   `TEST_V3_DO_NOT_OPEN.json`；
5. 将 test-v3 文件置为只读；
6. 当前 `access_authorized=false`、`test_v3_used=false`；
7. 最终执行前另行生成一次性授权和排他 `started guard`，执行失败也视为已消耗。

## 4. 当前允许与禁止的操作

允许：在 development-v3 内构造身份隔离的 fit、validation、calibration、threshold
和 final-development 角色；开展五种子模型比较、消融与鲁棒性分析。

禁止：根据 test-v3 选择网络、损失、超参数、校准、阈值、融合权重、弱分量定义或
晋级门槛；禁止以 smoke 或短跑替代正式 development-v3 证据。

## 5. 阶段 B 完成证据

- 两个数据文件存在且 SHA-256 已记录；
- receiver_noise_seed 唯一性与跨角色交集审计通过；
- test-v3 封存目录不可重复创建；
- 封存清单明确 `historic_locked_test_used=false`、`test_v2_used=false`、
  `test_v3_used=false` 和 `access_authorized=false`。
