# Route-A 新六类 locked-test 数据绑定计划（2026-09-08）

## 目的

为已冻结的 `A-final-prelock` 候选绑定一个真正独立的 E 盘 sealed test。该测试必须覆盖 10、15、20、25、30、35、40、45、50 dB，并且其 `receiver_noise_seed`/身份不能与训练、验证、开发测试或任何模型选择数据交集。

## 当前审计结论

1. 历史 `C:\LJQ_code\...0731_v3` 测试源不是本分支数据，排除。
2. `E:\论文相关\干扰数据\复合干扰数据0716\九类\power_sweep_9level_v1` 的九个 test 文件已由 C1 基线作为开发评估打开，排除。
3. `full_power_10_30dB_v1` 未被 C1 role manifest 列出，但只覆盖 10--30 dB，只能作为经明确批准的部分功率诊断，不能支撑完整声明。
4. `CJ_Inbalance_0724/generated_data/p2a/p2b` 虽覆盖九功率，但属于另一条 Route-B 数据线并已有 Route-B 开发使用，不作为本次新六类的独立 locked test。
5. 当前 E 盘工作区未发现可直接复用、覆盖九功率且未被开发使用的完整测试源；原 MATLAB 包装入口硬编码历史 C 盘路径，但 E 盘仍保留可迁移的生成器核心依赖。本次已建立 E 盘 test-only 生成脚本并通过 dry-run，尚未写入样本。

## 推荐动作

在不触碰 A-final-prelock 模型和阈值的前提下，使用已通过 dry-run 的 `experiments/generate_route_a_locked_test_source.m`，以 generation seed `4050000000` 和 test-only 模式生成九个 E 盘 test 文件；生成后只做一次身份、标签、功率覆盖和 SHA256 审计，不读取样本用于模型选择。随后生成不可变 sealed manifest，写入路径、字节数、哈希、身份交集计数和 `locked_test_used=false`。

## 失败边界

- 生成器依赖无法在 E 盘恢复：保留 dry-run/依赖缺失收据，不改用已有开发 test 冒充 locked test。
- 新源无法覆盖全部九个功率：只能登记为 partial diagnostic，禁止宣称完整功率泛化。
- 身份交集非零：停止绑定并重新生成，不进入授权阶段。

## 授权顺序

1. 完成新源生成和只读审计；
2. 固定 sealed manifest、模型 manifest 和一次性 runner；
3. 用户明确授权；
4. 一次性读取新 sealed test，先输出原始结果，再生成统计报告；
5. 读取后禁止任何模型、阈值、数据划分或报告口径修改。

当前状态：`binding_required_before_authorization`。本计划不读取任何 locked-test 样本。
