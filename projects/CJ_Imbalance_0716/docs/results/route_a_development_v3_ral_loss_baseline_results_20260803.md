# Route A development-v3：RAL 主流多标签长尾损失基线结果

日期：2026-08-03  
证据等级：五随机种子、既有 identity-isolated development-v3 的公平损失比较；不是
VTSE-VBF 候选选择，也不是 external 或 test 结果。

## 固定合同

比较 `L8_RAL_RawSaw` 与已冻结 `E2_Focal_36`。两者使用同一 F2 初始化、部分解冻
范围、RawSaw SawFM 路由、fit/validation/calibration/threshold roles、36 epoch
预算、AdamW、五个种子 `2039--2043` 和每标签 FPR `<=0.20` 阈值规则。唯一训练
机制变化是 RAL 损失。

RAL 公式固定为作者仓库 `WongiPark0628/RAL` revision
`0d7e7712c82ba9a36a08c888bb529885a871bc1f` 的 `Ralloss`：`gamma_neg=4`、
`gamma_pos=0`、`clip=0.05`、`eps=1e-8`、`lamb=1.5`、`epsilon_neg=0`、
`epsilon_pos=1`、`epsilon_pos_pow=-2.5`。项目实现以 batch-label 元素数对官方
sum 做固定正比例归一化；独立公式等价测试通过（`2 passed`）。无参数搜索。

所有运行记录 `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

## RAL − Focal-36 五种子结果

| 指标 | 均值差 | 95% bootstrap CI | 正向种子 | 判定 |
| --- | ---: | --- | ---: | --- |
| macro-AUPR | -0.002820 | [-0.004016, -0.001412] | 0/5 | RAL 退化 |
| macro-F1 | -0.002004 | [-0.003712, -0.000129] | 2/5 | RAL 退化 |
| 三目标平均 weak AUROC | +0.002880 | [+0.002143, +0.003653] | 5/5 | 小幅、稳定但不足以晋级 |
| 三目标平均 weak recall | +0.002128 | [-0.002310, +0.006520] | 3/5 | 不确定 |
| 三目标平均 FPR | +0.001398 | [-0.000860, +0.003495] | 3/5 | 无安全获益 |
| ISRJ AUPR | -0.006730 | [-0.008036, -0.005492] | 0/5 | 明确退化 |
| ISRJ weak recall | -0.005089 | [-0.008279, -0.002082] | 0/5 | 明确退化 |
| ISRJ strong recall | -0.017797 | [-0.021712, -0.014891] | 0/5 | 明确退化 |
| ISRJ weak AUROC | +0.000178 | [-0.001630, +0.001795] | 3/5 | 无证据 |

三目标平均 weak AUROC 的单侧 paired Wilcoxon 与 sign test 都为 `p=0.03125`，但
它的量级只有 `+0.00288`，同时牺牲 macro 和 ISRJ 指标；不能把这一单指标变化
报告为 RAL 优于 Focal-36，也不能作为统一模型的强基线替换。

## 结论与边界

RAL 现在是一个**已完成、公式可追溯的近期多标签长尾数值基线**。在当前六标签
I/Q 协议下，它显示类别级非对称鲁棒损失并不能重现冻结周期性专家的 ISRJ 弱证据
增量；最强普通训练损失仍保留为 Focal-36。该结论支持“频次/正负损失重平衡不等于
标签内物理可见性证据”，但不单独证明 VTSE-VBF，也不消除完整统一候选尚未晋级的
缺口。

不会搜索 RAL 的参数、归约、学习率或阈值；不会把 RAL 接入 H3/H4 或 test-v3。

## 工件

- 预注册：[route_a_development_v3_ral_loss_baseline_plan_20260803.md](../plans/route_a_development_v3_ral_loss_baseline_plan_20260803.md)
- 汇总：`outputs/tapc/route_a_development_v3_ral_loss_baseline_formal_r1/route_a_development_v3_ral_loss_baseline_summary.json`
- 每 seed checkpoint、history、predictions、calibration、threshold 与 receipt：
  `outputs/tapc/route_a_development_v3_ral_loss_baseline_formal_r1/seed_{2039..2043}/`
- 公式代码：`cjlt/losses/classification.py::RobustAsymmetricLoss`
