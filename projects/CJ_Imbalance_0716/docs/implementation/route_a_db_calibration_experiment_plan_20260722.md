# 路线 A：DB Loss 与独立校准正式实验方案

日期：2026-07-22  
状态：已执行（40/40 runs 完成）  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
边界：只使用 optimization、early-stop、calibration、threshold-selection；locked test 禁止读取

## 1. 实验问题

上一阶段中 DB Loss 在 8/8 个配对上提高 macro-AUPR、在 7/8 个配对上提高尾类 AUPR，但固定 0.5 阈值的尾类召回在 8/8 个配对上下降。本段验证该下降主要来自分数标度和逐类阈值失配，还是来自 DB 的真实分类退化。

## 2. 固定训练协议

| 因素 | 设置 |
|---|---|
| protocol | label-LT |
| imbalance factor | 3 |
| label-count tolerance | 20% |
| 尾类 | FMB、SinFM、MFT、ISRJ 轮换 |
| loss | ASL、DB Loss |
| seeds | 2026、2027、2028、2029、2030 |
| optimization / development | 1024 / 300 |
| epochs | 20，patience=20 |
| checkpoint selection | early-stop role 的 macro-AUPR |

共执行 `4 × 2 × 5 = 40` 个训练 run。原始数据本身均衡，所有长尾结论只针对 hash 记录的 label-LT manifest。

## 3. 校准与阈值角色

1. calibration role 只拟合单一温度参数；
2. threshold-selection role 只选择阈值；
3. early-stop role 只用于开发诊断，不参与温度或阈值拟合；
4. locked test 不读取。

温度缩放后比较五种固定算法：

| 模式 | 定义 |
|---|---|
| fixed_0p5 | 所有类别阈值为 0.5 |
| global | 用 threshold-selection 的 subset accuracy 选择单一全局阈值 |
| per_label_f1 | 每类独立最大化 F1 |
| shrinkage_per_label_f1 | 每类 F1 阈值按训练正例数向全局阈值收缩，beta=100 |
| per_label_youden | 每类最大化 TPR-FPR |

主检验预注册为 `shrinkage_per_label_f1`；其余模式用于解释阈值偏差和过拟合风险，不根据 early-stop 结果反向选择算法。

## 4. 验收门槛

DB 进入下一阶段需同时满足：

1. 五种子、四尾类共 20 个 DB run 中，主阈值模式相对 fixed_0p5 的尾类召回平均提高至少 0.10；
2. 主阈值模式的 Hamming 相对 fixed_0p5 平均不恶化超过 0.01；
3. DB 主阈值模式相对 ASL 主阈值模式的 macro-F1 不下降超过 0.01；
4. DB 的 macro-AUPR 配对均值仍高于 ASL；
5. 至少 3/4 个尾类的 DB tail-AUPR 均值不低于 ASL。

通过后才执行 IF=2/5 和六分类扩展；未通过则 DB 只保留为排序指标消融，不作为主分类损失。

## 5. 输出

- 每个 run：模型、history、温度、五种阈值、五种 early-stop 指标、manifest/split/config hash；
- 汇总：逐 run、逐尾类、逐 loss、逐阈值模式的均值/标准差与配对差；
- 所有汇总显式记录 `locked_test_used=false`。

结果汇总：`docs/results/route_a_db_calibration_and_full_review_20260722.md`  
原始汇总：`outputs/tapc/route_a_db_calibration/route_a_db_calibration_summary.json`
