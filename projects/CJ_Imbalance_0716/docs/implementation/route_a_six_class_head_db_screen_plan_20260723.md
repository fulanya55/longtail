# 路线 A：六类 Head-DB 三随机种子筛选方案（2026-07-23）

## 1. 扩展目标

四类 Head-DB 已通过四尾类五随机种子确认。本阶段将相同两阶段方法扩展到全部六类，并重点验证新增 SawFM、VFT 是否仍满足整体排序提升与尾类非劣。

本阶段仅使用六类 train 文件的 optimization/development roles，不使用 locked test。

## 2. 六类受控 Label-LT

MILP 可行性搜索覆盖 seeds 2041–2045 和六个尾类，冻结参数为：

- 总训练样本数：2400；
- 尾类目标正例数：400；
- 名义/实际 IF：4.0；
- 标签边际容差：5%，实际求解误差为0；
- 六类顺序：FMB、SawFM、SinFM、MFT、VFT、ISRJ。

对应标签目标约为：`[1600,1213,919,696,528,400]`，按指定尾类轮换排列。头类阳性率约为66.7%，避免四类 N=1600 时头类接近恒正的问题。

## 3. 方法与矩阵

每个 tail/seed 训练：

1. Base-ASL：20 epochs 端到端；
2. Base-DB：20 epochs 端到端；
3. Head-DB：加载对应 Base-ASL，冻结表征和BatchNorm，仅训练分类头10 epochs，学习率1e-3。

正式矩阵：

- tails：6；
- seeds：2041、2042、2043；
- variants：3；
- 正式训练：54次。

## 4. 筛选门槛

每个尾类 Head-DB 相对 Base-ASL：

- macro-AUPR >= -0.005；
- tail-AUPR >= -0.005；
- macro-F1 >= -0.005；
- Hamming loss恶化 <= +0.005。

总体门槛：

- 平均 macro-AUPR > 0；
- 平均 tail-AUPR >= 0；
- macro-AUPR 至少12/18配对获胜；
- tail-AUPR 至少9/18配对获胜；
- 至少4/6尾类的平均macro-AUPR为正；
- Head-DB相对Base-DB的总体tail-AUPR > 0。

SawFM、VFT必须单独满足每尾类非劣门槛。通过后扩展至seeds 2044、2045；失败则按标签和共现组合诊断，不进入locked test。

## 5. 输出

- runner：`experiments/run_route_a_six_class_head_db_screen.py`；
- 输出：`outputs/tapc/route_a_six_class_head_db_screen/`；
- 汇总：`route_a_six_class_head_db_screen_summary.json`。
