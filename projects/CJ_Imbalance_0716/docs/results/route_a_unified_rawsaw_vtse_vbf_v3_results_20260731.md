# Route A 六标签统一 RawSaw + VTSE-VBF development-v3 结果

日期：2026-07-31  
证据等级：五种子 development-v3；test-v3 已封存且未使用。  
结论：统一路由机制成立，但 VTSE 可见性机制未通过容量对照，候选不晋级。

## 1. 数据与证据边界

- development-v3 SHA-256：`b55a361014dfb5b065f242b71c514ee057cd15fa8f5c10950ff126958970ac47`；
- test-v3 SHA-256：`81e6e6eee295d7c36bdc7e84327189e496cfa489cf9c81199b40d37be22eb5f2`；
- development/test 各 25,200 个唯一 receiver identity，交集 0；
- test-v3 `access_authorized=false`、`test_v3_used=false`；
- 五种子 2039–2043，固定 fit/validation/calibration/threshold/early 角色；
- smoke 仅验证代码链路，以下数值均来自正式五种子运行。

## 2. 主要结果

| 模型 | Macro-AUPR | Macro-F1 | 三目标弱召回 | 三目标弱 AUROC | 三目标 FPR | SawFM AUPR | SawFM 弱召回 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 F2+C6 | 0.85050 | 0.73409 | 0.29937 | 0.58902 | 0.18812 | 0.76579 | 0.26188 |
| C1 RawSaw | 0.87795 | 0.76568 | 0.29937 | 0.58902 | 0.18812 | 0.93050 | 0.58259 |
| C2 VTSE-VBF | 0.85093 | 0.73464 | 0.30896 | 0.59167 | 0.19177 | 0.76579 | 0.26188 |
| C3 统一模型 | 0.87838 | 0.76623 | 0.30896 | 0.59167 | 0.19177 | 0.93050 | 0.58259 |
| C4 同参数量无可见性 | 0.88128 | 0.77043 | 0.31193 | 0.59568 | 0.19032 | 0.93050 | 0.58259 |

C3 相对 C0：Macro-AUPR `+0.02789`，Macro-F1 `+0.03214`，三目标弱召回
`+0.00959`，弱 AUROC `+0.00264`。五个种子的三目标弱召回增益均为正，FPR 仍低于
0.20，强正召回没有退化。

## 3. 已证实结论

### 3.1 标签解耦统一不会产生交叉写入

C3 的 SawFM AUPR/弱召回与 C1 完全一致，三目标弱召回/AUROC 与 C2 完全一致；
FMB、SinFM 维持安全旁路。说明“RawSaw 只写 SawFM、VTSE 只写 MFT/VFT/ISRJ”
可以工程上无冲突组合。

### 3.2 RawSaw 在新 development-v3 上得到强复现

SawFM AUPR `0.76579→0.93050`，弱召回 `0.26188→0.58259`。五种子 RawSaw
validation AUPR 为 0.9121–0.9567。RawSaw 仅增加 5,521 个参数，是本轮整体
Macro 指标提升的主要来源。

### 3.3 统一模型参数开销很小

基线 257,511 参数；C3 新增 7,066 参数，其中 RawSaw 5,521、VTSE 1,539、VBF 6，
增幅约 2.74%。C4 新增参数严格同为 7,066，容量公平性检查通过。

## 4. 未通过的核心门槛

1. 三目标弱召回仅 `+0.00959`，低于 `+0.030`；
2. weak-vs-negative AUROC 仅 `+0.00264`，低于 `+0.015`；
3. C3 相对同参数量 C4：Macro-AUPR `-0.00290`、Macro-F1 `-0.00419`、
   弱召回 `-0.00297`、弱 AUROC `-0.00401`；
4. 因此当前 target 增益不能归因于标签内可见性机制，C3 不晋级。

## 5. 失败根因

### 5.1 hard mask 造成正类方向估计方差过大

VTSE 每种子的 fit 弱正数量仅：MFT 134–151、VFT 67–84、ISRJ 66–78；与此同时
负样本约 1,166–1,502。hard mask 完全排除强正，使线性时序方向只由很少的弱正
确定。虽然 fit 上弱正与负类中位数分离明显，但跨角色泛化收益很小。

### 5.2 all-positive 容量对照反而更稳定

C4 使用相同 512→3 头和相同六标量融合器，只把全部正样本作为监督，就同时提高
弱召回、弱 AUROC 和强正召回。这说明强正样本不只是“会主导边界的干扰”，也提供
了稳定的正类方向锚点；完全删除它们属于过度解耦。

### 5.3 VBF 的早停暴露了专家分布不稳定

VBF 在种子 2040/2041/2042 的最佳 epoch 均为 1，仅 2039/2043 能继续优化。
这说明问题已发生在专家分数与基础 logit 的接回阶段，而不是增加融合器训练轮数即可
解决。继续扩大网络没有证据支持。

## 6. 下一项单一整改假设

固定引入低权重强正锚点：弱正与负样本仍为主要边界，强正仅以 0.20 权重约束正类
方向。网络、参数量、角色、校准、阈值、VBF 和残差上限全部不变。

该实验必须同时与 hard-mask C3 和 all-positive C4 比较。若 soft-anchor 仍不优于 C4，
则停止“hard/soft visibility expert”训练目标线，转向以连续物理可见性回归或配对功率
一致性作为新的可证伪机制，不再调整同一损失权重。

## 7. 关键产物

- 预注册：`docs/plans/route_a_unified_rawsaw_vtse_vbf_development_v3_plan_20260731.md`；
- 汇总：`outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3/route_a_unified_rawsaw_vtse_vbf_v3_summary.json`；
- 每种子：checkpoint、五类 history、校准、阈值、early predictions、metrics、receipt；
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。
