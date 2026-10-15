# Route A VTSE-VBF 目标完成状态审计（H4 与 RAL 后）

日期：2026-08-03  
结论：**总目标尚未完成；test-v3 继续封存且不得授权。**

## 已核验的协议边界

- development-v3 与 test-v3 各有 25,200 个唯一 receiver identity，交集为零。
- development-v3 SHA-256 为 `b55a...ac47`；test-v3 SHA-256 为 `81e6...eb5f2`。
- test-v3 为 `sealed`，且 `access_authorized=false`、`analysis_allowed=false`、`used=false`。
- H3/H4、PGL、RAL、ISRJ 的当前工件均记录 `locked_test_used=false`、
  `test_v2_used=false`、`test_v3_used=false`。

## A--H 阶段状态

| 阶段 | 当前状态 | 原因 |
| --- | --- | --- |
| A 文献/创新审计 | 部分完成 | PGL/RAL/MLC-NC/GEDFD 已做可复现性边界审计 |
| B dev-v3/test-v3 协议 | 完成 | 身份隔离、哈希与密封清单完整 |
| C 六标签 C0--C4 | 实现但未晋级 | C3 在关键目标门与同容量 C4 对照均失败 |
| D 核心 14 消融 | 覆盖完成 | 最新机器审计为 14/14 direct current-protocol coverage |
| E 主流基线 | 部分完成 | 常规损失、DR、RAL完成；无结构兼容的最新外部方法精确复现 |
| F 鲁棒性 | 仅窄 ISRJ 完成 | 高副本、AR(1) 及二者联合复现不等于完整统一候选泛化 |
| G 统计/工程 | 仅组件级充分 | 不存在已晋级 final unified bundle |
| H 最终冻结/test-v3 | 未开始且禁止 | C 阶段前置门未通过 |

## 统一晋级门的直接证据

development-v3 C3 相对 C0 的 macro-AUPR `+0.02789` 与 macro-F1 `+0.03214`
主要来自 RawSaw。VTSE 对三目标的关键增益为：weak recall `+0.00959`（低于
`+0.030`）和 weak AUROC `+0.00264`（低于 `+0.015`）。更关键的是 C3 相对同
参数 C4 的 macro-AUPR `-0.00290`、weak AUROC `-0.00401`。因此无法排除容量
解释；C3 不能成为最终候选。

SawFM 路由安全性与 target FPR/strong-positive 安全门虽通过，但只能证明工程组合
安全，不能弥补机制因果门失败。

## H3/H4 与近期基线的影响

- H3：真实时间戳物理统计可辨识，但真时间戳学习投影相对置换对照的 weak AUROC/
  AUPR 均为负，0/5 正，停止。
- H4：完整 delay-frequency 图的 P0 weak AUROC `0.51769`、真减置换 `+0.00971`，
  低于绝对及控制门，M1/M2 未训练。
- 单脉冲可观测性审计：0716 的 MFT 副本任意两两重叠下界为 `0.80`；VFT 的 1--7 kHz
  跨度仅为有效 100 us 脉冲 Fourier bin 的 `0.60`，最大脉冲内相位仅 `0.70` 周。因此
  原始协议不再接受容量/ROI/loss/阈值形式的 MFT/VFT 重试；这是一项协议约束，不是
  不可分类定理或新的模型结果。
- v25：历史 TS-CLE 的循环 branch comparator 在置换匹配初始化下等价于候选，故其
  MFT/VFT/ISRJ 差值仅保留为 exploratory screen，不能作为目标特异性因果证据或
  MFT 确认依据；其联合 gate 本身也失败。
- RAL：对目标 weak AUROC 仅 `+0.00288`，同时 macro-AUPR `-0.00282`、ISRJ AUPR
  `-0.00673`、ISRJ weak/strong recall 均 0/5 正；Focal-36 仍是普通损失强基线。
- PGL：固定官方源码无法解析且有未定义变量，不能自行修补后称为 PGL。

## 12 项完成标准

| 标准 | 判定 |
| --- | --- |
| 统一六标签晋级模型 | 未满足 |
| 核心消融 | 覆盖完成，但对象未晋级 |
| 公平主流基线 | 部分满足 |
| 多种子统计 | 组件/基线满足，最终模型未满足 |
| 强/弱/负机制 | ISRJ 满足，MFT/VFT 未满足 |
| 分层鲁棒性、外部复现 | 仅窄 ISRJ 满足 |
| 参数效率/工程 | 仅组件级存在 |
| dev-v3 全门通过 | 未满足 |
| 一次性 test-v3、final bundle | 未执行且禁止 |
| 工件和论文边界 | 已完成轮次满足；完整论文主张不可成立 |

## 当前允许的结论

可以主张：冻结 aligned-periodicity 专家在受控合成协议的三个新 receiver-identity
条件（高副本参数、AR(1) 噪声、两者联合）下稳定提高 ISRJ 弱证据；RawSaw 是安全的 SawFM
label-local 路由；RAL 等类别级损失没有复现 ISRJ 增量。

H0s-R1 的联合条件保持全部 B0/B2、calibration 和 threshold 冻结，且在 5,040
个新身份上得到 ISRJ AUPR `+0.11677`、弱召回 `+0.07285`、弱 AUROC `+0.09709`
（三者均 5/5 正；paired bootstrap 区间均为正），ISRJ FPR `+0.00113`（区间跨零）。
这补强 ISRJ 的联合压力鲁棒性，不改变 C3 对 C4 的失败结论、MFT/VFT 空缺或 test-v3
禁止状态。

不得主张：完整六标签 VTSE-VBF 已被验证；MFT/VFT 弱分量稳定改善；完整方法外部
泛化或通过最终独立测试。

## 下一研究门

当前单脉冲 0716 与已关闭的 H3/H4 多脉冲路径不允许再靠扩宽主干、调时间戳/ROI、
loss、阈值或重跑同实体来获得统一门。若重新研究 MFT/VFT，必须先建立与原协议隔离的
MFT delay-aperture/minimum-separation 或 VFT multi-PRI coherent-aperture 观测协议，再获得
一个与已关闭可观测量非重叠、文献可核验的物理机制，并从新的 identity-isolated H0/H1
开始。否则应将论文收缩为 RawSaw + 冻结 ISRJ aligned-periodicity 的窄证据论文，
不得消耗 test-v3。

## 关联工件

- `frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json`
- `outputs/tapc/route_a_core_14_ablation_coverage/core_14_ablation_coverage.json`
- `docs/results/route_a_unified_rawsaw_vtse_vbf_v3_results_20260731.md`
- `docs/results/route_a_h4_vft_beamspace_p0_results_20260803.md`
- `docs/results/route_a_development_v3_ral_loss_baseline_results_20260803.md`

## H0t/H1t asymmetric visibility-transfer closure

H0t verified only a prerequisite on frozen Focal checkpoints: high-power
posteriors contain more ranking information than low-power posteriors for MFT,
VFT and ISRJ. It did not train a model, choose a threshold, or access any test
role. H1t then used a fresh R1 train-only paired source and compared B0
supervision, T1 identity-paired high-to-low transfer, and P1 an exact
composition-stratified teacher-score-marginal permutation control across five
identity-grouped splits.

H1t is closed as a negative result. T1-B0 mean target weak AUROC is `+0.00076`
with a paired 95% CI crossing zero, T1-P1 is `-0.00137` (only 1/5 positive),
and mean target strong AUROC is `-0.08791`. The paired-control result excludes
the claimed identity-specific transfer mechanism for this recipe; the strong
loss excludes it as a safe candidate. No H1t retuning is permitted, and
`locked_test_used=false`, `test_v2_used=false`, and `test_v3_used=false` for
all formal H1t runs. This reinforces, rather than changes, the existing rule:
test-v3 cannot be opened until a genuinely new, independently admitted unified
candidate passes all development gates.

See `docs/results/route_a_h1t_asymmetric_visibility_transfer_r1_results_20260803.md`
and `docs/reports/route_a_h1t_asymmetric_visibility_transfer_stop_decision_20260803.md`.
