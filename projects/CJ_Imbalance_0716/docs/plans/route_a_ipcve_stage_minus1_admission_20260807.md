# Route A IP-CVE Stage -1：历史重叠与准入预注册

日期：2026-08-07  
证据角色：只读准入审计；不是模型效果证据。  
授权边界：不得生成正式数据、训练候选、校准、选阈值、访问 locked test/test-v2/test-v3。

## 1. 唯一研究问题

在 H0e 的 matched absent/low logit margin 和 H0f 的 counterfactual delta prototype 均已正式失败并关闭后，新的 high/low/absent 三状态协议是否具有足够实质差异，只授权进入“新身份数据合同验证 + 冻结 B0 可观测性 P0”，而不构成对旧 counterfactual 机制的换名重试？

## 2. 固定审计对象

- Goal Prompt：`docs/prompts/route_a_ipcve_autonomous_experiment_goal_prompt_20260807.md`
- H0e 计划、结果、生成器、runner 和 machine summary；
- H0f 计划、结果、模型、runner 和 machine summary；
- H0v natural-visibility-graph 无拟合可观测性结果；该固定图熵已关闭，不得改变节点数、统计量或 I/Q 投影后重试；
- B0 正式五 seed summary、五个 checkpoint、terminal receipt 和 scientific-stop manifest；
- 当前 `dfms_iq` 模型实现及 `forward_structure_maps`；
- test-v3 sealed manifest，仅检查状态字段和 manifest 哈希；
- 当前 Git 工作树，只记录 dirty 状态，不修改用户已有变更。

## 3. 预注册准入判据

只有以下条件全部满足，才允许进入 Stage 0/P0：

1. H0e/H0f 的失败状态被显式纳入，且训练代码不得直接复用为候选；
2. B0 必须是正式五 seed bundle，不得事后挑选单个最好 checkpoint；
3. 五个 B0 checkpoint 哈希与正式 summary 一致，并可严格加载到当前 `dfms_iq`；
4. 新协议的第一道学习前门是冻结 B0 上的 read-only P0；
5. 新协议使用 `high > low > absent` 三状态与非目标不变性，而非二元 pair margin；
6. P0 必须战胜 unmatched、component-permuted 和 time-permuted 控制；
7. P1 的低秩/有界接回当前不授权，必须等待 P0 后再次做 H0f 重叠审计；
8. test-v3 的 `access_authorized/analysis_allowed/used` 均为 false。

## 4. B0 选择规则

“唯一 B0”定义为同一正式协议下的五 seed B0 family，而不是一个经结果挑选的 checkpoint。P0 的五个冻结重复分别使用 seed 2039--2043 对应的正式 checkpoint。不得平均后只报告最优 seed，也不得重新训练 B0。

## 5. 数据身份预留

Stage -1 只提出、但不最终确认以下范围：

- development base seed：`4000000000`，预计 receiver range `4198677742--4198680141`；
- confirmation base seed：`4050000000`，预计 receiver range `4248677742--4248680141`。

二者均在 uint32 范围内。Stage 0 materialize 前必须重新扫描全部 live historical train source 和已记录身份范围；存在任何交集即重新预注册，不能生成数据后更换。

## 6. 决策规则

- 全部通过：`conditional_admit_stage0_and_p0_only`。
- 任一失败：`stop_before_stage0_and_repair_admission`。
- 即使通过，也不授权 P1、calibration、threshold、confirmation 或 test-v3。

机器审计必须写入：

- `outputs/route_a_ipcve/stage_minus1/admission_audit.json`
- `outputs/route_a_ipcve/stage_minus1/run_receipt.json`

本阶段没有性能指标；“准入通过”只代表下一项实验在科学上可以被执行，不代表 IP-CVE 有效。
