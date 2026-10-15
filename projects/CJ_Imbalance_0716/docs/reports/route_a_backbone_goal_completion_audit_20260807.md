# Route A Backbone 强化 Goal：完成性审计

日期：2026-08-07  
审计结论：Goal 的预注册科学停止路径已完整满足。

| Goal 要求 | 当前证据 | 审计结论 |
|---|---|---|
| PY10、E 盘项目和新 train-only development 源 | `route_a_backbone_development_v1` 配置、生成、identity audit；数据 SHA256 `c412…00ac` | 完成 |
| 与历史 identity 零重合、角色隔离 | `identity_role_audit.json`：25,200 unique receiver identity、overlap 0、五角色 hash | 完成 |
| test-v3 永久密封 | seal flags 均 false；B0/M1–M4 summary、terminal 和 stop manifest 均为 `test_v3_used=false` | 完成 |
| B0 真实 MAT/CUDA smoke 与五 seed 复现 | B0 smoke receipt、五个 seed artifact、`b0_summary.json`、terminal receipt | 完成 |
| M1–M4 每次只改一个机制 | 各自预注册、模型注册、runner model key、独立 summary/comparison | 完成 |
| M4 top-k 一次 validation 选择 | k=8/16 的 seed-2039 validation history 与 selection receipt；选择指标未读取 early | 完成 |
| 每候选完整五 seed artifact | 每个正式目录五组 history/checkpoint/metrics/calibration/threshold/predictions/run receipt/seed summary | 完成 |
| 统计、弱/强/负分层和成本收据 | `comparison_to_b0.json`、summary、cost correction（M1/M2）和 terminal receipt | 完成 |
| 晋级硬门槛后的自动决策 | M1–M4 均为 `fail_stage2_close_candidate`；失败 gate 明细已保存 | 完成 |
| 同容量/信息破坏控制 | 不适用：无候选通过 Stage-2，执行该阶段会违反“只有通过者才允许进入” | 正确省略 |
| 模块组合、M5、冻结 probe | 不适用：无通过单模块；M5 的前置条件未满足 | 正确省略 |
| 新身份 confirmation | 不适用：只允许最终冻结成功候选进入 | 正确省略 |
| forward、finite backward、实虚对齐、3000 stage shape | `tests/test_backbone_adapter.py` 覆盖 M1–M4；39 个相关测试通过 | 完成 |
| alpha/beta 初始等价与 local finite | M2/M4 同 seed B0 参数逐字节一致、初始输出精确一致；M4 local path backward 有限 | 完成 |
| checkpoint reload | B0真实 smoke reload receipt；M4 checkpoint save/reload 单元测试 | 完成 |
| test-v3 guard 和 E 盘配置测试 | `tests/test_route_a_backbone_development_v1.py` | 完成 |
| 最终技术设计、负结果与 SCI 证据边界 | 技术设计与证据报告、四个候选结果、科学停止报告 | 完成 |
| 所有哈希当前一致 | stop manifest 中 39 个工件 SHA256 已重新逐项核对，无 mismatch | 完成 |

## 关键停机事实

M1–M4 都未同时满足 macro-AUPR、目标弱正 AUROC、目标弱 recall、seed 一致性与安全标签门槛。因而不存在可合法进入因果控制或 confirmation 的候选。继续在同一 development-v1 上扩大网络、调整残差、替换核、改变阈值或组合失败模块，会违反预注册停止规则。

## 当前可声称的结论

只能声称四种指定 backbone 增强在新的 identity 隔离 development 来源上失败，及 M3 的 VFT/ISRJ 标签特定局部证据现象。不能声称统一 F2-EPF backbone、不能外推到真实/HIL 数据或 test-v3，也不能将开发结果视作独立确认。
