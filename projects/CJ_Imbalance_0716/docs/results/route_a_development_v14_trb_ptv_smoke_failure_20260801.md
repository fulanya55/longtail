# Route A development-v14 TRB-PTV：冻结求解器 smoke 停止记录

## 结论

决定：`stop_trb_ptv_before_formal_evaluation`。TRB-PTV 未进入 v14 正式五种子评价，也没有形成任何性能结论。

在冻结实现的单 seed、1 epoch、截断角色 smoke 中，fit-only SLSQP 从预注册初值 `(1.5,1.5)` 求解三目标弱分量风险边界时返回 status 8（`Positive directional derivative for linesearch`）。返回点仍是 `(1.5,1.5)`；MFT 的风险相对 C6 为 `+0.01018602`，远高于冻结容差 `1e-7`。因此求解器契约失败，seed 无效。

## 证据边界

- 该运行是 smoke，不报告 macro-AUPR、macro-F1、阈值、校准或正式开发性能。
- 训练产生的 C6/M5/掩码端点和 partial artifacts 仅用于定位实现路径；它们不能作为 TRB 或任何基线的开发结果。
- 未启动正式五种子，也没有任何 test 文件、locked test、test-v2、test-v3 或 test-v14 访问。
- v14 train-only MAT 已生成并通过身份审计，但该冻结候选不能再在该 identity 上用修改后的 solver 或机制重跑。

## 冻结失败事实

|项目|值|
|---|---:|
|seed|2039|
|q0 / returned q|`(1.5,1.5)` / `(1.5,1.5)`|
|C6 MFT/VFT/ISRJ weak BCE|1.12516869 / 0.95628466 / 1.01421022|
|returned MFT/VFT/ISRJ weak BCE|1.13535471 / 0.94002317 / 0.97863511|
|risk delta vs C6|+0.01018602 / -0.01626149 / -0.03557511|
|最大约束违反|0.01018602|
|固定容差|1e-7|

预注册要求 q 求解固定为 SLSQP、初值 `(1.5,1.5)`、`maxiter=200`、`ftol=1e-12`，且风险约束必须复核通过。不得因本结果更换初值、加入 fallback、缩放风险、修改边界或改变激活门。

## 制品与可追溯性

- 失败 receipt：`outputs/tapc/route_a_development_v14_trb_ptv_smoke_v2/run_receipt.json`。
- partial smoke artifacts 均保留在同一目录，不删除、不纳入正式比较。
- v14 train identity audit：`outputs/tapc/route_a_development_v14_generation/identity_audit.json`，MAT SHA256 为 `7fdf86cb45feaef81ed8325a60ef99da4e5ab9f06cf1911b0815407700d35e58`。
- 实现 repair freeze：`outputs/tapc/route_a_trb_ptv_implementation_repair_v2_freeze/implementation_freeze.json`。

## 停止规则执行

TRB-PTV 是 FC-PTV 停止后的第二个配对任务向量稳定化机制循环。该候选在冻结求解器的 smoke 路径即未满足数值/可行性契约，故本候选停止；不得生成另一份数据或通过修补当前约束重新尝试。只有一个实质不同、重新完成文献审计和预注册的后续研究问题才可被单独提出。
