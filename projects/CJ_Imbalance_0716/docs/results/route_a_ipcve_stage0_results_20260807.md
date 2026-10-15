# Route A IP-CVE Stage 0：三元组数据协议结果

日期：2026-08-07  
状态：`passed_data_contract_only`  
证据边界：只证明新身份数据合同和真实 B0 I/O 路径可执行；没有 P0 可观测性结论，没有训练候选模型。

## 1. Stage -1 准入

- 决策：`conditional_admit_stage0_and_p0_only`。
- 正式 B0 定义为 seed 2039--2043 的五 checkpoint bundle，不事后选择单个最优模型。
- 五 checkpoint 哈希与正式 summary 一致，并严格加载到当前 `dfms_iq`。
- H0e、H0f、H0v 均保持 fail/closed；P1 bounded reconnection 仍未授权。
- test-v3 seal 为 `access_authorized=false, analysis_allowed=false, used=false`。

## 2. 身份预检

- 扫描 61 个 live train/pair/source MAT，读取错误 0。
- 历史 live receiver 最大值：`4118682781`。
- smoke receiver：`4188677742--4188677756`。
- development receiver：`4198677742--4198680141`。
- confirmation 预留：`4248677742--4248680141`，本阶段未生成。
- 三段范围互斥，并与历史记录零重合。

## 3. Smoke

- 15 个三元组、45 个视图；MFT/VFT/ISRJ 各 5。
- 首次审计仅因参数 owner 列定义不完整而误报 non-target mismatch；定位后确认差异列属于目标 MFT 的完整参数块，而不是背景漂移。
- 按完整 79 维 schema 修正审计后全部合同通过。
- 最大 `high-absent` 对 `low-absent` 波形比例相对误差：`4.451886949078145e-16`。
- GPU 真实路径：MAT 读取、B0 严格加载、45 条前向、`[45,128,375]` temporal map、一次反向更新和 checkpoint 写入全部通过；smoke probe loss `0.70477545`，gradient norm `0.77783775`。

## 4. Formal development source

- 数据：`E:\LJQ_code\干扰数据\route_a_ipcve_development_cf_v1_20260807\ipcve_triples_train.mat`
- SHA256：`66d7158611e92bc952cbbbd0a9243514d93d101c1297f6841911392c54564fb3`
- 文件大小：334,999,911 bytes。
- 2,400 个三元组、7,200 个视图。
- 每目标 800 个三元组；每目标每 nuisance 160 个。
- 角色：fit 1,440、early 480、evaluation 480。
- 最大波形比例相对误差：`4.919908877140374e-16`。
- receiver/sample identity、标签、完整非目标参数、目标非 JNR 参数、角色、nuisance、波形比例和唯一 receiver 数全部通过。

## 5. 边界

Stage 0 不证明 matched counterfactual 信息可以被 B0 表征恢复。下一步只允许执行冻结 B0 P0，并必须与 unmatched、component-permuted、time-permuted 和等维 raw random projection 比较。P0 失败则停止 IP-CVE，不能进入低秩 adapter 或有界残差训练。

主要机器产物：

- `outputs/route_a_ipcve/stage_minus1/admission_audit.json`
- `outputs/route_a_ipcve/stage0/preflight_identity_audit.json`
- `outputs/route_a_ipcve/stage0/smoke_contract_audit.json`
- `outputs/route_a_ipcve/stage0/formal_contract_audit.json`
- `outputs/route_a_ipcve/stage0/role_triples.npz`
- `outputs/route_a_ipcve/stage0/b0_io_smoke/run_receipt.json`

`locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。
