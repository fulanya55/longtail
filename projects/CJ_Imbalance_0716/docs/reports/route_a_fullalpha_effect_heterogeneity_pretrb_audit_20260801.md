# Route A FullAlpha1p5 效应异质性定位与新机制预注册前审计

## 结论

**阶段 A 决定：`ELIGIBLE_FOR_TRB_PTV_PREREGISTRATION`。** 这是只读历史预测的资格判断，不是新机制的效果结论。

这不是 FullAlpha1p5 无效的结论；它是“现有历史制品不足以支持条件化任务向量机制”的结论。所有数值仅为历史开发证据，禁止用于重新选择模型、阈值或机制。

## 审计边界与来源

- 只读输入：development-v4/v5/v6/v12 的官方 summary、逐 seed metrics、thresholds/calibration、manifest、receipt、正式报告和已保存的 `predictions.npz`；没有加载 checkpoint、MAT 数据，也没有执行模型前向或历史重训。
- 完整性复核：每轮对 C6/M5/候选的 15 个 seed×variant 组合读取 metrics、thresholds、calibration；四个 summary 与相应 metrics 的 macro-AUPR、macro-F1、target weak recall、target FPR 全部一致。
- 对照：C6_Joint、M5_GradientIsolated；候选为 v4 的 GradientContrastFixed1p5 与 v5/v6/v12 的 FullAlpha1p5，均代表冻结的 `theta_C6 + 1.5(theta_M5-theta_C6)` 方向。
- 每个来源均为 5 个完成的开发 seed；全部 `smoke=false`、`locked_test_used=false`。本报告不触碰 locked test 或任何 test-v2/test-v3。
- 分数差为候选减对照；FPR 的正差表示更高 FPR，不能当作改善。`p/n` 是正差 seed 数/有效 seed 数。

|历史来源|候选|summary SHA256|manifest SHA256|receipt SHA256|report SHA256|seed|
|---|---|---|---|---|---|---|
|development-v4|GradientContrastFixed1p5|`80657250d6360fb3d7f3e158cd36ab49eb46ff6bf3c03d934faa081b6295d0d9`|`1c11388c522b76bc7288c697e3fb1af6bf69cf85a08913ff646e460ee7939142`|`f9dc613463b2942085aa1303f931b03e2b1e7ea4373f5bdc8d68899adbef15dd`|`14acc06da174f546e0712a9ccb0b3f07c5b62b763666f541c007645efe07d9d6`|2039,2040,2041,2042,2043|
|development-v5|FullAlpha1p5|`c674bd82dd477c7fccca7bc2b2bb7425dc6a5dcd84868d7ac5e01d6b3a17f8bf`|`fdd2067c4c64f299a14f493fa60c0eea6bcdb84eae27df55b255523efdab6370`|`61353cd09b2a4291fe2f360b1162cbc0102732f8b53fcb9f9143d06cef66e50a`|`e31077d77d3bfea228c6486ee7043f5e55849b6c90b49022fed655871106ce33`|2039,2040,2041,2042,2043|
|development-v6|FullAlpha1p5|`21d96a09383ddb5b6191d2944dbe61dda0c7d39170086a585c91bb39a97cf703`|`fbfe364b78d2fd5807de37903f493a8af10a2facc856febe17197eadd74439f9`|`efc1398925ccae9c4deb3ccd62ca0144d19698dcf14c41b5b742ee641caa6513`|`c0259998ce9c51e8d35559585915f9302a2fa80f6e7d18cd6dcc1b3a314d01ed`|2039,2040,2041,2042,2043|
|development-v12|FullAlpha1p5|`d7714af43035ff53ed9369d32950197614ef69c9d5d1e5db752b0b5375281c01`|`8a485af7ec1a1f09b109c8ee9d24e32f1ddeaebdb4ad38d51d98079cd94bbb0a`|`2857373fc051ef69105e2ece0aaebbc1ad576bc9904563ed53167c26c7c466eb`|`67c6dd2fee336036e8dd30a8e2fbacf42bec0684ec6845a615f3ed6298e564e2`|2039,2040,2041,2042,2043|

|来源|metrics.json|thresholds.json|calibration.json|prediction-like 文件|manifest 条目|
|---|---:|---:|---:|---:|---:|
|development-v4|49|49|49|89|9|
|development-v5|81|81|81|121|9|
|development-v6|45|45|45|85|9|
|development-v12|45|45|45|90|NA|

## 全局与目标标签：可见的历史差异（非条件机制证据）

### development-v4：候选 − C6

|宏 AUPR|宏 F1|目标弱分量 Recall|目标均值 FPR|
|---:|---:|---:|---:|
|+0.00646 (5/5)|+0.01163 (5/5)|+0.01691 (5/5)|+0.00005 (3/5)|

|变体|宏 AUPR|宏 F1|目标弱分量 Recall|目标均值 FPR|
|---|---:|---:|---:|---:|
|C6_Joint|0.86594|0.75172|0.29037|0.19602|
|M5_GradientIsolated|0.86980|0.75998|0.30362|0.19849|
|GradientContrastFixed1p5|0.87240|0.76334|0.30728|0.19608|

|目标标签|C6 weak recall|M5 weak recall|候选 weak recall|候选−C6|候选−M5|
|---|---:|---:|---:|---:|---:|
|MFT|0.35331|0.37928|0.38624|+0.03293 (5/5)|+0.00695 (3/5)|
|VFT|0.26532|0.27390|0.28430|+0.01898 (3/5)|+0.01039 (4/5)|
|ISRJ|0.25247|0.25768|0.25131|-0.00116 (2/5)|-0.00636 (1/5)|

最差的单个 target-weak-recall 配对：seed 2041，ISRJ，候选−C6 = -0.01365。

|目标标签|弱分量 Recall|AUPR|FPR|
|---|---:|---:|---:|
|MFT|+0.03293 (5/5)|+0.00597 (5/5)|+0.00887 (4/5)|
|VFT|+0.01898 (3/5)|+0.01164 (5/5)|+0.00387 (4/5)|
|ISRJ|-0.00116 (2/5)|+0.00686 (5/5)|-0.01258 (0/5)|

### development-v5：候选 − C6

|宏 AUPR|宏 F1|目标弱分量 Recall|目标均值 FPR|
|---:|---:|---:|---:|
|+0.00632 (5/5)|+0.01464 (5/5)|+0.02813 (5/5)|+0.00108 (2/5)|

|变体|宏 AUPR|宏 F1|目标弱分量 Recall|目标均值 FPR|
|---|---:|---:|---:|---:|
|C6_Joint|0.86367|0.74935|0.29456|0.19145|
|M5_GradientIsolated|0.86724|0.76021|0.31655|0.19274|
|FullAlpha1p5|0.86999|0.76400|0.32269|0.19253|

|目标标签|C6 weak recall|M5 weak recall|候选 weak recall|候选−C6|候选−M5|
|---|---:|---:|---:|---:|---:|
|MFT|0.36007|0.39878|0.41502|+0.05495 (5/5)|+0.01624 (4/5)|
|VFT|0.27292|0.29024|0.29172|+0.01879 (5/5)|+0.00147 (2/5)|
|ISRJ|0.25067|0.26062|0.26133|+0.01065 (5/5)|+0.00071 (2/5)|

最差的单个 target-weak-recall 配对：seed 2043，ISRJ，候选−C6 = +0.00321。

|目标标签|弱分量 Recall|AUPR|FPR|
|---|---:|---:|---:|
|MFT|+0.05495 (5/5)|+0.00709 (5/5)|-0.00065 (3/5)|
|VFT|+0.01879 (5/5)|+0.01227 (5/5)|-0.00016 (2/5)|
|ISRJ|+0.01065 (5/5)|+0.00609 (5/5)|+0.00403 (4/5)|

### development-v6：候选 − C6

|宏 AUPR|宏 F1|目标弱分量 Recall|目标均值 FPR|
|---:|---:|---:|---:|
|+0.00586 (5/5)|+0.01266 (5/5)|+0.02454 (5/5)|+0.00043 (2/5)|

|变体|宏 AUPR|宏 F1|目标弱分量 Recall|目标均值 FPR|
|---|---:|---:|---:|---:|
|C6_Joint|0.86245|0.74568|0.28522|0.19430|
|M5_GradientIsolated|0.86590|0.75370|0.30123|0.19629|
|FullAlpha1p5|0.86831|0.75834|0.30976|0.19473|

|目标标签|C6 weak recall|M5 weak recall|候选 weak recall|候选−C6|候选−M5|
|---|---:|---:|---:|---:|---:|
|MFT|0.34546|0.37825|0.38424|+0.03878 (4/5)|+0.00599 (5/5)|
|VFT|0.26412|0.26605|0.28138|+0.01726 (4/5)|+0.01532 (4/5)|
|ISRJ|0.24609|0.25940|0.26367|+0.01758 (5/5)|+0.00427 (3/5)|

最差的单个 target-weak-recall 配对：seed 2042，MFT，候选−C6 = -0.02349。

|目标标签|弱分量 Recall|AUPR|FPR|
|---|---:|---:|---:|
|MFT|+0.03878 (4/5)|+0.00639 (5/5)|+0.00032 (3/5)|
|VFT|+0.01726 (4/5)|+0.00895 (4/5)|+0.00419 (3/5)|
|ISRJ|+0.01758 (5/5)|+0.00704 (5/5)|-0.00323 (2/5)|

### development-v12：候选 − C6

|宏 AUPR|宏 F1|目标弱分量 Recall|目标均值 FPR|
|---:|---:|---:|---:|
|+0.00595 (5/5)|+0.01205 (5/5)|+0.01808 (5/5)|+0.00172 (3/5)|

|变体|宏 AUPR|宏 F1|目标弱分量 Recall|目标均值 FPR|
|---|---:|---:|---:|---:|
|C6_Joint|0.86432|0.74802|0.29722|0.19425|
|M5_GradientIsolated|0.86780|0.75623|0.30996|0.19414|
|FullAlpha1p5|0.87027|0.76007|0.31531|0.19597|

|目标标签|C6 weak recall|M5 weak recall|候选 weak recall|候选−C6|候选−M5|
|---|---:|---:|---:|---:|---:|
|MFT|0.35795|0.37965|0.39085|+0.03290 (4/5)|+0.01120 (4/5)|
|VFT|0.27626|0.27644|0.28006|+0.00380 (3/5)|+0.00361 (3/5)|
|ISRJ|0.25746|0.27379|0.27501|+0.01755 (4/5)|+0.00122 (3/5)|

最差的单个 target-weak-recall 配对：seed 2040，VFT，候选−C6 = -0.02847。

|目标标签|弱分量 Recall|AUPR|FPR|
|---|---:|---:|---:|
|MFT|+0.03290 (4/5)|+0.00624 (5/5)|+0.00355 (3/5)|
|VFT|+0.00380 (3/5)|+0.01075 (5/5)|-0.00097 (3/5)|
|ISRJ|+0.01755 (4/5)|+0.00667 (4/5)|+0.00258 (4/5)|

## 预定义因素：已保存的可读代理与证据缺口

|因素|历史中可读的指标|能否验证“至少两个目标标签的条件效应”|判定|
|---|---|---|---|
|SawFM 在场/不在场|保存预测可重算 MFT/VFT/ISRJ 各自的 AUPR、weak recall、FPR|是，见下方 DID|可用于阶段 A 资格判断|
|基数|保存预测可重算目标标签级 AUPR、weak recall、FPR|是，但无功率解释|仅描述性|
|组合/共现|保存预测可重算精确正标签组合的目标弱分量 recall|是，但精确组合内无负样本，AUPR/FPR 不定义|仅描述性|
|相对主/弱功率|未保存|否|缺失|
|SNR/JSR|未保存|否|缺失|
|排序 vs 阈值/校准|全局 AUPR 与决策后 recall/FPR|否：无逐样本分数/标签，无法在因素层分开 ranking 与 threshold/calibration|缺失|
|seed/最差标签|逐 seed 的全局与目标标签指标|仅能描述总体异质性，不能归因于预定义因素|仅假设生成|

### SawFM 组 macro-F1（候选 − C6；不可外推为目标标签效应）

|来源|SawFM 不在场|SawFM 在场|
|---|---:|---:|
|development-v4|+0.00759 (5/5)|+0.01621 (5/5)|
|development-v5|+0.00995 (5/5)|+0.02000 (5/5)|
|development-v6|+0.00757 (5/5)|+0.01828 (5/5)|
|development-v12|+0.00365 (5/5)|+0.02103 (5/5)|

### SawFM 目标标签差异中的差异（候选−C6 的 weak recall：在场 − 不在场）

正值表示候选相对 C6 的弱分量召回增益在 SawFM 在场时更大。此处使用保存的预测和已保存的 weak mask，未调用模型。

|来源|MFT|VFT|ISRJ|
|---|---:|---:|---:|
|development-v4|-0.06128 (0/5)|-0.06208 (0/5)|+0.01470 (4/5)|
|development-v5|-0.10957 (0/5)|-0.05460 (0/5)|+0.00412 (2/5)|
|development-v6|-0.06037 (0/5)|-0.05814 (0/5)|+0.03347 (5/5)|
|development-v12|-0.07736 (0/5)|-0.05144 (0/5)|+0.01261 (3/5)|

### 组合含标签 sample-F1（候选 − C6；组合层面，不是标签弱分量）

|来源|MFT|VFT|ISRJ|
|---|---:|---:|---:|
|development-v4|+0.01008 (5/5)|+0.00778 (5/5)|+0.00837 (5/5)|
|development-v5|+0.01163 (5/5)|+0.01068 (5/5)|+0.01346 (5/5)|
|development-v6|+0.01227 (5/5)|+0.00925 (5/5)|+0.01127 (5/5)|
|development-v12|+0.00960 (5/5)|+0.00724 (5/5)|+0.00934 (5/5)|

## 排序、阈值、功率可见性与频率/组合偏差的分离

- **排序：** 保存的 probabilities 支持按 SawFM/基数的目标标签 AUPR；精确正标签组合内没有负样本，AUPR 不定义。所有细表均保存在结构化审计 JSON。
- **阈值/校准：** weak recall、FPR、F1、组合 sample-F1 均来自保存的阈值化 predictions；保存的 thresholds/calibration 可被核验，但阶段 A 不调参、不重校准。
- **功率可见性：** predictions 中仍无相对主/弱功率、SNR/JSR 或可配对 `receiver_noise_seed` 字段；不得将 SawFM 条件差异解释为功率可见性机制。
- **频率/组合偏差：** 基数与组合层汇总表可显示样本层差异，不能区分组合频率、标签难度和阈值误差，也不能证明目标标签的弱分量机制。

## 停止规则执行

阶段 A 的资格规则要求：**至少一个预定义因素在至少两个目标标签上呈现跨 seed 一致的效应差异**。为避免均值掩盖反向 seed，SawFM 的执行约定为四个历史轮次的 20 个保存 seed 对均同号。实际标签结论见结构化审计的 `ptv_eligibility` 字段。

- 决定：`ELIGIBLE_FOR_TRB_PTV_PREREGISTRATION`；`eligible_for_trb_ptv_preregistration=true`；符合标签：MFT, VFT。
- FC-PTV 的既有永久停止决定不受本审计影响，禁止更改其半径、系数或门槛，也不得复活该分支。
- SawFM 分层即使通过资格，也只支持“条件异质性值得预注册”的下一步，不能被改写成相对功率或 SNR 机制结论。

## 可复现制品

- 程序：`experiments/analyze_route_a_fullalpha_heterogeneity.py`（只读历史 JSON）。
- 结构化审计：`outputs/tapc/route_a_fullalpha_heterogeneity_audit/heterogeneity_audit.json`。
- 本报告与结构化审计记录 input SHA256、来源、角色和停止决定；不存在预测重算或模型输出。
