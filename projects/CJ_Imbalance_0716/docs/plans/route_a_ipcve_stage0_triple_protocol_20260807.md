# Route A IP-CVE Stage 0：同身份 absent/low/high 三元组协议

日期：2026-08-07  
状态：预注册，尚未形成模型证据。  
上游准入：`conditional_admit_stage0_and_p0_only`。

## 1. 唯一目标

验证数据生成器能否产生逐点共享 receiver noise、参考信号、非目标分量及 nuisance 参数，并且唯一改变目标 MFT/VFT/ISRJ 存在性或功率的 `absent/low/high` 三元组。Stage 0 不训练分类器、不拟合 probe、不校准、不选阈值。

## 2. 固定来源

- 项目：`E:\LJQ_code\CJ_Imbalance_0716`
- 生成器：`E:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成`
- development：`E:\LJQ_code\干扰数据\route_a_ipcve_development_cf_v1_20260807`
- smoke base seed：`3990000000`，15 个三元组；
- formal base seed：`4000000000`，2,400 个三元组；
- confirmation 只预留 base seed `4050000000`，本阶段禁止生成。

## 3. 三元组合同

- 目标顺序：MFT、VFT、ISRJ；每目标 800 个 formal 三元组。
- 每目标五种固定 nuisance 组合，每组合 160 个 formal 三元组。
- common JNR：20 dB。
- absent：目标关闭；low：目标 offset -12 dB；high：目标 offset 0 dB。
- 三个视图共享相同 `sample_seed`、`receiver_noise_seed`、reference、背景分量、分量随机参数和接收噪声 realization。
- 保存 `iq_complex`、六标签、79 维参数与 mask、triple id、target、level、nuisance、role、sample/receiver seed 和目标 offset。
- 推理和训练均未在 Stage 0 发生。

## 4. 固定 nuisance

| 目标 | 五类背景 |
|---|---|
| MFT | FMB+SawFM；FMB+VFT；SawFM+ISRJ；VFT+ISRJ；FMB+SawFM+VFT |
| VFT | FMB+SawFM；FMB+MFT；SawFM+ISRJ；MFT+ISRJ；FMB+SawFM+MFT |
| ISRJ | FMB+SawFM；FMB+VFT；SawFM+VFT；MFT+VFT；FMB+SawFM+VFT |

## 5. 角色

按每个基础 identity 的 `receiver_noise_seed mod 10` 分组：

- 0--5：fit，1,440 个 formal 三元组；
- 6--7：early，480 个；
- 8--9：evaluation，480 个。

一个三元组的三个视图不得跨角色。Stage 1 只允许 early 选择一个 frozen feature source，evaluation 执行一次。

## 6. 强制数据门

1. 三视图 receiver/sample seed 完全一致；
2. absent→low 只增加目标标签；low/high 标签完全相同；
3. 非目标参数值和 mask 完全相同；
4. low/high 的目标非 JNR 参数完全相同；
5. `high-absent` 与 `low-absent` 的复数波形差满足预注册幅度比例 `10^(12/20)`，最大相对误差 `<1e-9`；
6. formal 中目标数、nuisance 数和角色数精确满足合同；
7. 新 receiver 范围与 live historical train sources、记录身份和 sealed test 记录零交集；
8. test-v3 seal 保持 false/false/false；
9. smoke 只验证路径，不作为 P0 或模型证据。

任一门失败时只允许修复生成器并重新生成 smoke；formal 一旦 materialize，不得修改合同或覆盖。
