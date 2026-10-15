# TAPC-CJNet 下一阶段网络结构与 Phase-A 功率机制筛选技术文档

版本：V2.1  
日期：2026-07-22  
状态：执行协议修订后冻结  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
locked test：禁止读取

## 1. 本阶段目标

本阶段不验证完整 TAPC-CJNet，也不评价参数分支和决策接口。唯一目标是回答：

> 在使用完全相同身份配对功率视图的条件下，带物理方向的功率约束是否优于仅增加功率视图的 CB-ASL 基线？

上轮 T5 smoke 暴露出阈值样本过少、prototype 高度共线和参数分支未收敛，因此本阶段排除 prototype、参数头和 decision utility，避免多个辅助任务互相混淆。

V2.1修订：首次V2.0执行发现训练器仍按subset accuracy选择checkpoint，与本文件规定的macro-AUPR优先级不一致。V2.0结果完整保留为失败审计；V2.1将checkpoint选择指标显式冻结为`macro_aupr`，并使用新的输出目录重跑，禁止覆盖V2.0结果。

## 2. 网络结构 V2

```text
reference I/Q ─┐
               ├─ shared complex CNN backbone ─ shared classifier ─ reference logits
paired I/Q ────┘                                  └─────────────── paired logits
        │
        └─ per-label independent projection heads ─ z_reference,c / z_paired,c

T0 : reference classification only
T0P: reference + paired classification only
T2 : T0P + low-to-high embedding distillation + directional logit ordering
T2S: T2 + unchanged-component probability stability
```

### 2.1 公平配对基线 T0P

T0P和T2使用相同的：

- optimization样本ID；
- 功率等级；
- `receiver_noise_seed`身份映射；
- pair SHA-256；
- 两个视图上的分类损失；
- backbone、初始化、batch size和训练轮数。

唯一差异是T0P的功率辅助损失权重为0。这样可以区分“增加功率视图”的收益与“方向约束”的收益。

### 2.2 标签条件表示 V2

旧结构使用一个共享投影向量乘以接近1的标签门控，L2归一化后不同标签方向容易接近，历史最大类间prototype余弦达到0.99以上。

V2为每个标签设置独立的轻量投影：

\[
z_{ic}=\operatorname{norm}(W_c h_i+b_c)
\]

四类只增加四个小型投影头。Student-B仍使用原有gated结构，本次修改不改变半监督路线。

## 3. 损失函数

### T0

\[
L_{T0}=L_{CB-ASL}(x^{ref})
\]

### T0P

\[
L_{T0P}=\frac{1}{2}\left[
L_{CB-ASL}(x^{ref})+L_{CB-ASL}(x^{pair})
\right]
\]

### T2

\[
L_{T2}=L_{T0P}+\lambda_o(L_{distill}+L_{order})
\]

其中：

\[
L_{order}=\max(0,m-(s_c^{strong}-s_c^{weak}))
\]

\[
L_{distill}=1-\cos(z_c^{weak},\operatorname{sg}(z_c^{strong}))
\]

### T2S

\[
L_{T2S}=L_{T2}+\lambda_s
\max(0,|p_c^{ref}-p_c^{pair}|-\epsilon)
\]

固定参数：

| 参数 | 数值 |
|---|---:|
| power weight | 0.20 |
| stability weight | 0.05 |
| ordering margin | 0.20 |
| stability tolerance | 0.25 |
| warmup | 10% |
| ramp end | 30% |

配对视图的分类损失从第一个epoch开始使用；curriculum只控制辅助功率约束，不再控制配对分类数据是否可见。

## 4. Phase-A 筛选协议

| 项目 | 设置 |
|---|---|
| 类别 | FMB、SinFM、MFT、ISRJ |
| protocol | Power-LT |
| optimization样本 | 512 |
| early-stop | 128 |
| calibration | 128 |
| threshold-selection | 128 |
| seed | 2026 |
| epochs | 8 |
| patience | 8 |
| batch size | 64 |
| backbone width | 16 |
| loss | CB-ASL |
| 功率等级 | 25、30、35 dB |
| reference level | 30 dB |
| conditioning | independent |
| checkpoint selection | macro-AUPR |
| AMP | 开启 |

选择25/30/35 dB是为了让±5 dB配对从训练初期即可进入功率curriculum。本阶段是机制筛选，不覆盖正式的±10/±15/±20 dB弱成分矩阵。

## 5. 指标与诊断

本阶段模型比较优先使用不依赖小样本阈值的指标：

1. macro-AUPR；
2. 固定0.5阈值macro-F1；
3. fixed-0.5 weak-component recall；
4. fixed-0.5 tail-label recall；
5. Hamming loss和cardinality MAE。

逐epoch额外记录：

- directional margin mean和p10；
- margin达到0.20的比例；
- unchanged-component drift mean、p50、p90、p95；
- power ordering/distillation/stability loss；
- prototype最大类间余弦，作为后续分支的结构健康度监控。

逐标签选择阈值的结果只作为辅助诊断，不用于本阶段的主要机制结论。

## 6. 判断规则

### 6.1 功率视图效应

比较T0P与T0。如果T0P明显优于T0，说明功率视图本身构成有效增强；后续所有方向机制必须以T0P作为主基线。

### 6.2 方向排序效应

T2相对T0P应同时满足：

- directional margin mean提高；
- directional satisfaction rate提高；
- macro-AUPR不下降超过0.01；
- weak recall和MFT recall不出现明显恶化。

本阶段不要求达到论文正式的+2/+5个百分点门槛，只判断机制是否值得进入更大规模实验。

### 6.3 稳定性效应

T2S相对T2应满足：

- unchanged drift p90下降；
- macro-AUPR下降不超过0.01；
- weak recall不下降。

若stability loss持续为0，则先依据实际drift分布重新评估容差，不直接宣称稳定性约束有效。

## 7. 执行入口和结果目录

执行命令：

```powershell
D:\anaconda3\envs\PY10\python.exe experiments\run_phase_a_screen.py
```

输出目录：

```text
outputs/tapc/phase_a_screen_v2_1_macro_aupr/seed_2026/
    T0/
    T0P/
    T2/
    T2S/
    phase_a_summary.csv
    phase_a_summary.json
```

每个变体保留完整科学配置、环境、事件流、清单哈希、训练历史、固定阈值和选择阈值指标、预测及成功/失败receipt。

## 8. 后续边界

只有Phase-A表明T2相对T0P具有正向机制证据后，才进入1024样本/300评估/20 epochs的Phase-B。只有独立标签投影在正式训练后仍避免0.99级共线，并且T1不造成负迁移，才重新引入prototype。参数头、决策接口、六类矩阵和locked test均不属于本阶段。

## 9. 执行结果索引

V2.1 已执行完成。正式结果、机制判断和下一步 T2v2 单侧方向约束方案见：

`docs/results/tapc_phase_a_power_mechanism_v2_1_20260722.md`

执行结论为：T0P 进入下一轮作为主基线；T2 的物理方向指标改善，但 macro-AUPR 和弱成分召回未过门槛；T2S 的稳定性损失未激活。因此当前 T2/T2S 不直接进入 Phase-B。
