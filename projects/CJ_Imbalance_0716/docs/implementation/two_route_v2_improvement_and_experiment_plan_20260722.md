# 两条路线 V2 模型改进策略与下一步实验计划

版本：V2.0  
日期：2026-07-22  
环境：`D:\anaconda3\envs\PY10\python.exe`  
数据：`C:\LJQ_code\干扰数据\复合干扰数据0716`  
locked test：禁止读取

## 1. 总体原则

本轮只验证上一轮已经定位出的根因，不扩展到六类、完整参数分支或 locked test。两条路线均采用单 seed、小规模开发集机制筛选，并使用 macro-AUPR 选择 checkpoint。

## 2. 路线 A：TAPC 功率约束路线

### 2.1 改进结构

保留 T0P 的共享 backbone、共享分类器和两个功率视图分类。将旧双向排序

\[
\max(0,m-(s_{strong}-s_{weak}))
\]

改为单侧排序：

\[
L_{ord}^{strong}=\max(0,m-[s_{strong}-\operatorname{sg}(s_{weak})])
\]

`sg` 表示 stop-gradient。排序损失只能提高强功率 logit，不能压低弱功率 logit。embedding distillation 保留原来的 low-to-high 方向，并与 ordering 分开消融。

### 2.2 消融矩阵

| 变体 | 双视图分类 | 单侧 ordering | embedding distillation |
|---|---:|---:|---:|
| T0P | 是 | 否 | 否 |
| T2O_V2 | 是 | 是 | 否 |
| T2D | 是 | 否 | 是 |
| T2OD_V2 | 是 | 是 | 是 |

### 2.3 冻结协议

| 项目 | 设置 |
|---|---|
| protocol | Power-LT |
| seed | 2026 |
| train / eval | 512 / 128 |
| epochs / patience | 8 / 8 |
| batch size | 64 |
| 功率等级 | 25、30、35 dB |
| reference | 30 dB |
| power weight | 0.20 |
| ordering margin | 0.20 |
| checkpoint | macro-AUPR |

保留门槛：相对 T0P，macro-AUPR 下降不超过 0.01、weak recall 不下降，同时 directional margin mean 与 satisfaction rate 均提高。

执行入口：

```powershell
D:\anaconda3\envs\PY10\python.exe experiments\run_route_a_v2_screen.py
```

## 3. 路线 B：DAPS 半监督路线

### 3.1 改进结构

本轮不启用上一轮失效的 prototype gate、分布对齐和参数头，只建立稳定的 Teacher 伪标签主链路。

```text
labeled signal ─ student ─ supervised CB-ASL
                     │
                     └─ EMA update ─ teacher

unlabeled weak view ─ teacher ─ confidence + two-view stability ─ pseudo label
unlabeled strong view ─ student ─ pseudo classification / feature consistency
```

训练机制修改：

1. 每个 epoch 固定为完整遍历 labeled loader，B0 与半监督变体拥有相同监督曝光量；unlabeled loader 循环取样。
2. 前 2 epochs 只做监督学习和 EMA 更新，不生成伪标签。
3. 第 3 epoch 起启用伪标签，起始阈值由过严的 0.95/0.05 调整为机制筛选用的 0.80/0.20。
4. 保留双弱视图稳定性门，暂不使用不可靠的 prototype reliability。
5. 使用 macro-AUPR 选择 checkpoint，并记录每轮 labeled/unlabeled 曝光量、正负伪标签数和 coverage。

### 3.2 消融矩阵

| 变体 | EMA Teacher | pseudo classification | feature consistency | dynamic threshold |
|---|---:|---:|---:|---:|
| B0 | 否 | 否 | 否 | 否 |
| B1 | 是 | 是 | 否 | 否 |
| B2 | 是 | 是 | 是 | 否 |
| B3 | 是 | 是 | 是 | 是 |

### 3.3 冻结协议

| 项目 | 设置 |
|---|---|
| labeled ratio | 20% nested split |
| seed | 2026 |
| labeled / unlabeled cap | 512 / 512 |
| eval cap | 128 |
| epochs / patience | 8 / 8 |
| batch size | 64 |
| Teacher warm-up | 2 epochs |
| tau positive / negative | 0.80 / 0.20 |
| stability ceiling | 0.10 |
| threshold update | 每 epoch |
| checkpoint | macro-AUPR |

保留门槛：半监督变体必须出现非零但非全覆盖的伪标签 epoch；相对 B0，macro-AUPR 不下降超过 0.01；若 B3 再次出现 coverage 全开或全关振荡，则停止动态阈值路线并改做基于 calibration 分位数的控制器。

执行入口：

```powershell
D:\anaconda3\envs\PY10\python.exe experiments\run_route_b_v2_screen.py
```

## 4. 执行边界

- 本轮只进行开发阶段机制筛选，不报告最终论文性能。
- 不读取 locked test。
- 路线 A 暂不恢复 prototype、stability 和参数头。
- 路线 B 暂不启用 prototype gate、distribution alignment 和参数头。
- 只有通过本文件门槛后，才扩大样本、增加 seed 或进入完整模型实验。

## 5. 执行结果索引

本计划已经执行完成，结果与下一轮判断见：

`docs/results/two_route_v2_screen_results_20260722.md`
