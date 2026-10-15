# 六类主网与扫频专家有界优势路由实验报告

日期：2026-07-26  
证据等级：root-train内部开发证据，tune-only  
locked test：未使用  
confirm：未使用

## 1. 试验问题

验证冻结的六类MLCNN/IQ-Grouped模型与冻结的U-Net、VFT瞬时频率轨迹专家、ISRJ周期专家能否通过有界优势路由，转化为九档identity-safe条件下的六类识别和端到端参数门控收益。

预注册晋级门槛：

1. 10 dB与50 dB的SawFM、VFT、ISRJ平均Recall相对MLCNN提高至少0.10；
2. 六类Macro-AUPR下降不超过0.005；
3. 三个目标类别总体FPR均不超过0.10；
4. 3/3 seeds的端点Recall增益同方向；
5. identity-paired bootstrap的Recall增益95% CI大于0。

## 2. 冻结模型与路由结构

- 六类主网：`mlcnn_iq_grouped_task_routed`的`best_parameter.pt`，种子2041、2042、2043。
- SawFM专家：冻结U-Net的SawFM存在性概率。
- VFT专家：既有入选配置`stack_trajectory_leaf15`。
- ISRJ专家：既有入选配置`stack_all_periodicity`。
- 专家模型此前未完整持久化，因此本次仅按已冻结配置和原角色确定性重建，没有候选搜索。
- 路由：`StandardScaler + LogisticRegression(C=1, class_weight=balanced)`三状态优势预测。
- 三状态为positive rescue、negative protection和unknown。
- 路由输出严格限制在MLCNN概率和专家概率之间，不允许无界残差。
- 路由输入只包含主网、专家和U-Net概率及其差值/不确定性，不包含显式功率或JNR。

路由公式：

```text
p_route = p_base
        + positive_action * max(p_expert - p_base, 0)
        - negative_action * max(p_base - p_expert, 0)
```

只有动作概率不小于0.5时才启用修正，最终结果裁剪在`[min(p_base,p_expert), max(p_base,p_expert)]`。

## 3. 数据与角色

使用`power_sweep_9level_v1`的10、15、20、25、30、35、40、45、50 dB九档同身份视图。

| 角色 | 身份数 | 用途 |
|---|---:|---|
| train | 18,900 | 仅在10 dB确定性重建已冻结VFT/ISRJ底层专家 |
| early | 2,520 | 10 dB重建既有stacker；九档拟合优势路由 |
| calibration | 1,260 | 九档FPR约束阈值校准 |
| tune | 1,260 | 最终开发评价 |

九档视图始终跟随同一identity，bootstrap按identity重采样并携带全部九档视图，未将九个视图作为独立样本。

## 4. 主要结果

### 4.1 总体结果

| 指标 | MLCNN | 硬替换 | 有界路由 |
|---|---:|---:|---:|
| 六类Macro-AUPR | 0.966865 | 0.882514 | **0.969986** |
| 相对MLCNN的Macro-AUPR变化 | — | -0.084351 | **+0.003121** |
| 参数门控覆盖率 | 0.904466 | 0.729099 | 0.905951 |
| 预测门控参数NMAE | 1.145683 | 1.156106 | **1.134200** |
| miss-penalty NMAE | 1.128472 | **1.094459** | 1.119686 |
| 假阳性参数门控率 | 0.103734 | 0.096555 | **0.100597** |
| exact label gate accuracy | 0.566490 | 0.284891 | 0.564080 |

有界路由的六类Macro-AUPR增益bootstrap 95% CI为`[+0.002438,+0.003718]`。

有界路由的miss-penalty NMAE变化为`-0.008786`，bootstrap 95% CI为`[-0.009665,-0.007852]`，相对改善约0.78%。参数覆盖率仅提高0.00149，CI为`[-0.000482,+0.003304]`，尚不能确认覆盖率真实提升。

### 4.2 端点Recall

| 功率 | 方法 | SawFM Recall | VFT Recall | ISRJ Recall |
|---:|---|---:|---:|---:|
| 10 dB | MLCNN | 0.98594 | 0.23073 | 0.32448 |
| 10 dB | 有界路由 | 0.92760 | **0.37240** | **0.46094** |
| 50 dB | MLCNN | 0.48646 | 0.99115 | 0.98906 |
| 50 dB | 有界路由 | 0.40990 | 0.99375 | 0.99115 |

端点六个“功率×类别”Recall的平均增益为`+0.02465`，3/3 seeds正向；identity bootstrap 95% CI为`[+0.01739,+0.03049]`。

该增益主要来自10 dB的VFT和ISRJ：

- 10 dB VFT Recall：`+0.14167`；
- 10 dB ISRJ Recall：`+0.13646`；
- 10 dB SawFM Recall：`-0.05833`；
- 50 dB SawFM Recall：`-0.07656`；
- 50 dB VFT/ISRJ已接近饱和，增益仅约0.002。

### 4.3 FPR与功率漂移

有界路由九档总体目标FPR：

- SawFM：`0.10938`，超过0.10；
- VFT：`0.09701`；
- ISRJ：`0.09994`。

尽管总体VFT/ISRJ FPR接近上限，部分中间功率发生明显超限：

- 20 dB SawFM FPR：0.18763；
- 20 dB VFT FPR：0.15161；
- 20 dB ISRJ FPR：0.26935；
- 25 dB ISRJ FPR：0.23817。

说明在九档合并calibration上控制总体FPR，不能保证每个功率条件都安全。

## 5. 晋级判定

| 门槛 | 结果 | 判定 |
|---|---:|---|
| 10/50 dB三类平均Recall增益≥0.10 | +0.02465 | 失败 |
| 六类Macro-AUPR损失≤0.005 | 实际提高0.00312 | 通过 |
| 三类总体FPR≤0.10 | SawFM 0.10938 | 失败 |
| 3/3 seeds Recall同方向 | 3/3 | 通过 |
| bootstrap Recall CI下界>0 | +0.01739 | 通过 |

**最终判定：不晋级，不接入最终六类系统。**

这是一个统计显著但实际收益不足的负结果。不能因为AUPR和CI为正就忽略预注册的Recall幅度和FPR门槛。

## 6. 成果与失败机制

### 已验证成果

1. 有界优势路由能够避免专家硬替换造成的灾难性退化。硬替换使Macro-AUPR下降0.08435，有界路由反而提高0.00312。
2. VFT瞬时频率轨迹和ISRJ周期专家在10–15 dB确有可迁移证据。
3. 路由对miss-penalty NMAE产生小但稳定的改善，说明分类修正能够部分转化为端到端参数门控收益。
4. 整个身份安全九档执行链、逐样本产物和bootstrap流程已经建立并通过测试。

### 失败原因

1. **SawFM专家方向错误。** U-Net SawFM概率在10 dB和50 dB都降低主网Recall。路由器对SawFM的positive-rescue触发率低、negative-protection触发率较高，持续压低正类概率。
2. **VFT/ISRJ专家只在低功率真正互补。** 10–15 dB增益显著，25 dB以后主网已较强，专家出现分布漂移，不能继续提供稳定优势。
3. **路由目标过于宽松。** 当前动作标签只依据专家概率方向是否比主网更符合标签，没有要求改善幅度，也没有直接惩罚阈值附近的FPR风险。
4. **合并功率校准掩盖局部风险。** 九档总体FPR接近0.10，但20–25 dB局部FPR明显超限。
5. **参数覆盖率改善太小。** miss-penalty有稳定改善，但覆盖率CI跨零，尚不能支持“弱分量参数可用性明显提高”的论文主张。

## 7. 下一步决策

本轮方法与结果冻结，停止当前三标签路由版本。下一轮如果继续，应只做一个受控修复：

> 移除SawFM的U-Net存在性路由，仅保留VFT/ISRJ；将优势动作标签改为带最小收益margin和FPR代价的loss-aware三状态标签，并采用可见性分层校准，而不是九档合并阈值。

下一轮仍不得访问confirm或locked test。建议门槛保持不变，并额外要求每个功率档的VFT/ISRJ FPR均不超过0.10。若该受控修复仍不能把端点平均Recall增益提高到0.10，则停止六类扫频专家集成路线，仅将U-Net/VFT/ISRJ作为机制分析或低功率专用子系统报告。

## 8. 产物与验证

- 实现：`cj0724/integrate_swept_experts_with_six_class.py`
- 单元测试：`tests/test_integrate_swept_experts_with_six_class.py`
- 正式结果：`outputs/six_class_swept_bounded_router_20260726/results.json`
- 九档逐功率指标：`outputs/six_class_swept_bounded_router_20260726/per_power.csv`
- 每种子逐样本预测：`seed_2041/2042/2043/tune_predictions.npz`
- 每种子冻结专家、优势路由、checkpoint哈希和阈值均已保存。
- 全测试：`74 passed`。
- 正式运行耗时：268.91秒。
