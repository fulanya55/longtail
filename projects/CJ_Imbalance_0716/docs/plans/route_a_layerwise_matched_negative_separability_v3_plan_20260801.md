# Route A 分层组合匹配负样本可分性审计 development-v3 预注册

## 1. 目的与假设

连续三类整改均未达到门槛：跨功率一致性、推理期可见度调制、final-temporal 三层等权。共同可能根因是用于分类的压缩表示缺少弱正相对真实负样本的局部判别证据。该轮不提出新模型，只回答：F2 的 stage2、stage3、final temporal 或 global 哪一层仍保留可泛化的弱正局部证据？

## 2. 冻结表征

- G Global：128 维全局特征；
- T FinalTemporal：128 通道的 mean/std/max/top-k，512 维；
- S2 Physics：stage2 复数幅度、幅度差分、相位增量及变化的多统计摘要；
- S3 Physics：stage3 使用同样物理摘要；
- 主干与 C6 checkpoint 完全冻结，只训练每层每标签一个线性探针；
- 本轮是表示充分性诊断，参数量差异必须报告，不能作为最终方法比较。

## 3. 正负定义

- 正样本：目标标签为 1 且相对最强共现分量功率差 `<=-10 dB`；
- 负样本：目标标签为 0，且其他五标签组合与至少一个弱正样本完全相同；
- 强正样本不参与探针训练或评价；
- 匹配在 fit、validation、early 各角色内部独立执行，不跨角色配对；
- 每角色保存逐标签弱正数、匹配负数和组合数。

## 4. 探针协议

- 输入标准化仅使用 fit；
- 每层每标签独立 `Linear(dim,1)`，pos-weight BCE；AdamW `lr=1e-3`、weight decay `1e-4`、最多 80 epoch、patience 10；
- validation AUROC 早停，early 只报告；
- 五种子 2039--2043；保存 checkpoint、history、标准化参数、样本计数和 SHA-256；
- 禁止访问 locked test、test-v2、sealed test-v3。

## 5. 晋级门槛

某一候选层相对 T FinalTemporal 必须满足：early 三标签平均 AUROC `>=+0.030`、绝对 AUROC `>=0.620`、至少 4/5 种子增益为正、每个目标标签平均 AUROC `>=0.580`、validation 到 early 的平均落差 `<0.150`。只有通过者才允许进入局部专家训练。

若没有任何层通过，则冻结 F2 主干已不能充分分离弱正与组合匹配负样本；下一步应验证独立原始 I/Q 多尺度专家或小学习率解冻，而不能继续在冻结全局特征上更换损失、分类头或阈值。
