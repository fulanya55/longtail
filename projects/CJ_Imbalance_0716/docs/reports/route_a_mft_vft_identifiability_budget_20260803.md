# Route A：0716 协议下 MFT/VFT 的可辨识性预算与下一代数据条件

日期：2026-08-03  
性质：基于 0716 数据契约与已冻结失败证据的只读物理诊断。没有访问任何 sealed test I/Q，没有新训练或新数据生成。

## 结论先行

当前 MFT/VFT 的主要瓶颈是**观测协议可辨识性不足**，不是仅靠扩大主干、增加注意力或更换分类器就可解决的问题。

- **VFT**：每条记录只有一个 100 us 有效 LFM 脉冲，Doppler 是 1--7 kHz。其观测时长小于或接近分辨这一区间所需的相干时间；这与 H0i 单成分可辨、H1j 复合归因不稳定相吻合。
- **MFT**：3--8 个副本的相对延迟只在 30--50 us（20 us 跨度）内，远小于 100 us 脉宽。所有副本高度重叠；在组件总 JNR 固定时，提高副本数还会降低典型单副本能量。这与 H1k/H1m/H0o/H0p 的“局部物理干预可测、复合归因未过门”一致。

因此，在不改变 0716 观测协议的前提下，MFT/VFT 不再启动网络宽度、STFT 分辨率、时频形态、匹配滤波统计或延迟统计变体搜索。当前可可靠推进的是窄 ISRJ aligned-periodicity 证据链。

## 1. 数据契约中的观测预算

0716 数据说明给出：采样率 `fs=10 MHz`、脉宽 `T=100 us`、PRI `300 us`、每条记录脉冲数为 `1`、I/Q 长度为 `3000`，LFM 带宽 5 MHz。

### 1.1 VFT：单脉冲 Doppler 预算

VFT Doppler 范围为 `1--7 kHz`。对单个有效脉冲：

\[
\Delta f_{\mathrm{Rayleigh}} \approx 1/T = 10\ \mathrm{kHz},
\qquad
f_D T \in [0.1,0.7]\ \mathrm{cycles},
\qquad
2\pi f_DT \in [0.628,4.398]\ \mathrm{rad}.
\]

因此 6 kHz 的完整参数跨度仅为单脉冲 Fourier 分辨率的 `0.6`，低端 1 kHz 在有效脉冲内仅积累 `0.1` 个周期。已知 LFM 参考可以在 H0i 的单成分配对干预中显露方向性，但在复合背景下，其他成分、到达时延和噪声会占用同一有限相干孔径；H1j 对固定分数的 identity bootstrap CI 跨 0 正是该风险的经验表现。

这不等于“任何 VFT 识别都不可能”，而是表明：**不能把在一个 100 us 单脉冲内对 1--7 kHz 的细致 Doppler 可辨性当作当前协议下可靠、可泛化的标签内物理依据。**

### 1.2 MFT：重叠和单副本能量预算

MFT 的假目标数为 `N=3--8`，相对目标时延为 `30--50 us`，LFM 脉宽仍为 `100 us`。任意两个副本的相对延迟差不超过 `20 us`，所以在同一 100 us 脉冲支撑内，其重叠时长至少为：

\[
T_{\mathrm{overlap}} \ge 100\ \mathrm{us} - 20\ \mathrm{us}=80\ \mathrm{us},
\]

即至少 `80%` 的时间支撑重叠。由于真实相对幅度还在 `0.5--1.5` 随机变化，这些副本在复合条件下表现为相干叠加的混合 LFM，而非一组容易逐条隔离的目标。

若仅用等能、非相干近似说明固定组件总功率下的典型趋势，则单副本相对总功率约降低 `10log10(N)`：`N=3` 为 `4.77 dB`，`N=8` 为 `9.03 dB`。实际波形会因相干相加和幅度随机化偏离这一近似，因此它不是每条样本的精确 JNR 声明；但说明了为何“更多副本”不自动等于“更多可见局部峰”。

## 2. 与冻结实验的对应关系

| 实验 | 物理问题 | 结果如何支持诊断 |
| --- | --- | --- |
| H0i -> H1j（VFT） | 单脉冲内的去斜相位残余 | H0i 单成分干预有效；H1j 复合源固定分数的 weak AUROC/AUPR identity CI 均跨 0，说明有限孔径线索不能稳定归因。 |
| H1k / H1m（MFT） | 重叠副本的 delay support / 稀疏原子归因 | 复合匹配负样本上弱 MFT 排序反向，表明 global delay 结构主要反映混合背景而非 MFT 标签。 |
| H0o（MFT） | 四 PRI 的跨脉冲相干 | 相对幅度控制存在信息差，但绝对 AUROC `0.58066 < 0.60`，未达可推广的弱 MFT 归因门。 |
| H0p（MFT） | 单 PRI、无参考的延迟相关复杂度 | 3-vs-8 干预敏感，但相位置换控制同样敏感；candidate-control `+0.03316 < +0.10`，说明不是 MFT 特异信息。 |

## 3. 对当前网络设计的直接含义

1. 对 VFT，更多 pooling token、标签查询或更细 STFT 不会增加真实相干观测时长；它们只能重新分配现有、低分辨率的证据。
2. 对 MFT，注意力若从单一混合波形中选择局部显著点，极易服务于最强的相干叠加或共存干扰，而非低能量的单副本属性。
3. 所以“特征学习不足”的更准确表述是：网络被要求从物理上高度混叠且标签归因不足的观测中恢复 MFT/VFT 局部证据。没有先验可辨识性门通过时，端到端增容只会增加选择自由度和证据泄漏风险。

## 4. 新一代协议的最小条件（新研究问题，不与 0716 test-v3 混用）

以下建议仅用于未来建立**独立的新 development/test 协议**，不是对现有 0716 进行调参或重写结论。

### VFT：增加相干孔径

- 生成至少 `K=8` 个相位连续、同 Doppler 的 PRI 脉冲；300 us PRI 下首末相干基线约 `2.1 ms`，Fourier 尺度约 `476 Hz`，1--7 kHz 将积累约 `2.1--14.7` 个周期；
- 对照必须破坏跨 PRI 的 Doppler 相干而保留每脉冲能量、LFM 形状、噪声和标签上下文；
- 先做只读/零拟合 H0 的低/高 Doppler 成对可辨性，再做复合匹配负样本 H1；不能直接将多 PRI 网络接入六标签模型。

### MFT：增加可归因的时延孔径

- 扩展相对延迟 aperture 至至少一个脉宽，并记录最小副本间隔；若要主张逐副本结构，应让指定副本在部分观测区间内不完全重叠，或使用多 PRI 重复结构；
- 保持总组件 JNR、每副本振幅规则及负样本组合上下文明确；同时报告总 JNR 与每副本有效能量，避免“更多副本”实际只是更高总功率；
- 对照应破坏副本的真实时延结构，同时匹配能量、时域占用、组件数和计算预算；不能只使用随机标签或较小网络作控制。

### 共同的证据边界

新协议必须使用全新 receiver-noise identities、独立 development 和 sealed test；不得读取、重复运行或用于选择 `test-v3`。即使新协议成功，它支持的是**新的多脉冲/扩展时延观测问题**，不能倒灌改写单脉冲 0716 的 locked-test 结论。

### 4.1 均匀多 PRI 的附加否决条件

上面的 `K=8` 只是相干孔径的必要条件，不足以直接形成可执行方案。官方生成器的 VFT 实现对完整脉冲串乘 `exp(j2*pi*fD*t)`，因此其跨 PRI 相位物理上连续；但如果仍使用固定 `PRI=300 us`，慢时间 PRF 仅为：

\[
f_{\mathrm{PRF}}=1/PRI=3.333\ \mathrm{kHz},
\qquad |f_D|_{\mathrm{unambiguous}}\leq f_{\mathrm{PRF}}/2=1.667\ \mathrm{kHz}.
\]

原 1--7 kHz VFT 范围包含相差一个或多个 PRF 的别名，例如 `1.0 kHz` 和 `4.333 kHz` 具有相同的逐脉冲相位递推。脉内 100 us 相位仍可能提供弱的消歧信息，但这正是 H0i/H1j 已显示不稳定的有限孔径信息，不能把它当作“多 PRI 已解决”的证据。

因此，**固定 300 us 的均匀多 PRI VFT-H0 不准入**。如需新协议，至少要采用经物理验证的交错 PRI（并记录每个实际脉冲时间戳），或重新限定无混叠 Doppler 区间；两者都属于新数据问题，必须先有独立预注册与全文文献冲突审计。

## 5. 当前决策

- 0716：继续保留 RawSaw 与冻结 ISRJ aligned-periodicity 的可审计窄结论；
- MFT/VFT：保持无新候选状态，等待具备全文审计支持且数学上非近邻的观测量，或另行启动独立的新观测协议；
- `test-v3`：继续 sealed，未被读取或授权。

## 6. 可复算元数据审计

已执行：

`D:\anaconda3\envs\PY10\python.exe experiments/audit_route_a_mft_vft_identifiability_budget.py`

它只打开 `C:\LJQ_code\干扰数据\复合干扰数据0716\closedset_6class_train.mat` 的 HDF5 配置节点，不扫描或打开任何 test 文件。输出：

`outputs/tapc/route_a_mft_vft_identifiability_budget/metadata_budget.json`

其中记录的官方 train 文件大小为 `2,549,129,808` bytes；为避免不必要地顺序扫描 2.5 GB payload，完整 payload SHA-256 在本次元数据审计中设为 `null`，而不是伪造哈希。运行验证如下：

- `single_pulse=true`；
- `vft_span_below_single_pulse_fourier_scale=true`；
- `mft_delay_span_below_pulse_width=true`；
- `mft_overlap_at_least_0p80=true`；
- `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

测试命令：

`D:\anaconda3\envs\PY10\python.exe -m pytest -q tests/test_route_a_mft_vft_identifiability_budget.py tests/test_route_a_isrj_evidence_bundle.py`

结果：`4 passed`。
