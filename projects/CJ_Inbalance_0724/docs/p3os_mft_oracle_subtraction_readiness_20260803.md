# P3o-S：MFT oracle 扣除因果诊断——冻结前准备

## 已冻结的问题

在新身份、同一冻结主分类器上，精确且按生成器实际功率标定的 MFT 分量是否是 `MFT=1, ISRJ=0` 子集出现 ISRJ 假阳性的因果来源？这不是可部署模型：C1 使用仅评估时可见的 MFT 真值。

## 三个条件

| 条件 | 输入给同一冻结模型前的处理 |
|---|---|
| C0 | 原始复合 I/Q，再作既有逐视图 RMS 归一化 |
| C1 | 扣除由 `delay_01..08`、`amplitude_01..08` 重建、并按 `noise_power + MFT.JNR` 标定的 exact MFT 分量，再作同一归一化 |
| C2 | 扣除 C1 分量的确定性 FFT 相位扰动版本；逐样本能量严格相同，但不保留 MFT 时延结构 |

冻结分类器为原始 `mlcnn_iq_grouped_task_routed` 参数最优 checkpoint（2041/2042/2043），使用 P2c 已冻结 `safe_thresholds`。不重训、不调阈值、不调用路由器或参数头。

## 已完成的验证

- P3o-S Python 专项单元测试 3 项通过；同时回归 P3o-R 专项测试，合计 `6 passed`。
- MATLAB `executeGeneration=false` 预检通过，`checkcode` 未报出诊断；未生成任何视图。
- 生成器继续采用每一个“场景×功率”单独 MATLAB 进程、原子 `resume` 事务和可写性/5 GiB 容量预检，因此不回退到已修复前的长生命周期并发写入模式。
- 预注册、Python 实现和 MATLAB 生成器的 SHA-256 已记录于 [P3o-S method freeze](../configs/p3os_mft_oracle_subtraction_method_freeze_20260803.json)。

## 尚未执行及其原因

正式开发数据为 seed `1920000000/1930000000`、五场景、25/35/50 dB、每视图 2079 样本，共 30 个视图。D: 当前约 7.34 GiB 空间，小于约 8 GiB 的原始数据加重分块缓存需求；C: 约 373.6 GiB 可用，但更换修复后默认 D: 输出根属于需要用户确认的存储决策。

在得到该选择前，禁止运行 `executeGeneration=true`、禁止访问任何 locked test，也不把 P3o-R 已打开身份充作 P3o-S 的新身份数据。
