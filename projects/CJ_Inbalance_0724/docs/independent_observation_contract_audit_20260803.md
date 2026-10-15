# 独立观测契约审计：当前生成器与 V10 启动条件

## 结论

当前 MATLAB 生成器实现的是**单接收通道、单 PRI、单次接收机噪声实现**的数据
契约。它不含第二接收通道、第二次独立测量、双斜率发射参考或部署可用的外部时延
参考。因此，当前仓库不能把已有 `iq_complex` 的切片、STFT、FFT、ACF、功率配对视图、
component seed 或分离器输出称为“独立观测”。

这是一项代码/数据合同审计，不是新的模型实验；没有生成新数据、没有训练、没有读取
test 或 locked test。当前 V10 状态保持为 `contract_required`。

## 已核对的实现证据

| 合同层 | 当前实现 | 对 V10 的含义 |
| --- | --- | --- |
| 记录长度 | `create_closedset_6class_config.m` 固定 `num_pulses=1`；说明每条样本只保存一个 PRI | 现有记录不提供受控的跨 PRI 重复测量。 |
| 观测形成 | `compose_jamming_sample.m` 先叠加全部干扰/可选目标，再由一个 `receiverNoiseSeed` 生成一次复高斯噪声，并输出一个 `sample.iq_complex` | 同一身份目前只有一个接收观测；从该向量变换出的特征不是新测量。 |
| 保存格式 | `generate_dataset_from_manifest.m` 保存 `bundle.iq_complex`；`data_legacy` 严格为 `[real(iq_complex), imag(iq_complex)]` | 数据文件没有多通道张量或第二观测字段。 |
| 校验格式 | `validate_generated_dataset.m` 只要求单个 `receiver_noise_seed` 和单个 `iq_complex`，并检查 legacy 与该单观测全等 | 不能在不修改并冻结数据契约的情况下偷偷加入第二通道。 |
| 功率视图 | 历史协议以同一 `receiver_noise_seed` 成组产生多功率反事实 | 它们不能被当作同一时刻、部署可用的独立传感器输入，也不能按独立样本统计。 |

## 允许由用户/采集系统明确授权的最小契约

下面是可审计的候选，不是已获证明的方法，也不授权在缺少实际合同的情况下合成代理数据。

| 代号 | 真实推理输入 | 与原 I/Q 的物理差异 | 必填字段 |
| --- | --- | --- | --- |
| `M1_dual_receiver` | 同步双接收机的两条复 I/Q 记录 | 独立前端噪声/通道响应，而不是同一向量重排 | 两通道采样同步误差、各自噪声模型、共享和非共享传播项、部署硬件成本。 |
| `M2_controlled_repeat_pri` | 同一物理状态下两个受控 PRI 接收记录 | 真正重复接收而非对原记录切窗 | 物理状态保持范围、PRI 间变化模型、两次噪声独立性、在线缓存/延迟。 |
| `M3_dual_slope_probe` | 两种已知 LFM 斜率下的回波/干扰观测 | 激励本身改变，使 delay-Doppler 耦合具有新的可验证约束 | 两种发射波形、带宽/时长/功率、交替时序、干扰对两斜率的响应合同。 |
| `M4_external_delay_reference` | 接收机外部可测且有误差界的时延参考 | 外部传感器或同步协议，不是 generator true parameter | 来源、精度、失效模式、延迟、推理时可用性和误差校准方法。 |

`M1` 和 `M2` 即使物理可行，也只能先接受 V10 的 oracle screen；它们可能仅改善
噪声平均，而不能解决复合干扰的结构混淆。`M3`/`M4` 改变的则是可识别性条件，必须先
证明它们在实际采集/仿真合同中可部署，不能借用 component-level oracle。

## 若合同被提供后的不可省略步骤

1. 在改生成器前，新建预注册，定义身份、两类噪声/通道子流、共同状态和独立状态；
   明确旧 `receiver_noise_seed` 如何扩展且不改变 identity 定义。
2. 新增数据张量和严格验证器，确保训练、selection、development、confirmation 的
   identities 与所有历史角色零交集，并使各通道/多视图不被错误计为独立样本。
3. 先运行 V10 的 SawFM 或 VFT oracle screen，包含失配/置乱强控制；screen 失败时
   不训练分离器、router、分类器或全 12 参数模型。
4. 只有 screen 通过，才允许在另一个新身份角色上运行等计算 C0/C1/C2/C3 三 seed
   development；所有模型、阈值和成本测量必须在生成 development 数据前冻结。

## 证据边界

`test_generated=false`；`test_opened=false`；`locked_test_used=false`。

审计的源文件：

- `C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成\create_closedset_6class_config.m`
- `C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成\core\compose_jamming_sample.m`
- `C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成\core\generate_dataset_from_manifest.m`
- `C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成\core\validate_generated_dataset.m`
