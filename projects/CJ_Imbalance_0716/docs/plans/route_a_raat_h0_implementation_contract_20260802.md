# RAAT-H0 实现冻结契约（生成数据前）

本文件是 `RAAT-H0/v1` 的实现接口，不是模型实现。只有以下接口、静态检查和哈希被逐项满足后，才允许生成预注册的 train-only 源；不得在实现后依据 smoke 指标修改本契约。

## 必须新增的文件

1. `cjlt/models/raat_local_map.py`：仅含三个确定性 map extractor 与 `RAATLocalMapNet`；不得 import 全局 Route-A 分类器、query、logit 或参数元数据。
2. `experiments/generate_route_a_raat_h0_trainonly_v1.m`：沿用 0716 生成协议，固定身份范围与 63x80 组合；不得生成任何 test 文件。
3. `experiments/audit_route_a_raat_h0_trainonly_v1.py`：检查哈希、5040 唯一 receiver identities、范围、角色、历史交集为 0、无 test-v3 访问。
4. `experiments/run_route_a_raat_h0_v1.py`：严格实现两臂、三目标、五种子、epoch-30、paired bootstrap 与所列审计收据。
5. `tests/test_raat_local_map.py`：CPU 确定性、形状、错误参考图差异、无 NaN、同输入 shape、参数量、无禁用输入的测试。

## 静态禁止检查

实现冻结脚本必须扫描上述代码，且不得 import 全局 Route-A 模型或读取任何 `power`、`parameter`、`width`、`replica`、Doppler 标签元数据、全局 `logit`、`query`、`token`、`position`、`threshold`、`calibration`、`Platt`、`scale_hough`、`hough`、`pair`、`prototype`、`focal`、`stft`、`spectrogram`、`target_specific_complex_evidence`。优化器中唯一允许的 `weight_decay=1e-4` 不构成重加权。字符串检查只是最低要求；人工 import 审计也必须通过。

允许在信号数学中使用固定变量名 `doppler_hz` 来建立**固定 RAAT 网格**，但不得从数据 metadata 读取或拟合 Doppler；静态审核须将该唯一例外单独列出。

## 生成前 smoke（不生成正式数据）

只允许在内存中合成最多 8 条非正式 I/Q 波形，用于验证图的形状、有限值、错误参考差异和模型前向；不得保存为开发数据、不得训练、不得产生指标或选择任何常数。若失败，只能修正实现使其符合本契约，不能改变数学、网格、网络或超参数。

## 放行条件

生成前必须写出 `outputs/tapc/route_a_raat_h0_implementation_freeze/implementation_manifest.json`，其中包含五个代码文件和本预注册文件的 SHA-256、Python/MATLAB 环境、CPU smoke 收据、静态禁用检查、参数量 `7425`、以及 `test_v3_opened=false`。这份 manifest 一旦生成即不可改写；任何实现变更需新候选和新的 Stage 0，不能在 RAAT-H0/v1 上修订。
