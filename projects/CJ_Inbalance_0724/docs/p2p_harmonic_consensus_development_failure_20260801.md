# P2p 倍频峰列回溯开发试验：失败记录

P2p 使用全新开发种子 `3000000000`，与 P2i--P2o 的身份重叠为 0。该轮只增加整数约数峰列一致性机制，未修改分离器、分类器、损失函数或 P2m/P2n 冻结实现。

## 结果

P2p 未通过开发门禁，不进入新种子确认，也不挂接 12 参数系统。

- exact catastrophic error 数从冻结 P2m 的 `54` 增加到 `372`，其中新引入 `324` 个。
- exact near-low MAE 从 `0.1991 us` 恶化到 `2.9286 us`，Pearson 从 `0.7074` 降到 `0.1652`。
- separated far-high 平均 Pearson 从 `0.2941` 降到 `0.0467`，平均 MAE 从 `16.13 us` 增加到 `67.25 us`。
- 唯一明确改善发生在 far-low：exact MAE 从 `0.1133 us` 降到 `0.0261 us`，Pearson 从 `0.7080` 升到 `0.9998`；separated 平均 Pearson也从 `0.8891` 微升至 `0.8913`。

## 失败原因

SawFM 的单个全长 ACF 中不仅有复扫周期峰，还含有调频结构产生的密集局部峰。仅凭“候选整数倍位置存在峰”无法区分真实复扫基频和结构性旁峰：

- exact ID-control 回溯率为 `8.125%`；
- exact near-low 回溯率为 `25%`；
- separated near-high 回溯率约 `40%`；
- separated far-high 回溯率约 `56%`。

因此失败的不是“倍频异常”假设本身，而是单窗口 ACF 内部证据不具独立性。下一轮必须引入跨窗口重复性：只有不同时间窗口独立给出同一个较短周期，并且它与全长候选构成稳定整数比时才允许回溯。

## 证据边界

- 预注册 SHA256：`04046235e30ecba250202c4a64431d80cef12f4dc7c659c7afad90151147bc30`
- 实现 SHA256：`cf68c7780fce0e242674f9a9d5297b88c341ec9d833ee8f55c5588f16681cfe1`
- 汇总 SHA256：`c8a942f28b7ed1e26afc01416dfcb87c396915a58eaede42e6dc7f01a610751a`
- `development_passed=false`
- `locked_test_used=false`
