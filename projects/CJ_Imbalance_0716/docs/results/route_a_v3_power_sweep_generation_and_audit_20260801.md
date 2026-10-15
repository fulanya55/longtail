# Route A development-v3 九挡身份配对功率数据生成与审计

## 结论

已完成 `10:5:50 dB` 九挡、train-only、同 receiver identity 的 development-v3 功率视图生成。
总计 226,800 个视图、22,877,724,784 bytes。九挡身份、标签和 development-v3 角色继承
审计全部通过；未生成 test 功率视图，也未打开 test-v3 MAT。

## 数据契约

- 基础 seed：`24000000`
- 每挡样本数：25,200；每挡 identity 唯一数：25,200
- 63 种非空组合，每组合 400 条
- FMB/SawFM/SinFM 固定 30 dB；MFT/VFT/ISRJ 同步扫描 10–50 dB
- 基础 identity SHA：`003e1f6a4eedd61257aa1e7b226ae0811e377891aae5fff2c68597025743b81a`
- 标签 SHA：`5fe0a4e2b969c9bfe664718452ddde868a9d712c1cacf8ee12f105fbccaf86d6`

| 挡位/dB | 文件 SHA-256 |
|---:|---|
| 10 | `cda25bb47b38fb2b01a40cf8bc65a9d4ee685fac10201d7f09ab51c5719fb8ca` |
| 15 | `7b36966f58a0a5e79ea1b71b1c8f9f5a4faf8746083b9fbb1b42a6afd8b9d6ca` |
| 20 | `a4ddbfbccbeede57c54ea51fdd074f2cf110fe42a2eb64b12854f7b84171247c` |
| 25 | `2d1b75a13e82348b6367757d006a62b43964cbf542415475a5b4c353d033373f` |
| 30 | `b0833ec6c3ef72b10c302f0ed4cb3c1c0901e1c5371987b8cbdf6e6a9f345963` |
| 35 | `46464cf6caac55ac506dda6843a301d6e59d51b0a886720640dcfffb474861cf` |
| 40 | `2112e45307fd33af0c65936d07c7e83400530ecbe2a20fdf14ba8ac6c9c7875c` |
| 45 | `79215ccd312b1eec43778bab7cf5cd4b8ed21b1d73ef43f36f5bc0672d7c8c8a` |
| 50 | `0ddcaebd73e8e8878cb2032d476687b1ac26490e9c1a1ce1c0377b5d410c86d3` |

五个随机种子的 fit/validation/early/calibration/threshold identity 均保持互斥。封存 manifest
继续记录 development-v3 与 test-v3 receiver identity 交集为 0，且 test-v3
`access_authorized=false`、`used=false`。

## 产物

- 生成 receipt：`outputs/tapc/route_a_v3_power_sweep_generation/formal_power_sweep_generation_receipt.json`
- 身份审计：`outputs/tapc/route_a_v3_power_sweep_audit/route_a_v3_power_sweep_identity_audit.json`
- 文件清单：`outputs/tapc/route_a_v3_power_sweep_audit/route_a_v3_power_sweep_file_inventory.csv`
- 数据根目录：`C:\LJQ_code\干扰数据\复合干扰数据0731_v3\power_sweep_9level_v3`

