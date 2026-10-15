# Route A：0716 MFT/VFT 生成器谱系审计

日期：2026-08-03  
结论：**0716 的实际生成器语义与单脉冲可观测性审计使用的配置一致；早期参考目录中的旧函数不作为 0716 物理或结果证据。**

## 核验范围

审计固定了官方生成器根目录 `C:\LJQ_code\干扰数据\干扰信号生成程序\可配置组合干扰数据生成`：

- `create_closedset_6class_config.m` 固定 100 us LFM、300 us PRI、单脉冲、MFT 3--8 个目标及 30--50 us 相对时延、VFT 1--7 kHz；
- `generators/generate_mft.m` 使用无放回离散时延槽、排序后的最终时延和完整 PRI 支撑校验；
- `generators/generate_vft.m` 对单 PRI LFM 施加连续多普勒相位并加入到达时延。

实际 MATLAB 生成器测试 `TestSingleJammingGenerators` 已运行并得到 **57 passed, 0 failed, 0 incomplete**。这直接排除了把旧目录 `Data_gen_Multi_fake_target.m` 的历史实现细节外推到当前 0716 数据集的做法。

## 与方法设计的关系

该审计加强了先前的配置级结论：MFT 与 VFT 的困难首先是当前单脉冲观测孔径与复合背景中的可归因性，而不是旧生成器误用、标签数计算错误或无效时延槽造成的表面失败。

它不重建既有数据的每一个随机样本，也不证明任何分类器、损失或融合有效；所有工件均记录 `iq_opened=false`、`training_performed=false`、`test_v3_used=false`。

## 工件

- [机器审计程序](../../experiments/audit_route_a_0716_generator_lineage.py)
- 不可覆盖收据：`outputs/tapc/route_a_0716_generator_lineage/summary.json`
- [单脉冲可观测性审计](route_a_single_pulse_observability_limit_results_20260803.md)
