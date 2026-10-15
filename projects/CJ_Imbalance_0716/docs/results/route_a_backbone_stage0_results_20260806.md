# Route A Backbone 强化 Stage 0 结果

结论：Stage 0 已完成，证据属于**全新数据与工程合同确认**，不属于模型性能
确认。Stage 1 的 B0 五 seed 正式运行可以启动，test-v3 仍未授权、未分析、未使用。

## 数据结果

- train-only MAT：`E:\LJQ_code\干扰数据\route_a_backbone-development-v1_20260806\closedset_6class_train.mat`
- 大小：`2,549,110,933` bytes
- SHA-256：`c4122ffae67de855f97a74b42c17ef3df824fad31888dcd6e9dd28b9952c00ac`
- sample seed：`2550000001–2550025200`
- receiver identity：`2748677743–2748702942`
- 样本/receiver identity：`25200/25200`，每 identity 仅一个视图
- 63 种非空组合各 400 条；六类各 12800 个正例
- 10 dB 弱正计数：FMB 2853、SawFM 2855、SinFM 2865、MFT 2865、
  VFT 2905、ISRJ 2814
- MATLAB partial 和 final 回载均确认 manifest、config、metadata、尺寸一致
- 未生成 test manifest 或 test MAT

历史合同扫描登记 105 条非当前 receiver range 记录，registry SHA-256 为
`81c41f7fcb8c72d075391b3db12e706340a3ebd4f2ce37080d6515737865467b`，
候选区间重合数为 0。

## 五 seed 角色

每个 seed 的角色固定为 fit 1920、validation 480、early 2520、
calibration 2520、threshold 2520，样本和 receiver identity 均无跨角色重合。

| seed | role hash |
|---:|---|
| 2039 | `9c548f2f9b80bb31316728b43931c104d5c5c1b69d9809a9a98ebc2d8f9bd1fb` |
| 2040 | `6d8c26c8bebd15a041d81950925d7b38013123baac6143d2dea0afab9ffdc2c1` |
| 2041 | `51dabe260cceb7279d925d50ebc5eb7aec5a27d23ed897196957be332a7fcff7` |
| 2042 | `a2cc54dfbdc3298fc7e9d7856f9376e5561ea6027602145de663b2bd03a61f68` |
| 2043 | `1bda237534e1182983efb575307ad047d32e668c8e8f8e0f543aea9ad45e25a2` |

## B0 工程验证

真实 MAT、RTX 5080 CUDA smoke 已完成：

- B0 参数：257511
- logits/global：`[8,6] / [8,128]`
- stage1/stage2/stage3/temporal：`[8,16,3000]`、`[8,32,1500]`、
  `[8,48,750]`、`[8,128,375]`
- 两个优化步 loss：`0.321267`、`0.441582`
- 梯度有限，clip 前 norm：`1.3381`、`1.4411`
- checkpoint reload 最大绝对误差：`0`
- 峰值 CUDA 显存：`151,823,360` bytes

这些结果只证明输入、训练和工件链路可运行，不证明 B0 的正式性能。

## 工程异常与修复

首次 Python identity audit 因当前 `build_label_longtail` 返回对象未填充
`label_counts` 而中止。修复为依据已经冻结的 selected indices 直接计算实际
标签计数；数据、角色算法和预注册参数均未改变，随后五 seed 审计通过。

原 MATLAB generation receipt 使用系统默认文本编码，UTF-8 读取时中文路径
显示为替换字符。原收据保留，并新增路径纠正收据；实际文件、SHA 和增强回载
不受影响。生成器以后显式以 UTF-8 写 JSON。

## 边界

- `test-v3.access_authorized=false`
- `test-v3.analysis_allowed=false`
- `test-v3.used=false`
- development-v3、H3、H5o、H0v、P1、P2 均未用于本阶段选择
- 单 seed 两轮 B0 runner 试跑仅为 smoke；对应目录已有证据边界纠正收据，
  不计入正式五 seed B0 结果
