# 代码发布验证

日期：2026-10-16。结构化记录见 [`validation.json`](validation.json)。

## 已完成

- 2,690 个来源文件逐一核对仓库副本与原文件的 SHA-256，零差异。
- 1,178 份 Python 源码通过语法解析（两个工程、数据生成器与工具目录）。
- 324 份非空 JSON、25 份 YAML、1 份 TOML 通过解析；另保留 1 份来源工程已有的空 JSON，见下方说明。
- 路线 A 现有基础测试：**34 passed**（9.70 秒）。覆盖标签注册、参数 schema、指标与校准、分类损失、配置冻结、运行管理和模型接口。
- 路线 B 现有基础测试：**8 passed**（6.24 秒）。覆盖复数乘法、I/Q 输入、原生骨干、统一头和专家输出及组合指标。
- 四个 CLI 帮助入口退出码均为 0：路线 A `run_tapc.py`、路线 B `cj0724.train`、根目录可视化脚本和论文分支长尾 MILP 规划器。完整帮助输出保存在本目录。
- 提交清单没有 MAT、checkpoint、NumPy 二进制、PDF、缓存或超过 10 MiB 的文件。常见令牌、私钥和带密码 URL 模式扫描没有匹配。

测试命令与版本详见结构化记录。环境为 Python 3.10.20、PyTorch 2.11.0+cu128、NumPy 1.23.5、h5py 3.10.0、SciPy 1.10.1、scikit-learn 1.2.2。依赖文件列出兼容范围，不是对任意版本组合的完整验证。

## 原样保留的异常

`projects/CJ_Inbalance_0724/configs/p3ok_durable_generation_outcome_freeze_20260802.json` 在原工程中为空文件。本次不填造历史冻结结果，归档时原样保留；它不是有效运行配置。其余解析检查未发现异常。

## 验证范围

本次没有训练、读取锁定测试、运行 MATLAB 测试或执行完整历史测试集，也没有重验数据仓库的全部远端大文件。基础测试通过不意味着依赖原始 MAT、外部旧模型、生成数据或冻结预测的历史实验可在空环境中直接复现。源码快照的外部依赖见 [`PORTABILITY.md`](PORTABILITY.md)。

可在仓库根目录执行以下只读检查，复核源码快照：

```powershell
python tools/verify_source_snapshot.py
# 仅在原始工程路径仍可访问时：
python tools/verify_source_snapshot.py --check-originals
```
