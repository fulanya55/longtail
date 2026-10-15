# Route A：PGL 官方代码兼容性审计

日期：2026-08-03  
性质：只读、固定 commit 的方法复现审计；未加载任何项目数据，也未训练模型。

## 审计对象

- 论文：Lin et al., *Probability Guided Loss for Long-Tailed Multi-Label Image
  Classification*, AAAI 2023，DOI `10.1609/aaai.v37i2.25244`。
- 作者公开仓库：`Kunmonkey/PGLoss`，README 将仓库明确标为该论文的 official
  PyTorch implementation。
- 固定 revision：`d11a6520b0241946672364d974a5ebbd43c554e0`。
- 审计文件：`mllt/models/losses/pg_loss.py`，SHA-256
  `92d4d0ddef7e2f7590a63aeedbc60f6191db104762a27ff234e5088418d7fa64`。

PGL 是比 token-level MLC-NC 更接近当前同主干公平比较的近期多标签长尾候选，
因为它表面上是损失/概率重平衡方法。正因如此，本审计要求保留其官方算法，而不
允许把几行可见代码自行修补后命名为 PGL。

## 固定源码的静态结论

在该 revision，官方 `pg_loss.py` 不能作为本项目的精确可运行复现：

| 检查 | 结果 | 影响 |
| --- | --- | --- |
| Python AST 解析 | 失败：line 148 `SyntaxError: expected ':'` | 源码本身不能导入或执行 |
| 依赖 | `mmcv` 与 VOC/COCO 的类别频率/划分文件 | 不是当前 I/Q 六标签全局头的直接 loss 插件 |
| 设备处理 | 构造器内多次硬编码 `.cuda()` | 不满足当前 CPU/GPU 统一训练合同 |
| 未定义变量 | `apaf`、`neg_scale1` 被直接引用 | 修补需决定未公开的算法语义 |

因此不能判断这些问题是否来自仓库维护状态、未提交依赖或原论文的未公开细节；
但它们足以说明**该固定公开版本不能被精确复现**。自行补上冒号、默认值或类别
切分后，即使得到一个数值结果，也只是“PGL-inspired 自定义损失”，不能作为 PGL
的公平基线。

## 决策与证据边界

`exact_equal_protocol_reproduction_admissible=false`。PGL 不进入五种子训练、
开发选择、校准或阈值比较；不以此补齐“最新多标签长尾方法的公平数值基线”。
该结论不会削弱已经完成的 BCE、pos-BCE、Focal、ASL、DB、CB、LA、两阶段解耦
和同容量对照；它只将 PGL 从“可能可复现”改为“有官方仓库但固定代码不足以
审计复现”。

审计本身记录 `locked_test_used=false`、`test_v2_used=false`、`test_v3_used=false`。

## 工件

- 审计程序：`experiments/audit_route_a_pgl_compatibility.py`
- 机器收据：`outputs/tapc/route_a_pgl_compatibility_audit/summary.json`
- 单元测试：`tests/test_route_a_pgl_compatibility_audit.py`（`1 passed`）
- 相关但结构不兼容的近期基线：
  `docs/reports/route_a_mllt_reproducible_baseline_candidate_audit_20260801.md`
