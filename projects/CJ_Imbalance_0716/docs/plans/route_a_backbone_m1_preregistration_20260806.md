# Route A Backbone M1 独立筛选预注册

候选：`B1 = B0 + Complex Anti-Aliased Downsampling`  
唯一改变机制：三次 stride-2 下采样之前加入固定复数共享低通预滤波。

## 固定实现

- 核：五点二项式 `[1,4,6,4,1]/16`
- padding：时间维 reflect 2 点
- 实部、虚部分别卷积，但使用同一固定 buffer
- 固定滤波不含训练参数
- 滤波后仍使用原 `AvgPool1d(kernel=2,stride=2)`
- stage 长度保持 `3000→1500→750→375`
- 其余复数卷积、多尺度分支、late fusion、global mean/std、128 维投影和
  classifier 全部不变
- 模型注册名：`dfms_iq_anti_alias`
- 参数量必须与 B0 完全相同：257511

训练、五 seed 角色、Focal 损失、优化器、校准、阈值和 early 报告协议完全
继承 Stage 1 B0。候选必须从与同 seed B0 相同的随机初始化规则和 batch order
开始，不能加载 B0 已训练权重。

M1 先相对 B0 执行五 seed development 筛选。只有满足 Goal 中全部硬门槛后，
才进入同容量/信息破坏控制；失败则保存收据并永久关闭 M1 及其换核、加宽或
调阈值版本。
