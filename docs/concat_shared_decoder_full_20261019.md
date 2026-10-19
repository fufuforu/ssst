# 点云1024重建token + 超点理解token：Concat共享Decoder/GFM

用户在A点云1024双分支版本基础上授权本结构，并授权17服务器八卡full4训练。
独立注册键 `concat_shared`，架构 `OMEGA_POINT1024_SUPERPOINT_CONCAT_SHARED_DECODER_GFM_V1`。
模型：`tokengs/models/concat_shared_decoder.py`。不改67952及其固定代码快照。

## 数据流

1. 冻结Omega/SAM只读两张context RGB；不读GT生成token。
2. 原点云置信度/网格覆盖/FPS初始化1024个重建锚点，保留基础token、几何/RGB内容和来源GFM特征投影。
3. **一套**GFM Adapter生成1024维共享K/V及其投影网格。该网格经同一1024→256特征读出，
   用于融合点云上的超点邻域编码和像素Mask特征；不再有独立语义GFM Adapter或原模型的gfm_projection。
   点云token初始化中的来源GFM投影是初始化投影，不是额外Decoder K/V分支。
4. 理解thing token保持原超点初始化器，256维投影到1024维；另有两个stuff token。
   序列严格为 `[1024 reconstruction | N superpoint thing | 2 stuff]`。
5. 只通过一套12层1024维Decoder：共享GFM cross attention → 所有token间无阻断self attention → FFN。
   原A的四层独立256维理解Decoder被删除。没有stop-gradient、跨类型attention mask或额外Write注入。
6. 重建token继续原锚点PE、ray bias和位置/半径更新；理解/Stuff位置只作固定初始化几何参考，内容在每层更新。
7. 每层仅截取前1024个token供原重建监督；最终仅这1024个token生成64 children/anchor，共65536 GS。
   后N+2个token经LN/1024→256任务读出，沿用A分类、Gaussian子特征/像素证据混合、Mask点积、渲染和匹配损失。
   Mask按理解query读出，GS父关系仅为gaussian_parent_anchor。

统一Decoder并不是原A的Write=0双Decoder联合训练；两类任务现在能经共享注意力直接相互影响。
因此本实验除拼接外，也改变理解Decoder、共享GFM特征来源、理解几何更新方式及参数量，不能声称严格单变量。

## 初始化与预算

初始化仅读取attempt21的 `checkpoint_untrained_initial.pt` 及preflight，不读取其在训权重/优化器或查询训练状态。
所有同名同形状共享张量逐值继承该零步文件；仅新的共享特征投影和理解输入/输出投影用seed42初始化。
空Adam、progress全0、共同零步RNG显式1→8；实际数目与SHA以新run/preflight.json为准。

训练沿用A full4：1191场景/8337窗口、4172更新、batch8、AdamW LR1e-4、100步LR warmup、
理解权重.05/1043步ramp、每50步checkpoint、初始/epoch末原探针。未增加新的损失或改变官方后处理。
17服务器实际为8×4090、4090分区；此前13服务器为3090，需注明硬件差异。

CPU测试包含唯一Adapter/Decoder结构检查、与原Adapter K/V逐值对比、双向token梯度、
固定理解几何，以及真实12层1024+N+2序列与65536 GS读出。
CPU使用合成Omega/SAM及替代renderer，不能代替真实GPU/八卡首步证据。

提交后的作业号、一次调度状态、最终预检和交接见 `concat_shared_full_submission_20261019.md`。
