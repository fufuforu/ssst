# 旧 A4：完整 V4 重建初始化与 No-Anchor-CA 两组独立实验

用户于 2026-10-22 指定回到旧 A4，并明确选择第一组“完整迁移 V4 重建初始化：固定 1024 个采样 Anchor＋可学习基底＋内容/视觉编码”。不取消、不修改此前提交的 P1024iso C/D 作业。

## 实验定义

|组|重建初始化|Anchor 数 / GS 数|Object→Anchor CA|Object→FM CA|梯度隔离|
|---|---|---|---|---|---|
|旧 A4，已有结果|超点特征 256→1024|超点数量 / 数量×64|开|开|无|
|V4init，本轮|V4 采样位置半径＋可学习基底＋7D 内容编码＋原始 GFM 视觉编码|1024 / 65536|开|开|无|
|NoCA，本轮|原旧 A4 超点初始化|超点数量 / 数量×64|关，几何 R 更新 c/s|开|无|

两组互相独立，不同时叠加两个改动。V4init 检验完整初始化方案，同时改变 Anchor/GS 数量、视觉投影、可学习基底及内容编码；不能将结果单独归因于容量或可学习基底。NoCA 则是在旧 A4 上直接检验 Anchor CA 是否有用。

## 冻结旧 A4 配方

- 8337 训练窗 / 1191 场景；原 manifest 与次序，**2 context＋2 novel**，256×256；不带入 P1024iso 的 2+4 数据扩展。
- 8 卡×1，全局 batch 8；4 epoch /4172 updates。
- AdamW、weight decay 分组、betas=(.9,.95)、eps=1e-8、全参数统一 clip=1 沿用旧 A4。
- **全部参数**基础 LR=1e-4，warmup100，cosine→1e-5；不迁入 V4 几何组的 1e-5 基础 LR。
- 理解权重 `0.05*min(1,(step+1)/1043)`。
- 重建监督 **L6/8/10/12，权重 [.1,.2,.3,.4]**；不改为 sup612。
- 理解 query 仍为原投影点云特征→超点初始化、256D、四层理解 decoder；不换成 FM query 初始化。
- SemanticMemory 原生 **128×128**，按旧路径在 child lifting 前插值到 256；不换成 P1024iso 的原生256 FM。
- Write=0；不加隔离、不加 adapter、不加 IoU/depth/boundary 损失，不修改 GS 颜色头。
- 从同一个**旧 A4 零步**权重重新训练；不从 epoch4 接着训练，不继承 smoke 权重/优化器。

旧零步路径：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_untrained_initial.pt`

SHA256：`3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744`。

原训练 manifest SHA256：`87e3cde298569c4e7735cf75ee596868eadf01e9a6a95438c6be51e42f55a523`。

## V4init 实现

`tokengs/models/olda4_initialization_ablation.py`：

```python
mu, radius, color, source = seed_anchors(
    omega['points'], omega['confidence'], rgb, 1024,
    return_source_indices=True)
visual = sample_anchor_gfm(gfm_grid(omega['patch_layers']), source)
origin = mu.mean(1, keepdim=True)
ell = ((mu-origin).square().sum(-1).mean(-1)).sqrt().clamp_min(.05)
content = cat([(mu-origin)/ell, radius/ell, color])  # 7D
tokens = gs_tokens + anchor_content(content) + anchor_visual_projection(visual)
```

- `gs_tokens` 是持久的 `[1024,1024]` 参数表，标准差取原 V4 `gs_token_std`。
- `anchor_content = Linear(7,1024)→GELU→Linear(1024,1024)`。
- `anchor_visual_projection = LayerNorm(2048)→Linear(2048,1024)`；对精确来源视图/像素的**原始 GFM**采样，不经过旧的2048→256瓶颈。
- `seed_anchors`、GFM构造/采样函数与旧 V4 使用的原文件相同，不改采样口径。
- RGB 只作为 **token 内容编码**，没有将颜色写入 Gaussian RGB 初值。
- 移除无用的重建超点初始化器及256→1024投影；理解超点初始化器、共同点云构建仍保留原权重。
- 父索引 `gaussian_parent_anchor` 与兼容别名 `gaussian_parent_query` 都表示 Anchor 索引，不表示理解 query 索引；显式返回 `gaussian_parent_query_semantics`，删除不再成立的 `anchor_to_superpoint` 输出。训练支持匹配使用原理解侧 `source_superpoint_ids`，不使用这两个父索引。

## NoCA 实现

四个 DynamicObjectLayer 移除 Anchor Attention 的 Q/K/V/O 参数，读出改为：

```python
ev = zeros_like(q)
R = geometry_bias(mu, c, s).softmax(-1)[:, None]
q1 = norms[0](q + ev)
# 后续 FM CA、自注意力、FFN、update_geometry 不改
cnew, snew = update_geometry(R.mean(1), mu, qnew, c, s, ell)
```

保留 Query 残差后的 LayerNorm，R 对 Anchor 归一化；stuff 行均匀。没有新增 detach：LU 仍可通过 GS 几何、child mask 特征和共享初始化等路径影响重建。这是去掉指定 CA，不能声称整个理解/重建分支已断开。

## 提交前验证与迁移

执行快照：`/space/mawb/ssst_olda4_v4init_noca_code_20261022_v1`。

运行根目录：`/space/mawb/ssst/group_plus/semantic_resolution_v1/oldA4_v4init_noca_ablation_20261022_v1`，分 `V4init/{smoke,train,eval}`、`NoCA/{smoke,train,eval}`。

真实零步 CPU 迁移已通过：

|组|保留原张量，逐值一致|精确移除张量|新增张量|可训练参数|
|---|---:|---:|---:|---:|
|V4init|655|38（重建初始化36＋旧投影2）|9|197014128|
|NoCA|661|32（四层Anchor CA）|0|192746609|

每组均验证并拒绝7类错误：已训练权重、非空Adam、数据不符、rank RNG不全、额外张量、缺失保留张量、移除张量形状错误。CPU收据：运行根目录 `CPU_VALIDATION.json` 与各 stage `PREFLIGHT.json`。

`tests/test_olda4_ablation.py` 验证：与 V4 token 公式逐位一致且各输入/新参数有效；NoCA 的几何归一化、Anchor 特征不影响 q、FM 仍有梯度；两组真实 core 上实际理解/重建损失、原共享特征初始化和非隔离梯度路由；2+2 batch 合约拒绝2+4。CPU集成只缩小渲染与 Anchor 数以控制资源，不代替 GPU 验证。

为明确恢复旧 A4，快照中的 `object_locus_dual_stream_v1.py`、`unified_query_v1.py`、`unified_query_loss.py` 恢复为旧 A4 原文件；原 decoder、种子函数、GFM模块、点云超点模块、训练调度策略也与旧 A4 逐字节相同。

## GPU smoke、依赖链与评测

每组单卡、同一个原训练窗口、两步 optimizer update，每步累积8次；V4init 使用真实1024 Anchor /65536 GS，NoCA 使用原动态超点数×64 GS。实际256²、2+2，不是256步短训练。

记录实际显存峰值/耗时、reconstruction/understanding/shared 梯度、V4初始化9个参数梯度、定向 LU/LR 反传梯度、实际 FM128 输出形状、GS父索引、首步1rank参数变化哈希及预算。LU→重建的梯度在本实验应非零，不能沿用隔离实验的“必须零”断言。

每组：`1GPU smoke afterok→8GPU train afterok→8GPU official1860 eval`。smoke收据核验后，训练仍从原零步重新开始。失败写 `JOB_FAILED.json`，依赖作业阻断，不静默重试训练。

训练成功须核验4epoch4172步、8rank首步参数哈希一致且已更新、完整训练元数据，才硬链接冻结 latest 作评测；不复制大checkpoint。

正式评测使用原 official1860 cohort、312场景、固定 SIU3R commit `8ea80166be76854f938e90521f1a5b688b755c87`；输出与原 A4 的输入RGB/GT清单逐窗对齐并生成对比表。target保持既有 target-all6 口径。单种子PQ的小幅差异须结合逐类和TP/FP/FN判断。

节点池12/13/14/17/18，单节点训练8卡，允许3090/4090，不使用节点列表要求同时占用五台。

提交后只做一次有界状态检查即停止，不等待 GPU 首步或 smoke 完成。实际作业号与执行commit由独立 publication worktree 的运行记录补充，不修改执行快照HEAD。
