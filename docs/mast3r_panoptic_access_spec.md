# MASt3R + 配套 panoptic 预训练理解路径：接入规格（待评审）

> 面向：GPT（设计评审）。状态：**仅设计，不启动正式训练**。
> 本文件的所有"现状"陈述都来自**实测或代码**，已标注出处；所有"待确认"都显式标出。

---

## 0. 固定方向（本次设计范围）

| 部分 | 固定方向 |
|---|---|
| 现有重建路径 | **保留** LocusGS reconstruction pretrained、GS/anchor tokens、Gaussian 生成 |
| 新图像理解路径 | 接入 **MASt3R 图像特征 + 与之配套的 panoptic 预训练 adapter／mask decoder** |
| Object／mask tokens | 从预训练理解路径读取图像证据，**保留其可复用的预训练能力** |
| 两组 token 的联系 | 设计 reconstruction tokens 与 object tokens 的**双向交互**，参与生成过程 |
| 最终输出 | 几何、外观与实例 membership **落在同一组 3D Gaussians 上** |
| 初始化边界 | **不使用**已在 ScanNet 上完成任务训练的 `siu3r_epoch100.ckpt`，避免把现成任务能力混入结构验证 |
| 当前工作范围 | 完成**接入规格 + 权重映射**；**暂不启动新的正式训练** |

**核心约束（来自评审）**：不能把理解 adapter 和 mask decoder 拆下来接到任意 encoder 上就宣称保留了 panoptic 预训练能力。设计必须说明**它们原本接收什么特征**，以及**新前向如何保留这套配合**。§2 与 §3 就是对这两问的回答。

---

## 1. 双方接口的实测事实

### 1.1 我们这边（LocusGS / TokenGS，重建路径）

出处：`tokengs/options.py`、`tokengs/models/canonical_recon_models.py`、`tokengs/models/input_types.py`、`tokengs/models/spatial_grounded_tokens.py`

| 事实 | 值 / 出处 |
|---|---|
| 图像尺寸 / patch | `img_size=(256,256)`、`patch_size=8` ⇒ **32×32 patch/视图**（`options.py:43-44`） |
| 上下文视图数 | 2 ⇒ encoder 侧 **2048 token**（2 × 1024） |
| encoder 输出形式 | `EncoderLatent.keys/values: [B, H_heads, N, C//H_heads]` —— **已经是投影后的 attention K/V**，不是原始 patch 特征（`input_types.py:59-62`） |
| 实测形状 | `values = (1, 16, 2048, 64)`（即 16 heads × 64 = **d_out 1024**） |
| 几何对齐 | `patch_plucker_rays(...)` 把稠密逐像素 Plücker ray 池化到 patch 网格，**顺序 = view-major → row-major**（`spatial_grounded_tokens.py:89-99`） |
| 消费点 | `LocusGSRecon.forward` → `get_gs_tokens(batch_size, encoder_latent, patch_rays)`（`canonical_recon_models.py:146-152`） |

### 1.2 理解路径（SIU3R）原本接收什么

出处：`/space/mawb/SIU3R/src/models/{model.py, backbone_croco.py, vit_adapter/vit_adapter.py, mask2former/video_seg_decoder.py}`

```
model.py:
  _set_backbone()      → AsymmetricCroCo(CroCoNet)          # patch_size=16, d_out=1024
  _set_adapter()       → CroCoViTAdapter(num_block=enc_depth, embed_dim=enc_embed_dim,
                          size=image_size, patchsize=croco.patch_size,
                          interaction_indexes=[5,11,17,23], with_cffn, deform_ratio,
                          add_vit_feature=True, use_extra_extractor=True)
  _set_mask2former()   → VideoMask2FormerForVideoSegmentation(Mask2FormerConfig(
                          id2label=..., num_queries=..., train_refer_segmentation=False))

forward:
  feat1, feat2, all_feat1, all_feat2, dec1, dec2, shape1, shape2 = self.backbone(...)
  multi_scale_feat1 = self.adapter(img1, all_feat1)          # ← 逐视图
  multi_scale_feat  = [stack([f1, f2], dim=1) for f1, f2 in zip(msf1, msf2)]
  context_seg_output = self.mask2former(multi_scale_feat=multi_scale_feat, ...)
```

`CroCoViTAdapter.__init__` 里与"配合"直接相关的成员：

| 成员 | 含义 |
|---|---|
| `self.H = size[0]//patchsize`, `self.W = size[1]//patchsize` | **硬绑定输入分辨率与 patch 大小** |
| `level_embed = nn.Parameter(torch.zeros(3, embed_dim))` | **3 个尺度**的空间先验层级嵌入 |
| `spm = SpatialPriorModule(inplanes=64, embed_dim=embed_dim)` | **吃原始图像**的 CNN 空间先验 |
| `interactions = [InteractionBlock_Efficient(...) for i in range(len(interaction_indexes))]` | **4 个**形变交互块，**一一对应 4 个 ViT 深度** |
| `up = ConvTranspose2d(embed_dim, embed_dim, 2, 2)`；`norm1..norm4 = SyncBatchNorm(embed_dim)` | 输出 **4 个多尺度特征图** |

`VideoMask2FormerForVideoSegmentation`：`self.model = VideoMask2FormerModel(config)`、`class_predictor = nn.Linear(hidden_dim, num_labels+1)`；其 pixel decoder 消费 `multi_scale_features`。

### 1.3 权重现状（**阻塞项**）

`/space/mawb/SIU3R/pretrained_weights/` **只有** `siu3r_epoch100.ckpt`（5.46 GB）。代码引用的三个文件**全部不在盘上**：

| 代码引用 | 位置（代码中） | 盘上状态 |
|---|---|---|
| `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` | `model.py::load_recon_ckpt` | **缺失** |
| `panoptic_coco_pretrain_vitadapter_maskdecoder_epoch60.ckpt` | `model.py::load_seg_ckpt("coco")` | **缺失** |
| `panoptic_ade20k_pretrain_vitadapter_maskdecoder_epoch75.ckpt` | `model.py::load_seg_ckpt("ade20k")` | **缺失** |
| MASt3R 权重 | SIU3R 官方训练说明 | **缺失** |
| `siu3r_epoch100.ckpt` | 完整 SIU3R（**已 ScanNet 任务训练**） | 存在，但**按初始化边界禁用** |

全盘 `find` 未发现任何 `*mast3r*` / `*dust3r*` / `panoptic` 预训练权重（命中的 `.pth` 属于 sambamotr / MOTR / gaussian-grouping 等无关项目）。

⇒ **设计可以完成，权重需要先获取**。§6 给出获取与校验规程，并把"缺失"作为显式前置条件。

---

## 2. 为什么不能拆：三重耦合的机制论证

panoptic 能力不在"adapter"或"mask decoder"单个模块里，而在**三者的配合**上：

```
① ViT 的【四个特定深度】特征  all_feat[l]  for l ∈ {5,11,17,23}
        ↓   （宽度 = enc_embed_dim，深度由 interaction_indexes 固化）
② 【原始图像】 → SpatialPriorModule（1/4, 1/8, 1/16 CNN 先验）+ level_embed(3)
        ↓   （deformable interaction 把 ① 的语义与 ② 的空间细节逐层融合）
③ 【4 个多尺度特征图】 → Mask2Former 的 pixel decoder（通道/步长布局固定）
        ↓
   object/mask queries + per-query mask logits（这套 query 与分类头也是配套预训练的）
```

破坏其中任一环都会使"预训练配合"失效：
- 换掉 ViT（或改用**投影后的 K/V**）⇒ ① 的**深度语义**与**宽度**对不上，且丢失了 4 个深度；
- 去掉/改动 `SpatialPriorModule` 的输入 ⇒ ② 的统计量与 `norm1..4`、`level_embed` 的训练分布不一致；
- 改变 patch 网格（`self.H/self.W`）或输入分辨率 ⇒ 形变采样的位置语义整体偏移；
- 只搬 mask decoder ⇒ 它期望的是 ③ 的**具体多尺度布局**，不是任意特征。

**并且**：① 与 ③ 之间没有直接通道，全靠在 ② 里的逐层交互——所以"保序保形地接回原输入"不是可选项。

---

## 3. 我们与理解路径的不兼容清单（必须正面处理）

| 维度 | 我们（重建路径） | SIU3R 理解路径 | 后果 |
|---|---|---|---|
| patch / 分辨率 | 8 @ 256×256 → 32×32 | 16 @ 512×512 → 32×32 | **token 数相同（1024/视图）纯属网格算术巧合**，每 token 覆盖的空间范围差 2× |
| encoder 输出形式 | **1 个**投影后 K/V `[B,16,N,64]` | **4 个深度**的未投影 block 特征 `[B,N,1024]` | adapter 需要 4 个深度；我们只暴露 1 个且已投影 |
| 空间先验 | 无（几何靠 Plücker ray 偏置） | `SpatialPriorModule(原图)` + `level_embed(3)` | ② 环缺失，不能省 |
| 位置/几何编码 | Plücker patch ray（view-major） | CroCo 位置编码，**无 ray 输入** | 需重建位置对应，不能沿用 |
| 输出形态 | token → 64 child Gaussians → 渲染 | 2D multi-scale → per-query **2D mask logits** | 需要一层"2D mask → per-Gaussian membership"的**可渲染**归属模块 |

**结论**：不能把我们的 K/V 当 adapter 的输入，也不能把理解支路压成我们的形状。**理解支路必须自己持有 ①+② 的原始输入**（即 MASt3R/CroCo 的多深度特征 + 原图），并以 **object/mask token 层**与重建 tokens 交互——而不是在特征张量层强行对齐。

---

## 4. 接入设计

### 4.1 模块图（新增，全部为新文件；不修改 V3 的四个锁定科学模块）

```
                       ┌──────────────────────── 新增：理解支路（预训练，保配合） ───────────────────────┐
 原图 (2 ctx views) ──►│ MASt3R/CroCo encoder ──► all_feat[l], l∈{5,11,17,23}  (每视图)                │
                       │        │                                                                     │
                       │        └─► CroCoViTAdapter(img, all_feat) ──► 4×multi-scale maps (每视图)     │
                       │                                   │                                          │
                       │                                   ▼                                          │
                       │                      VideoMask2Former ──► {q_obj, mask_logits_2d}             │
                       └───────────────────────────────────┬──────────────────────────────────────────┘
                                                           │  object/mask tokens（携带预训练语义）
                                                           ▼
  重建 tokens（anchors, 1024/view, 来自 LocusGS） ◄── 双向交互（§4.3）──► object tokens
                                                           │
                                                           ▼
                     per-Gaussian instance membership（§4.4，可渲染） ──► 同一组 3D Gaussians
```

**注意**：重建支路**完全保留**——LocusGS 预训练权重、anchor/GS token 解码、Gaussian 生成、渲染与既有 mask/Hungarian 监督均不变。理解支路是**新增并联**，只在 token 层与重建交互。

### 4.2 理解支路的接口契约

| 接口 | 契约 | 断言（实现时必须写成代码断言） |
|---|---|---|
| 输入图像 | 与理解 backbone 训练时一致的分辨率/归一化（**待确认**：CroCo 训练用的 `image_size` 与 mean/std） | 输入尺寸 == backbone 配置尺寸 |
| `all_feat` | 4 个深度 `{5,11,17,23}` 的 block 输出，宽度 == `enc_embed_dim` | `len(all_feat)==4`；每个 `[B,N,C]`；`C==enc_embed_dim` |
| patch 网格 | `H=size[0]//16, W=size[1]//16`，与 `adapter.H/W` 一致 | 网格与 adapter 内部量一致 |
| 空间先验输入 | **原始图像**（不是特征），尺寸与 `SpatialPriorModule` 期望一致 | 形状断言 |
| adapter 输出 | 4 个多尺度 map，通道/步长与 mask decoder 期望一致 | 每尺度通道数与 `hidden_dim` 相关断言 |
| mask decoder 输出 | `class_queries_logits`、`masks_queries_logits` | 形状断言；`num_queries` 与配置一致 |

**未对齐项必须显式记录**：我们现有 pipeline 的 256×256 / patch 8 与理解 backbone 的输入规格不同。**两种可选**（需评审选一）：
- **(A) 双分辨率前向**：理解支路按 backbone 原生规格（如 512×512 / patch 16）单独前向；重建支路维持 256×256 / patch 8。代价：多一次 encoder 前向；优点是**完全不动预训练配合**。
- **(B) 统一到 512×512**：重建支路也升到 512×512（patch 8 → 64×64/视图 ⇒ token 数 ×4）。代价：改变重建路径与显存；相当于同时改两个变量。
**我建议 (A)**：它把"理解路径接入"保持为**单一变量**，符合本项目一贯的预注册纪律。

### 4.3 双向交互（生成过程中，不是末端融合）

沿用 Joint 已验证的**非对称**形式（一侧本来就有，一侧是新增）：

| 方向 | 现状 | 新设计 |
|---|---|---|
| reconstruction → object | **已有**：object 读 anchor 特征做 evidence | 改为**同时**读 anchor 特征与理解支路的 object tokens（一路新证据） |
| object → reconstruction | Joint 已有：token-only 残差写回（4 个零初始化 `Linear(256→1024)`，β 前 200 步 ramp） | 保留该机制；写回内容改为来自**有预训练语义的 object tokens** |

约束（与 Joint 一致，避免重蹈覆辙）：
- 只写 tokens，**不写 μ/ρ**；不加额外 attention bias；
- 路由归一化轴固定（`softmax` 沿 object 轴）；
- 写回强度用 β ramp，禁止 detach（"不能只在末端融合、也不能 detach 两条交互路径"）。

### 4.4 从 2D mask 到"同一组 Gaussians 上的 membership"

这是本设计的技术难点，也是"最终输出必须落在同一组 Gaussians 上"的落地处。

```
理解支路给出：mask_logits_2d[q, v, h, w]      （q = object/mask query, v = 视图）
重建支路给出：gaussians [B, 65536, 14]、每 child 的特征 f_gaussian [B,65536,256]
                         │
                         ▼  归属（三种候选，需评审）
 (i) 在【理解支路自己的像素域】算 c_i 合成权重：每个 Gaussian 在视图 v 的 EWA 足迹 × 前向-后向
     合成权重 → 对 mask_logits_2d 加权求和 → per-Gaussian membership
 (ii) 用理解支路的 mask feature 与 f_gaussian 做 dot-product（Mask2Former 风格）→ sigmoid
 (iii) 学习一个 query-to-Gaussian 交叉注意力
                         │
                         ▼
 per-Gaussian membership [B, 65536, Q]（经 alpha 合成器渲染 → 保证遮挡正确）
```

**必须满足**（来自仓库的历史教训）：
- membership **必须可渲染**（经 alpha 合成器），不得退回"投影中心判定"——后者曾产生 62.8% 的跨实例伪影，改用遮挡感知合成权重后 purity p50 = 1.000；
- 若采用 (i)，需**重新验证**理解支路的 `c_i` 近似与渲染 alpha 的一致性（我们已有该验证脚本 `token_instance_compositing.py`，实测 MAE 0.0050–0.0058；新支路需要重跑同一验证）。

### 4.5 参数级梯度边界（**预先写死，不接受"detach 就算隔离"**）

| 参数组 | 语义损失梯度 | 重建损失梯度 | 理由 |
|---|---|---|---|
| 理解支路（MASt3R encoder / adapter / mask decoder） | **允许** | **禁止**（`requires_grad=False` + no_grad 包裹重建损失路径） | 保护预训练配合不被重建目标破坏 |
| object tokens / 交互模块 | 允许 | 允许 | 它们是两组 token 的接口 |
| anchor/GS 解码器（重建） | **禁止**（规格要求） | 允许 | 避免再次出现 V3-SM 的 −2 dB 类代价 |
| `activation_head`（Gaussian 几何） | **禁止** | 允许 | 语义不得通过"挪 Gaussian"走捷径 |
| 允许集之内的共享参数 | 见上 | 见上 | — |

**注意**：仅 `gaussians.detach()` **不等于**隔离——`gaussian_child_features` 读 `μ/ρ`，`anchor_embedding` 也由 `μ/ρ/ell` 构成。因此上表是**逐参数**的，实现时必须用钩子/`requires_grad` 逐项落实，并附一个"梯度只应出现在允许集合内"的自动断言。

---

## 5. 初始化与权重映射

### 5.1 允许与禁止

| | 内容 |
|---|---|
| **允许** | MASt3R 图像编码器权重；**配套**的 panoptic 预训练 ViT-Adapter 与 Mask2Former（COCO 或 ADE20K 版）；LocusGS reconstruction pretrained（既有） |
| **禁止** | `siu3r_epoch100.ckpt`（已 ScanNet 任务训练）作为任何模块的初始化；把上述四类资产混称为一种 |

四类资产**分别标注**（评审要求）：
1. DUSt3R / **MASt3R** 的几何预训练 —— 提供几何/匹配能力；
2. DINO 类通用视觉预训练 —— 通用视觉表征；
3. **encoder + adapter + mask decoder 配套的 panoptic 预训练** —— 提供实例分类/分割能力（**本次要点**）；
4. 已在 ScanNet 上完成训练的 SIU3R 完整 checkpoint —— **禁用**。

### 5.2 映射表（骨架；具体张量名待权重到位后自动生成）

| 目标模块 | 源 | 动作 |
|---|---|---|
| understanding backbone | MASt3R 权重 | 全量加载（形状断言） |
| adapter（`spm` / `level_embed` / `interactions` / `norm1..4` / `up`） | panoptic 预训练 ckpt | 全量加载（**从 backbone 前缀剥离后**，与 `load_seg_ckpt` 同策略） |
| mask decoder（`model.*` / `class_predictor`） | 同上 | 全量加载；`class_predictor` 依据目标类别数决定是否重建 |
| object/mask query 初始化 | 同上（`queries_embedder`/`queries_features`） | 加载（保留预训练 query 语义） |
| LocusGS 重建路径 | `workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt`（既有，SHA `5fcf71b9…9634f`） | 既有传输逻辑不变 |
| 交互模块（新增） | 无预训练 | 零初始化（沿用 Joint 的做法） |

### 5.3 获取与校验规程（前置条件）

1. 从官方发布获取 §5.1 允许的三类权重；记录**来源 URL + SHA256**，落盘 `weights_provenance.json`。
2. **禁止**从 `siu3r_epoch100.ckpt` 里剥取 adapter/mask decoder 来"凑"——它已见过 ScanNet，会污染结构验证。（如坚持要用，必须单独标注并作为**不同**的实验臂。）
3. 每个加载步骤写形状断言 + 缺失/多余键清单，禁止 `strict=False` 静默通过。

---

## 6. 验证计划（在正式训练之前）

| 阶段 | 内容 | 通过标准 |
|---|---|---|
| V1 接口单测 | 形状/尺度/网格断言；`all_feat` 深度与宽度；adapter 输出 4 尺度 | 全部断言通过 |
| V2 参数级梯度审计 | 反向一次，列出**实际出现非零梯度的参数集合** | 与 §4.5 允许集**完全一致**，多一个即失败 |
| V3 控制臂逐位等价（M1 类比） | 理解支路断开（交互置零）时，重建前向与 LocusGS 基线**逐位一致** | `abs_diff == 0` |
| V4 活动性探针（M2 类比） | 40 步正式前向，检查交互项超出数值噪声包络 | 各注册层 `passed: true`（同 Joint 的 M2 形式） |
| V5 渲染一致性 | 新支路的合成权重近似 vs 渲染 alpha | MAE 与既有 0.0050–0.0058 同量级 |
| V6 预注册判据 | **主**：candidate 实例 + 最终 panoptic + 官方 AP/PQ + PSNR（照主协议，含地板值如实报告）；**副**：`raw mask IoU`、`classification accuracy`、`class-agnostic recall`、冻结特征探针 | 探针**只作副指标，不作为启动门槛** |

---

## 7. 阻塞项与开放问题

**阻塞**
1. **三类预训练权重缺失**（§1.3）。需要下载与校验；若环境无外网，需要用户侧提供。
2. 理解 backbone 的训练规格（输入分辨率、mean/std、位置编码实现）**待确认**——它决定 §4.2 的契约与 (A)/(B) 方案的选择。

**开放问题（请评审）**
1. **4.2 选 (A) 双分辨率前向 还是 (B) 统一 512×512？** 我建议 (A)（单一变量）。
2. **4.4 的归属机制选 (i)/(ii)/(iii) 哪个？** 我倾向 (i)（复用已验证的合成权重，可渲染、遮挡正确），但成本最高。
3. **类别空间如何对齐？** Mask2Former 的 `id2label` 是 COCO/ADE20K，ScanNet 是 20 类。是保留 COCO 标签空间并只取可用类，还是替换 `class_predictor`（会丢掉一部分预训练语义）？
4. **交互写回的目标**：只写 anchor tokens（Joint 的做法），还是也允许写 encoder 特征？
5. 是否需要一条**只在理解支路上**的辅助监督（ScanNet panoptic）？若是，它必须落在 §4.5 的允许集内。

---

## 8. 本次明确不做的事

- ❌ 启动新的正式训练（本次只交付规格与映射）
- ❌ 用 `siu3r_epoch100.ckpt` 初始化
- ❌ 把 adapter / mask decoder 从原 backbone 上拆下接任意 encoder
- ❌ 仅替换 DUSt3R encoder 而保留随机初始化的理解分支（评审明确禁止）
- ❌ 以探针表现作为接入设计的启动门槛（探针只作副指标）

---

## 9. 复现与出处索引

| 内容 | 路径 / 出处 |
|---|---|
| 我们的 encoder 接口 | `tokengs/options.py:43-44`、`tokengs/models/input_types.py:59-62`、`tokengs/models/spatial_grounded_tokens.py:89-99`、`tokengs/models/canonical_recon_models.py:146-152` |
| SIU3R 理解路径 | `/space/mawb/SIU3R/src/models/model.py`（`_set_adapter`/`_set_mask2former`/`load_seg_ckpt`）、`vit_adapter/vit_adapter.py:305+`、`mask2former/video_seg_decoder.py:2257+` |
| SIU3R backbone 返回契约 | `backbone_croco.py::AsymmetricCroCo.forward` → `(feat1, feat2, all_feat1, all_feat2, dec1, dec2, shape1, shape2)` |
| 权重现状 | `/space/mawb/SIU3R/pretrained_weights/`（仅 `siu3r_epoch100.ckpt`） |
| 合成权重验证（可复用） | `scripts/token_instance_compositing.py`（alpha MAE 0.0050–0.0058） |
| Joint 的非对称交互先例 | `docs/object_locus_joint_v1_codex_spec.md` §2、§9.3 |
