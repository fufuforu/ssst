# MASt3R + 配套 panoptic 预训练理解路径：接入规格 **Rev 2**（按评审决定修订）

> 面向：GPT（评审）→ Codex（实现）。
> **Rev 2 变更摘要**：(1) 五个开放问题按评审决定收敛（§0.2）；(2) **§4.5 梯度边界重写为真正的端到端联合训练**（原写法自相矛盾，已废）；(3) **两处 adapter 接口事实纠正**（`all_feat[indexes]` 与 `add_vit_feature` 直加路径，§1.2/§2）；(4) 权重收敛为两份并区分"实际加载"与"重新初始化"（§5）；(5) 验证计划按评审简化（§6）；(6) 权重获取受集群出网白名单限制，改由用户侧下载（§5.4）。
> 状态：仅设计。**不启动正式训练。**

---

## 0. 范围与已定决策

### 0.1 固定方向

| 部分 | 固定方向 |
|---|---|
| 现有重建路径 | **保留** LocusGS reconstruction pretrained、GS/anchor tokens、Gaussian 生成 |
| 新图像理解路径 | 接入 **MASt3R 图像特征 + 配套 panoptic 预训练 adapter / mask decoder** |
| Object／mask tokens | 从预训练理解路径读取图像证据，**保留其可复用预训练能力** |
| 两组 token 联系 | reconstruction tokens 与 object tokens 的**双向交互，参与生成过程** |
| 最终输出 | 几何、外观与实例 membership **落在同一组 3D Gaussians 上** |
| 初始化边界 | **不使用**已 ScanNet 任务训练的 `siu3r_epoch100.ckpt` |
| 本次工作范围 | 完成**接入规格 + 权重映射**；**不启动正式训练** |

### 0.2 评审已定的五个选择（Rev 2 采纳）

| 决策项 | 固定选择 |
|---|---|
| **分辨率** | 理解支路 **512×512**，重建维持 **256×256**；**使用同一 context 图像与同一裁剪范围**（两分支看到相同内容） |
| **Gaussian membership** | **遮挡感知聚合预训练 pixel decoder 的稠密 mask features**，再与 **object query 的 mask embedding 做点积**、sigmoid（详见 §4.4） |
| **类别空间** | 使用 **COCO panoptic 预训练**；**保留** adapter、pixel decoder、query decoder、query embeddings、mask embedding；**重新初始化 ScanNet 分类输出** |
| **Object 写回** | 只写**中间注册层的 anchor tokens**；**不写 encoder 特征、不改 μ/ρ** |
| **额外理解监督** | 首版**不加** 2D 辅助监督；沿用最终 Gaussian 渲染结果上既有的集合预测分类与 thing/stuff mask 监督 |

**membership 决策的理由（评审原文要点）**：不采用"把 2D mask 概率加权平均作为最终 membership"。
- 合成贡献负责"**这个 Gaussian 能读取哪些图像证据**"；
- query 点积负责"**它属于哪个实例**"；
- 直接反投影 2D mask 概率会把理解任务退化为**搬运已有 2D 预测**，且反投影后再渲染并不天然恢复原 mask。
- **低可见 / 不可见 Gaussian 必须保留由自身 child features 预测 membership 的路径**，不得因缺少 context 合成贡献而直接赋零或 void。

---

## 1. 双方接口的实测事实（含 Rev 2 的两处纠正）

### 1.1 我们这边（LocusGS / TokenGS，重建路径）

| 事实 | 值 / 出处 |
|---|---|
| 图像尺寸 / patch | `img_size=(256,256)`、`patch_size=8` ⇒ 32×32 patch/视图（`options.py:43-44`） |
| 上下文视图数 | 2 ⇒ 2048 token |
| encoder 输出形式 | `EncoderLatent.keys/values: [B, H_heads, N, C//H_heads]` —— **已是投影后的 attention K/V**（`input_types.py:59-62`） |
| 实测形状 | `values = (1, 16, 2048, 64)` |
| 几何对齐 | `patch_plucker_rays(...)` 池化稠密 Plücker ray 到 patch 网格，**view-major → row-major**（`spatial_grounded_tokens.py:89-99`） |
| 消费点 | `LocusGSRecon.forward` → `get_gs_tokens(batch_size, encoder_latent, patch_rays)` |

### 1.2 理解路径原本接收什么（**Rev 2 纠正两处**）

```
model.py:
  _set_backbone()    → AsymmetricCroCo(CroCoNet)      # patch_size=16, d_out=1024
  _set_adapter()     → CroCoViTAdapter(num_block=enc_depth, embed_dim=enc_embed_dim,
                        size=image_size, patchsize=croco.patch_size,
                        interaction_indexes=[5,11,17,23], with_cffn=True,
                        deform_ratio=0.5, add_vit_feature=True, use_extra_extractor=True)
  _set_mask2former() → VideoMask2FormerForVideoSegmentation(Mask2FormerConfig(id2label, num_queries))

forward:
  feat1, feat2, all_feat1, all_feat2, dec1, dec2, shape1, shape2 = self.backbone(...)
  multi_scale_feat1 = self.adapter(img1, all_feat1)      # 逐视图
  multi_scale_feat  = [stack([f1, f2], dim=1) for f1, f2 in zip(msf1, msf2)]
  context_seg_output = self.mask2former(multi_scale_feat=multi_scale_feat, ...)
```

**纠正 ①：adapter 需要完整的层输出列表，不是长度 4 的列表。**
官方实现里逐个 interaction 按**索引**取用：
```python
indexes = self.interaction_indexes[i]      # [5, 11, 17, 23]
x = all_feat[indexes]
```
⇒ 若只传 4 层特征，**必须同时改写索引接口**。**首版选择保留原接口**（传完整层输出列表）。
（`CroCoViTAdapter.__init__` 另有 `self.H = size[0]//patchsize`、`self.W = size[1]//patchsize`、`level_embed(3, embed_dim)`、`spm = SpatialPriorModule(inplanes=64, embed_dim)`、`up = ConvTranspose2d(embed_dim, embed_dim, 2, 2)`、`norm1..norm4 = SyncBatchNorm(embed_dim)`。）

**纠正 ②：存在"ViT 特征 → 输出"的直接融合路径。**
`add_vit_feature=True` 时，四层 ViT 特征被**插值后直接加到四尺度空间特征上**。
⇒ 我上一版说的"ViT 特征到输出没有直接通道、全靠交互"**不准确，已删除**。这条**直接融合路径必须一并保留**。

### 1.3 权重现状

`/space/mawb/SIU3R/pretrained_weights/` 目前只有 `siu3r_epoch100.ckpt`（5.46 GB，**按边界禁用**）。两份必需资产**缺失**（获取路径见 §5.4）。

---

## 2. 为什么这套配合必须整体保留（Rev 2 修订论证）

panoptic 能力存在于**四个要素的配合**：

```
① ViT 的【四个指定深度】特征 all_feat[l], l ∈ {5,11,17,23}
        （宽度 = enc_embed_dim；由 interaction_indexes 固化，且索引进入完整列表）
② 【原始图像】 → SpatialPriorModule（1/4,1/8,1/16 CNN 先验）+ level_embed(3)
        （deformable interaction 在 ② 内逐层融合 ① 的语义与 ② 的空间细节）
③ 【四尺度输出】 = 交互结果 + 【直接相加】的插值 ViT 特征（add_vit_feature=True）
        ↓ （通道 / 步长布局固定）
④ Mask2Former 的 pixel decoder + query decoder + query embeddings + mask embedding
        → object/mask queries + per-query mask logits（查询与分类头同样是配套预训练的）
```

**Rev 2 修正后的表述**：①→③ 既有**交互路径**也有**直接相加路径**（两者都要保留）；破坏任一项都会使预训练配合失效：
- 换掉 ViT 或改用**投影后的 K/V** ⇒ ① 的深度语义与宽度对不上，且 4 个深度丢失；
- 去掉/改动 `SpatialPriorModule` 的输入 ⇒ ② 的统计量与 `norm1..4`、`level_embed` 训练分布不一致；
- 改变 patch 网格（`self.H/self.W`）或输入分辨率 ⇒ 形变采样位置语义整体偏移；
- 只搬 mask decoder ⇒ 它期望 ③ 的具体多尺度布局。

---

## 3. 不兼容清单

| 维度 | 我们（重建路径） | SIU3R 理解路径 | 后果 |
|---|---|---|---|
| patch / 分辨率 | 8 @ 256×256 → 32×32 | 16 @ 512×512 → 32×32 | **token 数相同（1024/视图）纯属网格算术巧合**；每 token 覆盖范围差 2× |
| encoder 输出形式 | **1 个**投影后 K/V `[B,16,N,64]` | **完整层列表**，adapter 按索引 {5,11,17,23} 取 | 需要完整列表；我们只暴露 1 个且已投影 |
| 空间先验 | 无（几何靠 Plücker ray 偏置） | `SpatialPriorModule(原图)` + `level_embed(3)` | ② 环不能省 |
| 输出构成 | — | 交互 + **ViT 特征直接相加** | 直接融合路径需保留 |
| 位置/几何编码 | Plücker patch ray（view-major） | CroCo 位置编码，无 ray 输入 | 需重建位置对应 |
| 输出形态 | token → 64 child Gaussians → 渲染 | per-query **2D mask logits** + 稠密 mask features | 需 §4.4 的归属模块（可渲染） |

**结论**：理解支路必须**自己持有 ①+② 的原始输入**（完整层列表 + 原图），并以 **object/mask token 层**与重建 tokens 交互；**不能**把我们的 K/V 当 adapter 输入，也**不能**把理解支路压成我们的形状。

---

## 4. 接入设计

### 4.1 模块图

```
  同一 context 图像（同一裁剪范围）
        ├─────────────────────────────► 重建支路 256×256 / patch 8（LocusGS，保留）
        │                                   │  anchors/GS tokens (2048)
        │                                   ▼
        │                           reconstruction decoder + Gaussian 生成
        │
        └──► 理解支路 512×512 ──► MASt3R encoder ──► 完整层列表 all_feat
                                     │
                                     ├─► CroCoViTAdapter(img, all_feat)  ← 原接口 + 直接相加路径
                                     │        └─► 四尺度 multi-scale maps
                                     └─► VideoMask2Former
                                              ├─► object/mask query embeddings {q_obj, q_mask}
                                              └─► pixel decoder 稠密 mask features
                                                        │
                        object tokens ◄── 双向交互 ──► reconstruction anchor tokens
                                                        │
                        per-Gaussian membership（§4.4）◄┘
                                                        ▼
                        同一组 3D Gaussians → 渲染 → 既有集合预测分类 / thing-stuff mask 监督
```

重建支路**完全保留**；理解支路**新增并联**，只在 token 层交互。

### 4.2 分辨率与对齐（决策已定）

- 理解支路 **512×512 / patch 16**，重建支路 **256×256 / patch 8**；
- **同一 context 图像、同一裁剪范围**（两分支内容一致，只是重采样到各自分辨率）；
- 断言项：图像内容一致（同一 frame id + 同一 crop 参数）；各支路形状与各自配置一致；
- 位置对应：理解支路的 patch 网格 → 通过已知的 crop/缩放关系映射回重建支路的 patch 网格（实现时写显式函数 + 单测）。

### 4.3 双向交互

| 方向 | 现状 | 新设计 |
|---|---|---|
| reconstruction → object | **已有**：object 读 anchor 特征做 evidence | 改为**同时**读 anchor 特征与理解支路的 object/mask tokens |
| object → reconstruction | Joint 已有：token-only 残差写回（零初始化 `Linear`，β ramp） | 保留；写回内容来自**有预训练语义的 object tokens** |

约束：**只写中间注册层的 anchor tokens**；**不写 encoder 特征、不改 μ/ρ**；路由归一化轴固定；β ramp；两条交互路径都**不得 detach**。

### 4.4 Gaussian membership（按评审决策）

```
输入：
  理解支路 pixel decoder 的稠密 mask features   F_msd[q?, c, h, w]（或 feature map × mask embedding 形式）
  object query 的 mask embedding                q_mask[q, c]
  重建支路每个 child Gaussian 的自身特征        f_child[i, d]   （保留路径）
  遮挡感知合成贡献（context 视图）              c_i,v  （由 EWA 足迹 + 前向-后向透射率得到，作为【固定几何读取权重】）

聚合：对每个 Gaussian i，用 c_i,v 在 context 视图上聚合 F_msd 的稠密 mask features
        → 得到该 Gaussian 的"可读取图像证据"表示  F_i
预测：membership(i, q) = sigmoid( <F_i, q_mask[q]> / sqrt(d) )     （query–Gaussian 点积）
兜底：当 Gaussian i 在 context 中无合成贡献（低可见/不可见）时，
        membership 由 i 自身 child features 的预测路径给出（独立小头或与 f_child 的投影点积），
        **不得赋零、不得赋 void**。
输出：per-Gaussian membership [B, 65536, Q] → 经既有 alpha 合成器渲染 → 既有实例监督
```

**待 GPT 写死（评审已声明由其完成）**：具体融合公式、通道维度、零支持处理。
**实现时必须满足**：membership **可渲染**（经 alpha 合成器），不得退回投影中心判定（历史教训：投影式归属产生 62.8% 跨实例伪影；遮挡感知合成权重下 purity p50 = 1.000）。
**允许**：把合成权重当作**固定几何读取权重**用于图像特征聚合 —— 但这**不等于**截断理解任务对重建特征与生成过程的梯度（见 §4.5）。

### 4.5 梯度边界（**Rev 2 重写：首版为真正的端到端联合训练**）

> 上一版"理解支路允许语义梯度 + `requires_grad=False`"**自相矛盾，作废**。
> 也**不再**把历史 V3-SM 的约 −2 dB（成因未证明）转化为永久性结构禁令。

| 参数组 | 理解 loss | 重建 loss |
|---|---|---|
| 理解 encoder、adapter、query/mask decoder | **允许** | **允许**（沿 object 回写路径传入） |
| Object states、双向交互模块 | **允许** | **允许** |
| 重建 encoder、anchor decoder、geometry 与 activation head | **允许**（继续采用已定义的 **GC α=0.01** 缩放） | **允许** |

- 保留重建侧的温和学习率与 **GC α=0.01**。
- **新理解预训练参数的学习率单独确定**，不得与"随机新增模块"（零初始化交互模块）沿用同一学习率。
- 允许将合成权重作为固定几何读取权重使用，但这不是梯度截断。

### 4.6 监督（决策已定）

首版**不增加** 2D 辅助监督；沿用最终 Gaussian 渲染结果上**既有**的集合预测分类与 thing/stuff mask 监督。目的是**先验证预训练能力能否进入统一 3D 任务**，避免再增加一套监督变量。

---

## 5. 权重：两份资产 + 加载边界

### 5.1 固定使用的两份（评审已收敛，不用 DUSt3R / ADE20K 版本）

| 资产 | 官方来源 |
|---|---|
| MASt3R 图像编码器 | `https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` |
| panoptic 预训练（adapter + mask decoder） | `https://huggingface.co/datasets/insomnia7/SIU3R/blob/main/panoptic_coco_pretrain_vitadapter_maskdecoder_epoch60.ckpt` |

（来源：SIU3R `README.md` 第 84 行。）

### 5.2 加载边界（**必须区分"实际加载"与"重新初始化"**）

官方 `load_seg_ckpt()` **明确排除** `class_predictor`、`criterion`、`backbone`。因此映射表必须写成三栏：

| 目标 | 动作 | 说明 |
|---|---|---|
| adapter（`spm` / `level_embed` / `interactions` / `norm1..4` / `up`） | **全量加载**（从 backbone 前缀剥离后） | 预训练能力所在 |
| mask decoder（`model.*`：pixel decoder、query decoder、query embeddings、mask embedding） | **全量加载** | 预训练能力所在 |
| **分类输出**（`class_predictor` 等） | **重新初始化**为 ScanNet 目标类别数 | 官方亦排除；COCO 类别无法覆盖 ScanNet |
| MASt3R backbone | 全量加载 | 与 SIU3R 的 `load_recon_ckpt` 同类 |
| LocusGS 重建路径 | 既有传输逻辑不变（SHA `5fcf71b9…9634f`） | — |
| 交互模块（新增） | 零初始化 | 与 Joint 一致 |

禁止 `strict=False` 静默通过；每个加载步骤输出**缺失/多余键清单 + 形状断言**。

### 5.3 授权与禁止

- **允许**：MASt3R 编码器；配套 COCO panoptic 预训练 adapter + mask decoder；既有 LocusGS reconstruction pretrained。
- **禁止**：用 `siu3r_epoch100.ckpt`（已 ScanNet 任务训练）初始化任何模块；四类资产混称。

### 5.4 获取状态（**集群侧阻塞，已实测**）

实测出网为**白名单制**（登录节点与计算节点一致）：

| 端点 | 结果 |
|---|---|
| `pypi.tuna.tsinghua.edu.cn` | **200 ✓** |
| `mirrors.aliyun.com` | 301 ✓ |
| `modelscope.cn` | 302 ✓ |
| `huggingface.co` | **000 ✗** |
| `hf-mirror.com` | **000 ✗** |
| `download.europe.naverlabs.com` | **000 ✗** |
| ModelScope 上的 `AI-ModelScope/MASt3R`、`AI-ModelScope/SIU3R` 等 | **404 record not found** ✗ |

⇒ **两份权重无法从本集群获取**（HF 与 naverlabs 均被封锁）。需要**用户侧**下载后上传。落地后必须记录 `weights_provenance.json`（来源 URL、文件大小、SHA256、加载映射）。
**这不阻塞规格与接口设计**，只阻塞真实前向的实现与 V1–V5 验证。

---

## 6. 验证计划（Rev 2 按评审简化）

| 阶段 | 内容 | 通过标准（**放宽后**） |
|---|---|---|
| V1 接口单测 | 形状/尺度/网格断言；完整层列表；adapter 四尺度输出；两分支相同 content | 断言全过 |
| V2 梯度检查 | **禁止路径无梯度**；**关键允许路径有有限且非零梯度** | ✅ **不要求**每个允许参数在一次 backward 中非零（零初始化注入时上游暂时零梯度是正常现象） |
| V3 控制前向比较 | 与**同臂重复**的数值包络比较 | ✅ **不要求**渲染输出逐位一致 |
| V4 真实 batch smoke | **3090 单步 smoke + 一次短活动性检查** | 通过即可；**不追加探针链** |
| V5 归属性质量 | 仅作**诊断**记录 | ⚠️ 历史合成 MAE 0.0050–0.0058 **只是历史结果，不是新分辨率下的天然门槛** |
| V6 预注册判据 | 主：candidate 实例 + 最终 panoptic + 官方 AP/PQ + PSNR（照主协议，含地板值如实报告）；副：raw mask IoU、classification accuracy、class-agnostic recall、冻结特征探针 | 探针**只作副指标，不作启动门槛** |

---

## 7. 尚未定稿的四项（评审要求补齐后即可形成实施提示词）

| # | 待补 | 责任 |
|---|---|---|
| 1 | **精确权重键映射表**（含实际加载 vs 重新初始化的键清单） | 权重到位后由实现侧自动生成并人工核对 |
| 2 | **前向顺序**（重建支路与理解支路的调用次序、交互发生的层与时机） | 实现侧出草案 → 评审确认 |
| 3 | **Gaussian 特征融合公式**（§4.4：融合式、通道维度、零支持处理） | **评审（GPT）已声明由其写死** |
| 4 | **optimizer 分组**（新理解预训练参数单独学习率；交互模块零初始化单独处理） | 实现侧出草案 → 评审确认 |

---

## 8. 本次明确不做的事

- ❌ 启动新的正式训练（只交付规格与映射）
- ❌ 用 `siu3r_epoch100.ckpt` 初始化
- ❌ 把 adapter / mask decoder 从原 backbone 拆下接任意 encoder
- ❌ 仅替换 DUSt3R encoder 而保留随机初始化的理解分支
- ❌ 以探针表现作为接入设计的启动门槛
- ❌ 为"防止重建退化"而设置未证明的结构性梯度禁令

---

## 9. 出处索引

| 内容 | 路径 / 出处 |
|---|---|
| 我们的 encoder 接口 | `tokengs/options.py:43-44`、`tokengs/models/input_types.py:59-62`、`tokengs/models/spatial_grounded_tokens.py:89-99`、`tokengs/models/canonical_recon_models.py:146-152` |
| SIU3R 理解路径 | `/space/mawb/SIU3R/src/models/model.py`（`_set_adapter` / `_set_mask2former` / `load_seg_ckpt`）、`vit_adapter/vit_adapter.py:305+`、`mask2former/video_seg_decoder.py:2257+` |
| adapter 索引接口与直接相加路径 | `vit_adapter/vit_adapter.py`（`all_feat[self.interaction_indexes[i]]`；`add_vit_feature=True`）——Rev 2 纠正依据 |
| backbone 返回契约 | `backbone_croco.py::AsymmetricCroCo.forward` → `(feat1, feat2, all_feat1, all_feat2, dec1, dec2, shape1, shape2)` |
| 权重来源 | `SIU3R/README.md:84` |
| 合成权重验证（历史） | `scripts/token_instance_compositing.py`（MAE 0.0050–0.0058，**仅历史结果**） |
| Joint 非对称交互先例 | `docs/object_locus_joint_v1_codex_spec.md` §2、§9.3 |
