执行一个新的架构版本：

# `Anchor-Group V1`

本轮你的角色只是**严格执行以下 specification**。

你没有架构设计权、超参数选择权、实验决策权。

禁止根据你的判断：

- 修改结构；
- 替换 loss；
- 调整权重；
- 改 query 数；
- 改 warm-up；
- 改 LR；
- 冻结额外模块；
- 添加所谓“更合理”的模块；
- 运行正式训练；
- 做 ablation。

如果 specification 与当前代码接口存在冲突：

> 以“保持本 specification 的数学行为不变”为最高优先级，只做必要的接口适配，并在最终报告中逐项说明。

---

# 0. Baseline / provenance

开始于当前：

```text
main
HEAD = d023bb6
```

确认远端：

```text
origin/main == d023bb6
```

如果 HEAD 已前进：

- 不 reset；
- 先报告；
- 确认前进 commit 是否仅是用户文件/无关变更；
- 本架构仍以 `d023bb6` 当前模型逻辑为语义 baseline。

已有以下结果全部只读：

```text
group_plus/instance_state_v1_generalization/
group_plus/instance_state_v2_s1_local3d/
workspace_group_plus/instance_state_v1_generalization/
workspace_group_plus/instance_state_v2_s1_local3d/
```

禁止删除、覆盖、重写历史 S0/S1。

新目录固定：

```text
group_plus/anchor_group_v1/
workspace_group_plus/anchor_group_v1/
```

---

# 1. 本轮唯一目标

把当前：

```text
1024 reconstruction anchors
↓
FPS 100 seeds
↓
single/local8 evidence
↓
100 thing states
```

改成新的独立架构：

```text
1024 reconstruction anchors
      │
      │ 全部保留
      ↓
1024 anchor-state embeddings
      │
      │ soft ownership / grouping
      ↕
100 learnable thing queries
2 learnable stuff queries
1 non-query void channel
      ↓
anchor assignment A [B,1024,103]
      ↓
每个 Gaussian 继承父 anchor 的 assignment
      ↓
rendered semantic / instance / panoptic representation
```

严格理解：

```text
1024 anchor embeddings ≠ 1024 object queries
```

object query 数仍然固定：

```text
NUM_THING = 100
NUM_STUFF = 2
NUM_QUERIES = 102
VOID_INDEX = 102
NUM_REGION_CHANNELS = 103
```

---

# 2. 新架构必须完全移除 FPS/local8 information bottleneck

Anchor-Group V1 禁止在 object grouping path 中调用：

```python
deterministic_fps(...)
local_3d_evidence_pool(...)
```

不得从1024 anchors中先挑100个。

不得用：

```text
fps_index
local8
single-anchor seed
```

初始化 thing query。

已有 S0/S1 函数保留，旧 preset 行为必须不变。

Anchor-Group V1 新 path 必须能够在：

```python
deterministic_fps = function_that_raises
local_3d_evidence_pool = function_that_raises
```

的情况下正常 forward。

加入 contract 验证这一点。

---

# 3. 新模型命名

新增模型：

```text
LocusGSAnchorGroupRecon
```

architecture name：

```text
LOCUSGS_ANCHOR_GROUP_V1
```

新增文件：

```text
tokengs/models/anchor_group_locusgs.py
tokengs/models/anchor_group_loss.py
scripts/anchor_group_v1.py
docs/anchor_group_v1_spec.md
```

旧：

```text
instance_state_locusgs.py
instance_state_loss.py
instance_state_s1_local3d.py
```

除必要 import / registry 外，不修改其算法行为。

---

# 4. Anchor embedding 定义

在原 state layer：

```text
6 / 8 / 10 / 12
```

仍使用全部 decoder anchor tokens。

对任意 state layer：

```python
a = controller.encode_token(tokens, mu, radii, ell)
```

要求：

```text
a.shape == [B,1024,256]
```

这就是本版本所说的：

```text
anchor-state embeddings
```

不再把 `a` FPS 到100个。

全部1024 anchors进入 grouping。

---

# 5. Object queries 的初始化

thing/stuff queries仍然是 learnable：

```python
query_init: [102,256]
```

layer6 初始：

```python
q = query_init.unsqueeze(0).expand(B,-1,-1)
```

不得加：

```text
single anchor feature
local8 feature
FPS feature
mean anchor feature
```

也就是说：

```text
q_thing_init = learned query only
q_stuff_init = learned query only
```

1024 anchor information通过后续 anchor↔query grouping interaction进入 query。

---

# 6. Void

Void保持：

```text
no learnable query
```

每个 anchor 使用：

```python
void_logit = controller.token_void(a)
```

得到：

```text
[B,1024,1]
```

因此 ownership channel：

```text
100 thing
+ 2 stuff
+ 1 void
= 103
```

---

# 7. Anchor-query assignment

新增：

```python
AnchorGroupController.assign_group(...)
```

不要调用旧的带 `c/s` spatial tightness 的：

```python
InstanceStateController.assign(...)
```

Anchor-Group V1 assignment **不使用显式 c/s distance penalty**。

原因：

- 1024 anchor embedding 自身已经通过 `encode_token` 包含 `mu/ell` 与 radius 信息；
- 本版本要让 ownership 由完整 anchor representation + query interaction学习；
- 禁止重新引入 FPS proposal prior。

定义：

```python
e = normalize(proj_e(LN(a)))
u = normalize(proj_u(LN(q)))
```

query logits：

```python
group_logits = einsum("btd,bqd->btq", e, u) / 0.1
```

其中：

```text
q = 102 queries
```

然后：

```python
logits = cat([
    group_logits,      # [B,1024,102]
    void_logit         # [B,1024,1]
], dim=-1)
```

ownership：

```python
A = softmax(logits, dim=-1)
```

要求：

```text
A.shape = [B,1024,103]
sum(A, dim=-1) == 1
```

固定：

```text
temperature = 0.1
```

本轮禁止温度调参。

---

# 8. Anchor→query state update

新增：

```python
AnchorGroupController.update_group_states(...)
```

每个 state layer严格执行：

### 8.1 First assignment

```python
A_pre = assign_group(a, q, void_logit)
```

### 8.2 Normalize ownership per query

```python
mass = A_pre[:, :, :102].sum(dim=1)             # [B,102]

w = A_pre[:, :, :102] / (
    mass.unsqueeze(1) + 1e-6
)
```

### 8.3 Aggregate all 1024 anchor embeddings

```python
z = einsum("btq,btd->bqd", w, a)
```

要求：

```text
z.shape = [B,102,256]
```

### 8.4 Query update

严格复用当前已经验证过的：

```text
GRUCell
LN
FFN
LN
```

逻辑。

即与当前 `update_states()` 中：

```python
gru
ln_gru
ffn_fc1
ffn_fc2
ln_ffn
```

完全同构。

low-mass query：

```text
mass < 1e-4
```

保持 previous q。

### 8.5 Recompute ownership

使用更新后的：

```python
q_new
```

重新：

```python
A_post = assign_group(a, q_new, void_logit)
```

最终 state layer 输出：

```text
q
A_pre
A_post
anchor_embedding = a
```

---

# 9. c / s 在新架构中的角色

Anchor-Group V1 的 assignment **不使用 c/s**。

但是保留：

```text
c
s
```

作为：

- object geometry diagnostic；
- `token_message()` 所需的 group geometry summary；
- 未来可能的 coupling接口。

它们必须从：

```text
A_post
```

推导，而不是 FPS 初始化。

对 thing query：

```python
mass_thing = A_post[:, :, :100].sum(dim=1)

w_thing = A_post[:, :, :100] / (
    mass_thing.unsqueeze(1) + 1e-6
)
```

center：

```python
c = einsum("btq,btd->bqd", w_thing, mu)
```

variance：

```python
diff = mu.unsqueeze(2) - c.unsqueeze(1)

var = einsum(
    "btq,btqd->bqd",
    w_thing,
    diff ** 2
)
```

support：

```python
s = sqrt(
    var + (0.05 * ell)^2
)
```

clamp：

```text
min = 0.05 * ell
max = 2.0 * ell
```

不做旧的：

```text
0.5 previous + 0.5 new
```

因为新 architecture 没有 FPS-derived previous c/s。

---

# 10. state layers

保持：

```text
6
8
10
12
```

流程：

```text
decoder layer
↓
1024 current anchor tokens
↓
encode all 1024 → anchor embeddings
↓
102 learnable group queries interaction
↓
A_pre
↓
aggregate all anchors
↓
GRU/FFN update q
↓
A_post
↓
derive c/s
```

每个 layer 都处理 **1024 anchors**。

不得 downsample。

---

# 11. Legacy beta / E coupling 本版本禁用

Anchor-Group V1 的 joint learning：

> 不使用旧 E arm 的 beta write-back 作为主要梯度路径。

固定：

```text
coupled = False
beta = 0
```

若调用：

```text
coupled=True
```

Anchor-Group V1 直接：

```python
raise RuntimeError
```

提示：

```text
Anchor-Group V1 uses joint optimization through shared anchor/reconstruction
features and does not use the legacy beta coupling path.
```

Joint 的含义是：

> understanding losses 可以直接反向传播进入 reconstruction / anchor-producing network。

不是 legacy E write-back。

---

# 12. Gaussian ownership：必须继承 parent anchor

这是本架构的重要 contract。

当前 S0/S1 在 Gaussian level 又重新：

```text
Gaussian embedding
→ assign(...)
```

Anchor-Group V1 禁止这样重新 grouping。

最终：

```text
A_anchor = final["A_post"]
```

shape：

```text
[B,1024,103]
```

假设每个 anchor decode：

```text
P Gaussians
```

则：

```python
A_g = (
    A_anchor
    .unsqueeze(2)
    .expand(B, T, P, 103)
    .reshape(B, T * P, 103)
)
```

因此：

> 一个 anchor 产生的所有 child Gaussians 完全继承父 anchor 的 group ownership。

必须 exact equal。

不得再用：

```python
ctrl.assign(e_gs, xyz, ...)
```

重新决定 Gaussian group。

---

# 13. identity render 保留

当前：

```text
e_gs
identity_render
identity pull/push
```

暂时保留。

`e_gs` 仍然可以由：

```text
anchor feature
+ Gaussian local residual
+ offset
```

得到。

render feature：

```python
cat([
    A_g,
    e_gs
], dim=-1)
```

保持。

不要删除 identity loss。

本轮只改变 object grouping主路径。

---

# 14. Semantic readout

保持当前语义定义：

```text
channel 100 → wall
channel 101 → floor

thing channels 0..99
×
thing query class probability
→ classes 2..19
```

thing classifier仍为：

```text
18 thing classes + no-object
= 19 logits
```

保持。

---

# 15. Direct anchor-domain GT grouping supervision

这是 Anchor-Group V1 的核心新增 supervision。

新增：

```text
tokengs/models/anchor_group_loss.py
```

禁止修改旧 S0/S1 loss 的数学定义。

---

# 16. Thing GT construction

对两个 context views：

```text
semantic_label_all[:, :2]
instance_label_all[:, :2]
```

构造 deterministic GT thing list。

每个 scene：

1. 找所有：

```text
semantic 2..19
instance_id > 0
```

2. 按：

```text
instance_id ascending
```

排序。

3. 每个 instance 必须对应唯一 semantic class。

得到：

```text
gt_instance_ids [K]
gt_classes      [K]
gt_pixel_masks  [K,2,H,W]
```

如果：

```text
K > 100
```

直接 fail closed。

---

# 17. Anchor GT affiliation

使用最终 state layer 的：

```text
final["mu"].detach()
```

即：

```text
[B,1024,3]
```

投影到两个 context views。

投影公式必须复用/严格等价于当前已通过 parity contract 的：

```text
scripts/token_instance_compositing.py
```

以及：

```text
anchor_proposal_failure decomposition
```

的相机 convention。

不得自行换 coordinate convention。

---

# 18. Anchor observation rule

对每个 anchor、每个 context view：

只有：

```text
z > 0
u in [0,W)
v in [0,H)
semantic in [0,19]
```

才算有效 observation。

pixel index使用当前已验证行为：

```python
u.long()
v.long()
```

禁止本轮改成 bilinear / round。

---

# 19. Multi-view consensus rule

对同一个 anchor收集两个 view 的有效 observation。

## Case A：thing

若所有有效 observation 都是：

```text
thing class 2..19
instance_id > 0
```

并且全部是**同一个**：

```text
(semantic_class, instance_id)
```

则：

```text
anchor_kind = thing
anchor_instance_id = instance_id
anchor_semantic_class = class
```

---

## Case B：stuff

若所有有效 observation 都是：

```text
wall
```

则：

```text
anchor_kind = stuff_wall
```

若全部：

```text
floor
```

则：

```text
anchor_kind = stuff_floor
```

---

## Case C：conflict

以下任一种：

```text
different thing instance IDs
different semantic classes
thing vs wall/floor
wall vs floor
```

则：

```text
anchor_kind = IGNORE
```

---

## Case D：无有效 observation

```text
IGNORE
```

---

# 20. 不监督 projected void

以下 anchors：

```text
outside views
semantic=255
conflict
no valid observation
```

全部：

```text
IGNORE
```

本轮不把它们硬监督成 void。

Void仍通过现有 image/render supervision学习。

---

# 21. Anchor GT matrix

对 K 个 GT thing instances，生成：

```text
Y_anchor [K,1024]
```

其中：

```text
1 = anchor confidently affiliated with该GT instance
0 = 其它 confidently supervised anchor
```

另生成：

```text
anchor_valid [1024]
```

只包含：

```text
confident thing
wall
floor
```

IGNORE不参与 anchor grouping loss。

---

# 22. Unified Hungarian matching

Anchor-Group V1 不允许：

```text
2D rendered mask loss
```

和：

```text
anchor grouping loss
```

分别做不同 Hungarian。

必须使用**一次 unified Hungarian**。

每个：

```text
100 thing queries
vs
K GT thing instances
```

匹配 cost：

\[
C =
C_{class}
+5C_{pixel-bce}
+5C_{pixel-dice}
+2C_{anchor-bce}
+2C_{anchor-dice}
\]

固定：

```text
class weight       = 1
pixel BCE cost     = 5
pixel Dice cost    = 5
anchor BCE cost    = 2
anchor Dice cost   = 2
```

禁止修改。

---

# 23. Pixel matching term

严格复用当前：

```text
instance_state_loss.py
```

里的：

```text
matching class term
pixel BCE cost
pixel Dice cost
4096 deterministic sampled points
```

数学行为。

不要重新设计。

---

# 24. Anchor matching term

predicted：

```text
P_anchor = final A_post[:, :, :100]
```

转换：

```text
query-major [100,1024]
```

对：

```text
anchor_valid
```

位置计算。

对于 GT k：

```text
Y_anchor[k]
```

如果该 GT：

```text
positive anchor count == 0
```

则：

```text
anchor BCE matching cost = 0
anchor Dice matching cost = 0
```

只依赖 pixel/class matching。

不得因为 center proxy 未覆盖就删除该 GT。

---

# 25. Matched query 2D losses

Hungarian完成后：

当前 rendered thing losses保持：

```text
CLASS CE weight = 2
pixel BCE       = 5
pixel Dice      = 5
```

unmatched query：

```text
no-object class
weight = 0.1
```

保持。

---

# 26. Direct anchor ownership CE

Hungarian后建立每个 confident anchor 的 target channel。

### Thing anchor

若其：

```text
instance_id = gt k
```

且 Hungarian：

```text
gt k ↔ query j
```

则：

```text
target_channel = j
```

### Wall

```text
target_channel = 100
```

### Floor

```text
target_channel = 101
```

IGNORE：

不参与。

计算：

```python
L_anchor_ce = mean(
    -log(
        A_post[t, target_channel]
        .clamp_min(1e-6)
    )
)
```

---

# 27. Anchor Dice

只针对 matched thing instances。

对每个：

```text
query j ↔ gt k
```

若：

```text
Y_anchor[k].sum() > 0
```

计算：

```text
pred = A_post[:, j]
target = Y_anchor[k]
```

仅在：

```text
anchor_valid
```

范围：

\[
1 -
\frac{2\sum py+1}
{\sum p+\sum y+1}
\]

然后 matched instances平均。

没有 anchor-positive GT：

不计该项。

---

# 28. Anchor group loss

固定：

```python
L_anchor_group = L_anchor_ce + L_anchor_dice
```

不要额外加 embedding pull/push。

当前 identity loss已经存在。

本轮禁止再新增：

```text
contrastive anchor embedding loss
triplet
InfoNCE
center loss
```

---

# 29. Understanding total loss

保持已有：

```text
L_thing_2d
L_stuff_2d
L_semantic
L_identity
```

新增：

```text
L_anchor_group
```

固定：

```python
L_understanding = (
    0.1 * L_thing_2d
    + 0.1 * L_stuff_2d
    + 0.1 * L_semantic
    + 0.01 * L_identity
    + 0.1 * L_anchor_group
)
```

禁止调权。

输出 metrics：

```text
loss_anchor_group
anchor_ce
anchor_dice
anchor_valid_count
anchor_thing_count
anchor_wall_count
anchor_floor_count
anchor_ignore_count
gt_with_anchor_support
gt_without_anchor_support
```

---

# 30. Reconstruction loss

完全使用现有：

```text
canonical reconstruction loss
```

所有 supervised layers、RGB、SSIM、visibility 等定义保持现有 preset。

不要新增 reconstruction loss。

不要修改 reconstruction weights。

---

# 31. Joint + warm-up：正式训练定义先写进 driver，但本轮禁止运行

新增：

```text
scripts/anchor_group_v1.py
```

正式训练策略固定如下。

## 31.1 Reconstruction 从 step 1 开始始终开启

```python
L_recon weight = 1.0
```

---

## 31.2 Understanding warm-up

定义：

```python
def understanding_weight(step):
    if step <= 200:
        return 0.0
    if step < 1000:
        return (step - 200) / 800.0
    return 1.0
```

即：

```text
step 0      → 0
step 200    → 0
step 600    → 0.5
step 1000   → 1
step >1000  → 1
```

final：

```python
loss = L_recon + understanding_weight(step) * L_understanding
```

禁止继续使用旧：

```text
0.2 → 1 over 200
```

作为 Anchor-Group V1 的主 schedule。

---

# 32. Joint training的梯度定义

正式 joint training 时：

**不能 freeze reconstruction backbone。**

所有 model parameters：

```text
requires_grad = True
```

除非原 canonical reconstruction preset 本来就有**结构性固定参数**。

不得调用：

```python
freeze_backbone(...)
```

不得以“稳定”为由冻结 encoder/decoder。

---

# 33. Differential LR

未来正式训练的 optimizer固定为 AdamW：

```text
betas = (0.9,0.95)
```

两类主 LR：

### Group / understanding parameters

所有：

```text
instance_state.*
```

以及 Anchor-Group V1 新增 grouping learnable parameters：

```text
peak LR = 1e-4
```

### Reconstruction parameters

其余 pretrained reconstruction model parameters：

```text
peak LR = 1e-5
```

即：

```text
10× LR ratio
```

---

# 34. Weight decay

保持当前规则：

普通 matrix/weight：

```text
weight_decay = 0.05
```

以下 no-decay：

```text
1D parameters
bias
LayerNorm parameters
query_init
已有 _no_weight_decay=True 参数
```

不要自行改变。

---

# 35. LR schedule

未来正式训练预注册：

```text
5000 steps
```

LR multiplier：

### step 1..200

linear warm-up：

```python
m = step / 200
```

### step 201..5000

cosine：

```text
peak → 0.02 * peak
```

因此：

group：

```text
peak 1e-4
floor 2e-6
```

reconstruction：

```text
peak 1e-5
floor 2e-7
```

本轮只实现，不执行5000 step。

---

# 36. 为什么 beta=0 仍然是 joint

在 docs明确写：

理解 loss：

```text
anchor ownership
rendered masks
semantic
identity
```

全部依赖：

```text
reconstruction anchor tokens
mu
radii
Gaussian render
```

因此 reconstruction backbone unfrozen 时：

```text
understanding gradients
→ anchor-producing reconstruction network
```

这已经是真正 joint optimization。

不需要 legacy beta write-back。

---

# 37. Training data specification

未来正式训练继续使用已有：

```text
128 scenes
1024 windows
```

locked manifest：

```text
group_plus/instance_state_v1_generalization/train128_windows1024.json
```

SHA：

```text
1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483
```

training plan：

```text
plan_C_frozen_5000.json
```

只借用：

```text
window ordering
```

不要借用 frozen optimizer。

这样未来 S1 与 Anchor-Group V1 数据顺序一致。

---

# 38. 旧模型 regression contract

任何新改动不能破坏：

```text
S0
S1
```

至少重新执行已有：

```text
instance_state eval contracts
S1 local3d helper contracts
```

旧 preset：

```text
instance_state_local3d=False
instance_state_local3d=True
```

行为必须正常。

不要重新训练。

---

# 39. Anchor-Group CPU contracts

新增：

```text
scripts/check_anchor_group_v1_contract.py
```

必须全部通过。

至少包含以下 contracts。

---

## C1 shape

synthetic：

```text
B=1
T=1024
D=256
Q=102
```

要求：

```text
anchor embedding [1,1024,256]
q                [1,102,256]
A_pre            [1,1024,103]
A_post           [1,1024,103]
c                [1,100,3]
s                [1,100,3]
```

---

## C2 assignment simplex

```text
A >= 0
sum(A,-1) == 1
```

误差：

```text
<=1e-6
```

---

## C3 no FPS

monkeypatch：

```text
deterministic_fps
local_3d_evidence_pool
```

为直接 raise。

Anchor-Group forward必须仍通过。

---

## C4 all-anchor participation

构造 differentiable anchor embedding。

loss：

```python
loss = -torch.log(
    A_post[:, :, 0].clamp_min(1e-6)
).mean()
```

backward。

要求：

```text
1024/1024 anchor embedding rows
```

收到 nonzero finite gradient。

---

## C5 query gradient

同一个 loss要求：

```text
query_init
```

收到 finite nonzero gradient。

---

## C6 Gaussian inheritance

对随机：

```text
A_anchor
```

repeat到 Gaussian。

对每个 anchor：

```text
所有 child Gaussian assignment
==
parent anchor assignment
```

使用：

```python
torch.equal
```

---

## C7 target same-instance consensus

两个 view：

```text
chair instance 7
chair instance 7
```

→ thing `(class,7)`。

---

## C8 one-view visible

view0：

```text
chair instance 7
```

view1：

```text
outside / invalid
```

→ thing7。

---

## C9 conflicting instance

```text
view0 chair#7
view1 chair#8
```

→ IGNORE。

---

## C10 thing/stuff conflict

```text
view0 chair#7
view1 wall
```

→ IGNORE。

---

## C11 wall

两个/一个有效 view wall：

```text
stuff channel100
```

---

## C12 floor

→：

```text
stuff channel101
```

---

## C13 no observation

→ IGNORE。

---

## C14 perfect anchor assignment

synthetic GT + matching。

构造 perfect：

```text
A_post
```

要求：

```text
anchor CE ≪ permuted assignment anchor CE
anchor Dice ≪ permuted Dice
```

并 finite。

---

## C15 Hungarian consistency

构造2个 GT thing +3 query：

要求 unified Hungarian：

- class/pixel/anchor 三项共同只做一次 matching；
- 返回唯一一对一 assignment；
- pixel loss与anchor loss使用同一 matched pairs。

---

# 40. Real-batch target audit

从 locked train manifest固定：

```text
window index 0
```

真实 forward。

输出：

```text
group_plus/anchor_group_v1/real_batch_anchor_target_audit.json
```

记录：

```text
1024 total anchors
confident thing anchors
wall anchors
floor anchors
ignored anchors
GT thing count
GT with >=1 confident anchor
GT with 0 confident anchor
per-GT anchor count
min/median/p90/max
```

不要用这个结果筛数据。

---

# 41. Reconstruction parity contract

在任何训练之前：

同一 pretrained checkpoint、同一真实 batch。

比较：

```text
canonical/pre-existing reconstruction path
vs
Anchor-Group V1 beta=0
```

必须检查：

```text
Gaussian tensor
RGB render
PSNR
```

因为 grouping不应该在初始化时改变 reconstruction forward。

目标：

```text
gaussian max_abs_diff <= 1e-6
RGB max_abs_diff <= 1e-6
```

如果由于正常 floating implementation order不能 exact，必须报告真实误差。

若：

```text
>1e-5
```

视为 blocker。

不得通过放宽到大容差解决。

---

# 42. Joint gradient contract

在真实 batch 上：

```text
understanding_weight = 1
```

只 backward：

```text
L_understanding
```

必须证明 finite nonzero gradient进入：

### A

```text
instance_state.query_init
```

### B

至少一个：

```text
late decoder block parameter
```

### C

至少一个：

```text
anchor position/refinement parameter
```

### D

至少一个：

```text
reconstruction feature-producing parameter
```

具体参数名全部写入 audit。

这证明：

> understanding supervision真正进入 reconstruction representation。

---

# 43. Recon gradient contract

单独：

```text
L_recon.backward()
```

确认 reconstruction parameters正常收到 gradient。

不要要求 query_init 收 recon grad。

---

# 44. Warm-up numerical contract

必须 assert：

```text
uweight(0)    = 0
uweight(1)    = 0
uweight(200)  = 0
uweight(600)  = 0.5
uweight(1000) = 1
uweight(5000) = 1
```

LR ratio任意训练 step：

```text
group LR / reconstruction LR == 10
```

误差允许浮点范围。

---

# 45. GPU one-step smoke

这不是正式训练。

仅执行：

```text
1 real training step
```

B=当前真实 B=1。

全模型按正式 joint setting：

```text
reconstruction trainable
group modules trainable
```

使用真正：

```text
L_recon
+
warmup-compatible understanding
```

但为了验证joint梯度，另有 C42。

GPU smoke记录：

```text
GPU model
allocated before
reserved before
peak allocated
peak reserved
forward finite
loss finite
backward finite
optimizer.step finite
```

并报告：

```text
max_memory_allocated GiB
max_memory_reserved GiB
```

---

# 46. OOM规则

如果3090 24GB one-step smoke OOM：

**立即停止。**

不得自行：

- freeze reconstruction；
- 降图像尺寸；
- 改 batch；
- gradient accumulation；
- mixed precision；
- checkpointing；
- 减 query；
- 减 anchor；
- 改 optimizer。

只记录：

```text
OOM位置
CUDA error
执行到哪一步
```

仍可以提交实现代码和 audit。

不要正式训练。

我们之后再决定 memory策略。

---

# 47. 本轮严禁正式训练

绝对禁止执行：

```text
5000-step
1000-step
200-step
任何多-step实验
```

允许：

```text
CPU contracts
real-batch inference
loss audit
gradient audit
1-step GPU smoke
```

仅此而已。

---

# 48. 不运行 official benchmark

本轮不需要：

```text
SIU3R official mAP/PQ
qualitative
S0/S1 re-eval
```

目标只是：

> 架构实现正确 + loss正确 + joint gradient正确 + 不OOM。

---

# 49. Report

生成：

```text
group_plus/anchor_group_v1/implementation_audit.md
group_plus/anchor_group_v1/contracts.json
group_plus/anchor_group_v1/real_batch_anchor_target_audit.json
group_plus/anchor_group_v1/reconstruction_parity.json
group_plus/anchor_group_v1/joint_gradient_audit.json
group_plus/anchor_group_v1/gpu_one_step_smoke.json
```

`implementation_audit.md` 必须明确回答：

1. 是否完全去掉 FPS/local8 bottleneck；
2. 是否1024 anchors全部参与；
3. query数量是否100 thing +2 stuff；
4. void是否非query；
5. A shape是否1024×103；
6. Gaussian是否继承parent anchor ownership；
7. GT anchor affiliation如何构造；
8. conflicting projection如何处理；
9. unified Hungarian是否只做一次；
10. anchor CE/Dice是否使用相同 matching；
11. understanding gradient是否进入 reconstruction backbone；
12. beta是否固定0；
13. warm-up是否精确；
14. one-step显存是否能在3090执行。

---

# 50. Commit

所有 contract 通过后（或仅 GPU OOM，但实现 contracts通过）：

创建**一个独立 commit**：

```text
feat: add 1024-anchor group queries with joint grouping supervision
```

包含：

```text
new model
new loss
new driver
options / registry
contracts
docs spec
implementation audits
```

禁止提交：

```text
workspace checkpoint
大模型权重
临时 cache
```

---

# 51. Push

必须：

```text
git push origin main
```

普通 push。

禁止：

```text
force
amend已有历史commit
rebase已push历史
```

---

# 52. Push之后停止

Push完成后：

**绝对不要启动正式训练。**

等待用户/ChatGPT检查GitHub代码。

---

# 53. 最终只回复以下内容

1. final commit SHA
2. push是否成功
3. 修改/新增文件列表
4. 新 architecture 名
5. 1024 anchors是否全部参与 grouping
6. FPS/local8是否完全从新 path 移除
7. assignment shape
8. Gaussian parent-assignment inheritance是否 exact pass
9. GT anchor target统计：
   - thing
   - wall
   - floor
   - ignore
   - GT with/without anchor support
10. unified Hungarian contract是否pass
11. reconstruction parity：
   - Gaussian max diff
   - RGB max diff
12. understanding gradient实际进入哪些 reconstruction parameter
13. warm-up contract是否全绿
14. one-step GPU：
   - pass/OOM
   - peak allocated
   - peak reserved
15. 所有CPU/GPU contracts pass/fail
16. 明确写：
   `正式训练未启动`

不要提出替代架构。
不要建议下一实验。
不要自行启动训练。