# Object-Locus Joint V1：Codex 中文实施提示词

本文件整体交给 Codex 执行。正文、决策表、合规映射表共同构成唯一实施规格。设计者已经完成所有科学与训练决策；Codex 是执行者，不自行选择架构、超参数、loss、数据或追加实验。

## 1. 任务、范围与固定起点

实现一个保留 V3-Set 实例监督和读出的生成耦合模型，并完成两个正式配对训练臂：

- `control`：交错前向，关闭 object→anchor 注入。
- `joint`：同一交错前向，开启 object→anchor token 注入。

两臂均从 reconstruction step47500 fresh transfer；object 分支和注入模块一起从零初始化。不得加载 V3 epoch64 或 V3 Expanded 的理解权重继续训练。

研究问题固定为：在相同 V3 监督、Gaussian child mask 表达和训练配方下，object states 参与 Gaussian 生成是否带来任务收益。不得将本轮称为“已实现严格实例局部 token”。

在spec/run manifest中保留第二轮评审P1给出的四条历史先验及其配置标签：V1无回写softmax/继承的低AP；G0/G0+广播归属的低IoU；旧recon-only贡献权重口径的purity中位数1；旧checkpoint中有效child少与footprint重叠。不得把这些数字标为当前V3测量，不当作容量上界，不重新启动这些旧实验。

### 1.1 基线、worktree 与文件边界

基线 worktree：`/space/mawb/ssst_object_locus_v3_set`；基线完整 SHA：`b00b94fdc45af6d2f78c55f05671aaa75906204f`。从此提交创建独立分支 `object-locus-joint-v1-exec`、独立 worktree `/space/mawb/ssst_object_locus_joint_v1`。

不得在 `/space/mawb/ssst` 的落后 main 上开发；不得修改、清理或暂存任何既有 dirty 文件；不得覆盖旧 run、checkpoint、报告；不得暂停正在运行的 V3 Expanded。

允许新增且仅新增：

1. `tokengs/models/object_locus_joint_v1.py`
2. `scripts/object_locus_joint_v1_runtime.py`
3. `scripts/train_object_locus_joint_v1.py`
4. `scripts/eval_object_locus_joint_v1.py`
5. `scripts/smoke_object_locus_joint_v1.py`
6. `scripts/submit_object_locus_joint_v1.sh`
7. `scripts/report_object_locus_joint_v1.py`
8. `scripts/replay_object_locus_joint_v1_failure.py`
9. `tests/test_object_locus_joint_v1_contracts.py`
10. `docs/object_locus_joint_v1_codex_spec.md`：保存本提示词及执行记录。

仅允许在 `tokengs/models/__init__.py`、`tokengs/options.py` 添加新模型注册和继承 V3 的配置项。禁止顺手重构。

禁止修改：任何 V1/V1.1/V2/V2.1/V3 科学文件；`canonical_recon_models.py`；`locusgs_recon.py`；现有 controller、loss、provider、renderer、V3 export/evaluator；SIU3R evaluator 或导出协议。新 runtime 可复制并有针对性适配 V3 runtime，但不能回改旧文件。

### 1.2 Pretrained 与 strict transfer

路径：`/space/mawb/ssst/workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt`。

使用代码中的真实 SHA256：

`5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f`

用户给出的 `5fcf71b9b01...` 与仓库值不同；本规格显式纠正此笔误。必须实测文件 hash 与上述64字符值一致。不得另选 checkpoint。

复用 `scripts/object_locus_v3_set_runtime.py::load_checkpoint_state` 的读取方式。先构造 `LocusGSRecon` canonical reference 并对 source strict-load；随后核对目标模型：

- reconstruction：450 个 state tensors，keys、shape、dtype 与 canonical reference 完全对应，逐项 copy。
- 既有 `object_locus_v3_set.*`：78 个新 state tensors，保持 V3 初始化和值。
- 新 `object_locus_joint_injection.*`：4 个新 state tensors，全部零。
- 不允许额外 missing/unexpected/shape mismatch；不得 `strict=False` 后忽略警告。

初始化 global seed=42。新模型继承 `LocusGSObjectLocusV3SetRecon`，在自己的 `__init__` 中直接调用 `LocusGSRecon.__init__(self,opt)`，建立新 decoder 子类，再在**同一个** `torch.random.fork_rng(devices=[])` 内执行 `torch.manual_seed(31415)`，依次建立原版 `ObjectLocusV3SetController(1024)` 和四层注入模块。不得先调用 V3 构造器创建 object 分支后又重复重建。注入模块在 object controller 之后构造，不改变既有78个 tensors 的初始化。两臂读取同一 fresh 初始化快照；optimizer fresh，各自独立。

## 2. 新子类、交错前向与所有张量合同

新增类：

- `ObjectLocusJointAnchorDecoder(LocusGSAnchorDecoder)`
- `LocusGSObjectLocusJointV1Recon(LocusGSObjectLocusV3SetRecon)`

模型注册 key：`siu3r_object_locus_joint_v1`；architecture name：`LOCUSGS_OBJECT_LOCUS_JOINT_V1`；options 从 `train_siu3r_object_locus_v3_set` evolve，只新增新模型类型及本任务路径，不改变原配置的科学值。

decoder 的 `forward_stateful(tokens, encoder_latent, patch_rays, controller, injection, *, enabled, step)` 实现在新文件中。参考 `instance_state_locusgs.py::InstanceStateDecoder.forward_stateful` 的交错循环范式，但数学运算顺序逐项照抄基线 `LocusGSAnchorDecoder.forward`，不照抄 GRU、compactness bias、FPS 初始化或其 loss。

### 2.1 Shape / dtype

全程 FP32，不使用 autocast/AMP，batch=1；计算设备沿用输入设备。

| 符号 | shape | 定义 |
| --- | --- | --- |
| h | `[B,1024,1024]` | reconstruction anchor tokens |
| μ | `[B,1024,3]` | anchor centers |
| ρ、r | `[B,1024]` | raw radius 与原 activated radius |
| a | `[B,1024,256]` | 原 controller anchor embedding |
| q | `[B,102,256]` | 100 thing + 2 stuff states |
| c、s | `[B,100,3]` | thing center/support，原版更新规则 |
| ell | `[B]` | L6 detached scene scale |
| R | `[B,8,102,1024]` | evidence attention，沿1024 anchors softmax |
| T | `[B,1024,102]` | 新消息路由，沿102 object states softmax |
| u | `[B,1024,256]` | 每个 anchor 接收的 object message |
| Δh | `[B,1024,1024]` | token 注入残差 |
| Gaussian | `[B,65536,14]` | 原 Gaussian 几何与外观 |
| Gaussian membership | `[B,65536,102]` | 原独立 sigmoid child masks |
| thing logits | `[B,100,19]` | 18 thing classes + no-object |

### 2.2 每层严格顺序

12层循环中，每层顺序固定：

1. 使用**当前** μ/ρ 计算 r、anchor-to-ray cross-attention bias、anchor positional embedding，保留原 V3 的 PE mode、参数与调用顺序。
2. 原 reconstruction cross-attention residual。
3. 原 anchor PE + reconstruction self-attention residual。
4. 原 reconstruction MLP residual。
5. 原 `head_mu(tokens)`、`head_rho(tokens)` additive refinement；重新计算 r。
6. 若为 L6：以此时、注入前的 tokens/μ/r 调用原 `scene_normalization(mu6)`，固定本次 forward 的 origin/ell；调用原 `encode_token`、`initialize_states`。保留 deterministic spatial/feature selection、16-neighbor pooling 和全部1024 anchors。L6 的初始化只执行一次。
7. 若为 L6/L8/L10/L12：调用原 `forward_registered_layer`，按原 evidence → residual object decoder → c/s update → anchor masks/pooling/classification 更新 q/c/s，保存完整原 result。
8. 以更新后的 q/c/s 和本层注入前 a/μ 计算 §3 的 T/u/Δh。
9. `enabled=True` 且 β>0 时，out-of-place 执行 `tokens = tokens + Δh`；否则直接保留原 tokens 对象，不执行 `tokens + 0`。μ、ρ、r 不直接写回。
10. 保存本层 state，`state['tokens']` 为注入后的 tokens，μ/ρ/r 为第5步值，object result 为第7步值；记录注入前 tokens 的 detached 统计而非覆盖计算图。
11. 将当前 tokens/μ/ρ 作为下一层循环输入。消息已合并进 tokens，因而在下一层 cross-attention 的输入处生效；不得在下一层重复加一遍 Δh。没有额外 pending attention bias，没有 object-conditioned self-attention bias 通道。

非注册层只执行第1–5、10–11步。L6 必须“先初始化，再更新 object state，再注入”；不允许先注入再生成 hypotheses。

禁止 inplace 修改 shared states/tensors，禁止任何消息 detach；只有 §3 的尺度因子及原 V3 已有 detached normalization/选择保留 detach。

### 2.3 L12、Gaussian head 与 readout

L12：原 μ/ρ head → object update → token 注入 → final state → `activation_head(final['tokens'], final['mu'], final['radii'])` → 原 `_readout`。

**只有 L12 的注入直接作用于最终 Gaussian head 输入；L6/L8/L10 的注入需经后续 decoder 层传导。** L12 token 注入不重跑本层 μ/ρ head，但会改变 Gaussian head 的 child offsets、scales、rotation、opacity、RGB，因而也能改变最终 child centers。

不在 L12 注入后重新更新 q/c/s 或重跑 anchor masks。原 readout 使用第7步的 a/f_anchor/m_query 与同一个最终 Gaussian tensor；child geometry 特征能反映被注入改变的 Gaussian。control 下此行为与 V3 一致。

新模型覆写 `decode_object_locus`、`forward_object_locus`、`step_loss`、本子类的 `_decode` 及 `forward_reconstruction_only`，使训练、local eval、official export 和 reconstruction-only 接口都走新交错路径；继承并直接复用原 `_readout` 和 reconstruction objective。本子类 `_decode` 只负责调用新交错路径，不改任何基类。不得走原 V3 事后遍历路径再事后补一次注入。

`self.inject_enabled` 固定代表 arm；`self.understanding_step` 在每次训练和评估前显式更新。`forward_object_locus(..., coupled=None, step=None)` 的缺省值取模型 arm 与当前 step，不得因 evaluator 没传 step 而始终按0计算 β。显式 `coupled=False` 用于 M1/反事实检查；新正式 runtime 不能照搬旧 `train_one_step` 中固定传 `coupled=False` 的调用。

## 3. 路由、注入公式、尺度和初始化

### 3.1 参数自由的独立路由

不直接复用 R 或 transpose(R)，不复用 final ownership/membership，不引入103路表。T只用于生成消息，不替换任何 V3 mask probability。

对更新后的 q 与注入前 a，定义：

```text
a_hat = normalize(layer_norm(a, normalized_shape=(256,), eps=1e-5), dim=-1, eps=1e-6)
q_hat = normalize(layer_norm(q_new, normalized_shape=(256,), eps=1e-5), dim=-1, eps=1e-6)
d2[b,i,j] = sum_xyz(((mu[b,i]-c_new[b,j])/(s_new[b,j]+1e-6))^2), j<100
geo[b,i,j] = -log1p(d2[b,i,j]), j<100
geo[b,i,100:102] = 0
Z[b,i,j] = dot(a_hat[b,i],q_hat[b,j])/0.1 + geo[b,i,j]
T = softmax(Z, dim=-1)                 # 每个 anchor 在102 states上归一化
v = layer_norm(q_new, (256,), eps=1e-5) # functional，无新增LN参数
u[b,i,:] = sum_j T[b,i,j]*v[b,j,:]
```

T 的每行和为1。没有 hard radius mask、top-k crop、void channel、no-object gate 或置信度阈值；所有 anchors 可接收所有 states 的消息。stuff 两个通道不使用紧凑 spatial support。

**不复用 `_geometry_bias`**：原 evidence 的 `clamp(-20,0)` 原样保留，仅新消息路由用上式 Student 型软尾偏置 `-log1p(d2)`，不做下限 clamp，因此没有6.32·s后的固定地板。d2必须 finite，不能用 `nan_to_num` 隐藏错误。此偏置不是 mask，不宣称远场被排除。

### 3.2 唯一新增可训练模块

`object_locus_joint_injection = nn.ModuleDict({'L6': Linear(256,1024,bias=False), 'L8': ..., 'L10': ..., 'L12': ...})`。

四层不共享权重；每个 weight `[1024,256]`，全部 `zeros_`。没有 hidden MLP、trainable route Q/K、bias、learned gate、新 normalization 参数或 trainable β。新增4个 tensors，共1,048,576个参数。

全部零初始化仅用于这四个新投影；不是将原78个 object tensors 清零。单层线性结构避免“多层全零网络”导致永久零梯度。构造与 zeros_ 在 §1.2 同一 fork_rng 岛内完成。

### 3.3 Token-only 残差与限幅

```text
beta_control(t) = 0
beta_joint(t) = min(max(t,0)/200, 1)
sigma_h[b,i,1] = sqrt(mean_d(h_pre[b,i,d]^2)).detach()
delta_h[b,i,d] = beta(t)*0.1*sigma_h[b,i,1]*tanh(W_layer(u)[b,i,d])
h_post = h_pre + delta_h
```

每个 anchor 的注入范数满足 `||Δh||2 <= 0.1*β*||h_pre||2`。sigma 不加人为 floor；零 token 的残差自然为0。日志分母使用1e-6仅防除零，不进入残差公式。

登记新增常数：route temperature=0.1；route geo coefficient=1；route denominator epsilon=1e-6；functional LN epsilon=1e-5；feature normalize epsilon=1e-6；token residual scale=0.1；β ramp=200 updates；日志 epsilon=1e-6。除此之外不引入新尺度。

**直接 Δμ=0、Δρ=0、Δr=0**，不新增 center attraction 或 log-radius residual，因而不需要新的几何/log空间 clamp；原 controller 的 c/s 限幅及 reconstruction radius activation 原样保留。不得把 c/s 的几何更新误记为 anchor μ/ρ 注入。

## 4. Loss、梯度与 optimizer：与 V3 配方一致

直接调用 `object_locus_v3_set_loss.py::object_locus_v3_set_losses`，不新增 loss 文件。

最终 L12 只做一次 Hungarian：cost=`2*(-P(correct class))+5*BCE_cost+5*Dice_cost`。原概率域 matcher clamp 与 probability loss 按基线代码保留。匹配使用两个 context views 的 valid pixels；IGNORE、GT instance union、stuff 标签均沿用原实现，不引入 anchor visibility filtering。

```text
L_U = 0.1*(2*L_classification + 5*L_thing_BCE + 5*L_thing_Dice
           + 5*L_stuff_BCE + 5*L_stuff_Dice)
w_U(t) = min(t/200,1), t>=0
L_total = L_reconstruction + w_U(t)*L_U
```

no-object=index18，CE weight=0.1；classes0..17对应原 semantic classes2..19；wall/floor 两个 stuff 通道沿用原映射。保留 independent sigmoid child masks，不做 parent inheritance、不改 panoptic fusion。

reconstruction supervision layers、权重及 RGB/SSIM/visibility 等原 loss 与原 options 完全相同，不重新定义重建损失。禁止 auxiliary、anchor CE、semantic CE、独立 objectness、identity、contrastive、entropy、compactness 或任何新正则。

### 4.1 可训练清单与 GC

reconstruction encoder/decoder、anchor μ/ρ与 refinement heads、activation head、原 object controller、四个 injection weights 全部 `requires_grad=True`。frozen tensors=0。

梯度控制沿用原两次 backward：先 `L_reconstruction.backward(retain_graph=True)`；只在随后 `w_U*L_U` backward 期间，对 reconstruction 参数注册乘0.01的临时 hook，结束移除。

`object_locus_v3_set.*` 和 `object_locus_joint_injection.*` **均排除在 reconstruction GC hooks 外**。注入模块及 object states 从重建 loss 收到的梯度不缩放；understanding 到 reconstruction 的梯度缩放0.01。不 detach object→anchor 路由，允许双任务训练生成耦合。

### 4.2 Optimizer 与 schedule

AdamW：betas=(0.9,0.95)，eps=1e-8，amsgrad=False，foreach=False，fused=False。全局 unique trainable parameters clip_grad_norm_=1.0，执行于两次 backward 完成之后、optimizer step之前。

固定四组，与控制臂共同：

- object_decay：原 object 分支与四个 injection weight，peak LR=1e-4，WD=0.05。
- object_nodecay：原 object 的1D/bias、stuff_seed、`_no_weight_decay=True` 参数，peak LR=1e-4，WD=0。
- reconstruction_decay：其余原 reconstruction 矩阵参数，peak LR=1e-6，WD=0.05。
- reconstruction_nodecay：原 reconstruction 的1D/bias及原no-decay标志参数，peak LR=1e-6，WD=0。

不得把 injection 放进1e-6组或 GC hook；不得把 WD=0.05/0误读为 object 全部0.05、reconstruction全部0。

两个正式臂与M2使用同一函数：

```text
m(0)=0
m(t)=t/200, 1<=t<=200
m(t)=0.1+0.9*(1+cos(pi*(t-200)/(3584-200)))/2, 200<t<=3584
lr_group(t)=peak_group*m(t)
```

不能调用旧 runtime 的默认5000步/0.02下限 schedule；new runtime 显式传 LR 和 w_U。t为本臂将执行的第t次更新，1-based，step0只评估。

optimizer 参数须无重复、无遗漏；control 注入关闭时四个权重 grad=None且保持零，是设计预期，不是 frozen。joint 首步 W 上游梯度可为0，但 W自身应能得到梯度；M2判断更新后的活动性。

## 5. 固定数据、正式运行量和评估注册

读取既有 V3 的 `/space/mawb/ssst/group_plus/object_locus_v3_set/data_manifest.json`，SHA256=`c6c1a0dbfb5c88745a9f633c93bd0bb513a946717ceca34a41ad5802cc3f9b35`。原样复制，不重新选择窗口。

8个训练场景固定：`scene0000_00, scene0003_02, scene0009_00, scene0013_01, scene0018_00, scene0024_02, scene0031_00, scene0035_00`。

每场景7个训练窗，共56；每窗2 context+2 novel，provider仍为 `scripts/object_locus_v3_set_runtime.py::build_batch` 使用的 `ObjectLocusV1Provider`。RGB256×256、GT poses、scene scale0.15、first_cam normalization、frame ordering、crop/resize全部原样。不得用 predicted depth 代替GT，不新增深度监督。

两臂64epochs，分别3584 updates，总正式7168 updates。每窗每臂64次 exposure。epoch index e=0..63：`np.random.default_rng(42+e).permutation(56)`，完全保留manifest顺序与此排序。

两臂独立 fresh model/AdamW/RNG；先跑control完整64epochs，再跑joint完整64epochs。不是control checkpoint接续joint。M2临时40更新不计正式更新，不允许复用其权重/optimizer/RNG。

### 5.1 注册 splits / scopes / 节点

四个 splits：`train_all56`、`same_scene_holdout8`、`dev8`、`val32`，逐项来自上述manifest，不注册第五个训练probe，不改其 frame identities。

两臂在 epoch `0/8/16/32/64`，对应step `0/448/896/1792/3584`，**四个splits全部执行local与official all/novel评估**。这比历史V3补齐了评估注册，不改变训练数据或预测规则；两臂相同。

scope映射锁定：context→`official['all']['context_*']`；target_all→`official['all']['target_*']`；novel→`official['novel']['target_*']`。保留源JSON路径与SHA。-1为UNDEFINED、缺文件为MISSING，均不能变成0或借用其他scope。

复用 `eval_object_locus_v3_set.py::evaluate_windows`、`export_object_locus_v3_set_official.py::export_windows`、`write_official_pair` 和 `eval_object_locus_v1.py::_official_run`。新eval wrapper只负责加载本模型/arm/step以及编排split与per-scene子集，不能修改预测过滤、score、panoptic fusion、packed PNG或official算法。

GT valid域只限制IoU/任务GT统计；候选资格、面积和score按Evalfix的预测域规则。class19不是官方类别数；官方20语义类与void沿用既有协议。

报告local semantic mIoU all/thing/stuff、candidate mAP/AP50与CA/CW TP/FP/FN/P/R、panoptic CA/CW与PQ、official mIoU/PQ/mAP/AP50、context/target_all/true novel PSNR、raw mask与matched分类。三种实例指标不得互相替代。

在epoch0/64，四splits另执行每个scene子集的official评估用于§9；所有节点均保存每窗口local指标。定性图固定每split manifest前2窗，在0/8/16/32/64相同位置输出RGB、GT、candidate、panoptic、semantic，不根据好坏选图。

评估前保存training RNG，设置 `understanding_step=当前step`，eval/no_grad；评估结束恢复RNG和train状态。不得让评估消耗改变后续数据/随机状态。

## 6. CPU contracts 与 M1：控制臂等价性

CPU只保留必要检查：shapes；四W全零及两臂公共权重全等；路由softmax沿102维且row sum正确；原evidence沿1024维；四W不在重建LR/GC组；token限幅；β端点；可训练参数覆盖；原loss复用；control关闭不注入；forward执行顺序与最终readout。

### 6.1 M1真实输入与噪声包络

真实smoke batch固定为epoch0 permutation计划的第1、2、3个窗口，依据§5 manifest和 `default_rng(42)` 唯一确定；输出JSON写出scene/context/novel，不另选更容易的batch。参考原V3模型与新control载入完全相同450+78 tensors。

在同一RTX3090、相同Torch/CUDA/TF32设置、同输入、同权重、同RNG状态，未改动V3 forward执行两遍，记每个浮点输出tensor噪声包络：

`E_X = max(abs(X_V3_run1-X_V3_run2))`。

比较新control与V3_run1：`max(abs(X_control-X_V3_run1)) <= E_X`，不加1e-6、不乘安全系数、不设置经验floor。E_X=0时必须exact。离散indices/seed/labels要求逐项相等。

检查所有12层tokens/μ/ρ/r、4层q/c/s/evidence/anchor masks、最终Gaussian、RGB/alpha、region_mass、19类logits、semantic scores、PSNR与loss components。比较两边共同的科学字段，不要求新诊断字段存在于原模型。

新joint step0 β=0、W=0，同样符合上述包络。M1失败先定位交错顺序、PE、state保存、步数或入口错误，修实现后只重跑受影响contracts和M1；不得扩大容差。

### 6.2 3090完整一步 smoke

用上述第1真实batch，在临时模型上完成forward→原loss→GC backward→clip→AdamW step，所有关键科学输出/loss/梯度finite，shared reconstruction与object所需梯度非零；control的W无梯度属于预期。记录allocated/reserved峰值；不得OOM。

同batch运行local candidate/panoptic evaluator和single-pair official all/novel export/evaluator，UNDEFINED AP原样记录，不作为失败。正式训练不能复用smoke状态。

## 7. M2：固定40步活动性探针

M1通过后，在同GPU运行**一个**临时joint模型，从§1 fresh初始化、fresh optimizer开始，使用正式计划前40 entries，§4原schedule、loss、β、GC和clip执行40次更新。

第20与40步记录§8指标；在第40次更新完成后固定用计划第1窗口、eval/no_grad、step=40执行：

1. enabled forward重复两遍，建立每个活动量和最终Gaussian的同臂重复包络。
2. 四次leave-one-layer-out forward，每次仅将L6/L8/L10/L12其中一层注入关掉，其余不变；不训练、不改变任何参数。
3. 一次object-mean消息forward：四层把每anchor的u替换为本层102个v的无权均值，W/β/幅度不变；其余科学路径不变。
4. 同一第1batch、同RNG与step=40做两遍相同GC backward，期间不optimizer step，比较每层W梯度并量出梯度噪声；记录clip前梯度。用单独临时实例或清空grad，不能污染正式init。

M2必须满足四层分别：W范数>0；u与Δh范数>0；相对Δtoken>0且超过其同臂重复包络；W梯度范数>0且超过两次梯度之差的范数；该层单独关闭时最终Gaussian XYZ和其余属性均有至少一个元素差异超过enabled重复包络。object-mean替换也必须使最终Gaussian XYZ或属性超出重复包络，证明路径确实读取了object-specific内容。

**token-only直接Δμ/ell=0是设计值，不是M2失败。** 几何活动性以每层注入对最终child XYZ的leave-one-layer-out差异判断，必须非零且超噪；不得要求L6同一层head之前不存在的μ注入非零。

M2失败时停止正式启动，检查并只修明确实现错误：错误arm/step、关闭β、projection未参与forward、错误GC/LR归组、detach、消息未传至下层或head、全零多层网络。修复后重新fresh执行这一个40步探针。

若没有明确实现错误仍未活动，报告C“未形成有效干预”，STOP；不得自行增LR、增β、延长探针至50步、增加loss、换结构或正式开跑。这是一次实施里程碑，不是根据性能做机制搜索。

M2通过后丢弃全部临时权重/optimizer/RNG。重新建立共同fresh初始化快照，两臂重新从step0启动。

## 8. 日志、有限值和失败捕获

活动性指标在正式训练中**只记录、不设门限、不作为暂停或启动下一阶段的gate**；M2为用户指定的一次性例外，只验证干预实际发生。

每20updates及最后一步记录loss components、LR、w_U、β、clip前总grad norm、GC注册/移除数量、GPU allocated/reserved，以及每注册层：

- u范数、raw W(u)范数、Δh范数的mean/max；
- 每anchor `||Δh||/(||h_pre||+1e-6)` 的mean/max；
- 直接 `||Δμ||/ell=0`、`||Δρ||=0`，明确标注direct；
- T row sum min/max、entropy均值、thing/stuff路由质量；
- 每层W参数范数、clip前梯度范数；Δh的两次GC backward累计上游梯度范数；
- 原c/s finite与min/max、全部关键输出finite。

日志detach，不保留整个训练图；gradient观察hook一次性累加标量、每步移除，不能改梯度。control记录W梯度状态NONE_expected，不将其假装为有效梯度。

epoch0/8/16/32/64仅在train_all56第1窗增加上述四次leave-one-layer-out及一次object-mean no_grad测量，记录最终XYZ变化/ell、非XYZ属性变化、μ12变化/ell。用于观察干预，既不新增训练臂也不根据结果自动调整。

保留有限值守卫和失败捕获。loss/output/grad出现NaN/Inf时，在optimizer更新前STOP，保存batch与frame IDs、component scalars、tensor首个非有限位置、当前model/optimizer、pre-forward RNG、arm/epoch/position/step和traceback。不得nan_to_num、跳batch或重启optimizer掩盖问题。失败快照写各臂自己的目录；仅错误发生时保存大文件。

复用现有runtime捕获逻辑，在新runtime补齐pre-forward RNG及注入字段；新增replay入口按现场复现，不默认开启全程anomaly detection。仅修明确实现错误，历史V1 step4090根因继续标为UNKNOWN。

## 9. 预注册统计、可证伪预测和A/B/C结论

### 9.1 固定预测

工程预测：四层object→anchor消息会对最终Gaussian几何/外观产生可测影响，且不是将所有object平均后就相同的通用消息。由M2及固定checkpoint干预日志验证。

任务预测：在实例监督和读出保持不变时，joint相对control提高train_all56官方context AP50，并不损害holdout与重建。若未发生，不能宣称本轮验证了实例组织的任务价值。

不预注册“物体内RGB方差应下降”：真实物体存在不同材质与纹理，方差下降可表示过平滑。此具体代理不采纳；不添加纯度审计链来代替本轮任务比较。

共同conditioning≠已实现严格实例局部。此轮不以T、membership或预测中心构造GT，不输出“纯度已证实”的结论。若报告研究层面的实例局部性，必须另有独立GT、遮挡贡献、coverage与opacity证据；本轮默认只报告工程与任务结论。

### 9.2 Per-scene与bootstrap，算法固定

主结论只使用epoch64，不能挑epoch32最佳点。所有曲线仍报告。

主指标：train_all56 context官方packed-panoptic AP50。报告官方56窗汇总AP50与8个scene分别官方AP50；不能平均local candidate AP替代。

paired bootstrap使用**8个per-scene official AP50的差值的macro mean**，不是声称它等于官方56窗pool AP。对固定排序的8个scene，`d_s=AP50_joint_s-AP50_control_s`；`rng=np.random.default_rng(20261002)`；10000次，每次有放回抽取8个scene索引，两臂同一抽样，计算mean(d_sampled)。95% percentile CI用`np.quantile(samples,[0.025,0.975],method='linear')`。

holdout同样按8个scene配对，独立 `default_rng(20261003)`、10000次。dev8与val32按各自scene identities报告同型差值CI，分别seed20261004/20261005；重复同scene的窗口先按scene子集官方评估，不把窗口当独立scene。

任何必要endpoint per-scene官方值UNDEFINED/MISSING时，不填0、不删除该scene、不改CI算法；对应任务统计不可判定，B不成立。能重跑缺失的已注册eval，不能重训。

### 9.3 三档结论

判定优先级B→A→C；A表示工程通过但本配方未满足B，不能与B混写。

**A：工程通路已接通。** M1/M2通过即可确认工程A。正式两臂各3584 updates完成且finite、但B不满足时，写“工程通路接通，但该固定训练配方未证明任务收益”，逐项说明指标；不能外推为所有生成耦合无价值。若工程通过而正式任务未完成，分别报告工程A、任务C。

**B：生成耦合有价值。** 在A的工程条件上，正式两臂各3584 updates完成且finite，并同时满足：

1. train_all56 context官方pooled AP50，joint-control >0；paired per-scene macro AP50差值95% CI下界>0。
2. same_scene_holdout8 context官方pooled AP50下降不超过0.01绝对值；其per-scene差值CI下界≥-0.01。0.01是预设非劣效margin，不冒称机器噪声。
3. 四splits的context与true novel平均PSNR，joint相对control均不下降超过0.5dB；joint相对共同step0也均不下降超过0.5dB。
4. loss/readout/evaluator及mask粒度未改变，不能将提升归因于更换过滤、阈值或分辨率。

此B只支持“小规模固定训练条件下生成耦合有任务价值”，不自动支持跨场景泛化或严格实例局部。dev8/val32分别报告，未见场景收益若没有出现，明确写没有泛化证据。

**C：不可判定。** M1未通过、干预未活动、正式run未完成、主要证据缺失时使用。**“注入未活动 → 只能写‘未形成有效干预’，不得写成‘耦合无价值’”。** 有效干预且训练完整但无收益通常归A，不把失败隐藏成C；若B统计缺失则注明工程A、任务C。

## 10. Git顺序、GPU、正式提交和文件交付

所有GPU工作固定RTX3090 24GB，节点`3dimage-13`，单GPU，Python `/space/mawb/anaconda3/envs/tokengs/bin/python`。使用现有V3 submit脚本的SLURM partition/account/resource写法，新脚本增加本任务路径与arm顺序；不更换GPU节点，不抢占其他任务。GPU编号由SLURM分配，显式记录；13号忙则排队，不能改用11号。

严格顺序：

```text
implementation → CPU contracts → M1+3090 one-step smoke → M2(40 updates)
→ commit → push origin/main → 验证远端SHA → fresh control正式训练
→ fresh joint正式训练 → 配对报告与打包
```

**实现与 M1/M2 通过后 commit、push，再开始正式训练；失败只修明确实现错误，不自动改结构、不调参、不追加实验。**

origin/main可能已包含V3 Expanded新增提交。不得force push或把main退回b00b94f。先在独立执行分支提交通过的实现，再fetch origin/main并merge其新增历史；只解决新注册项的并存，不能改既有科学文件。核对基线V3 model/controller/loss/evaluator/export/runtime与b00b94f相同；若远端这些文件发生科学变化，停止并报告冲突，不自行换实验基线。

合并后新科学文件内容必须与M1/M2通过时相同，重跑CPU与M1验证最终执行树；若只增加无关历史而新科学内容未变，不重复40步M2。提交merge并push `HEAD:main`，远端verify完整SHA，然后以该SHA的clean worktree执行。正式训练启动前的所有修复均按此顺序提交推送。

两臂模型与optimizer group layout相同。control四W零、joint四W零，同一公共init SHA；正式每臂开始前seed/reset RNG同42，评估RNG保留；manifest与3584-entry plan hash相同。

报告根：`/space/mawb/ssst/group_plus/object_locus_joint_v1/{control,joint,pair}`；run根：`/space/mawb/ssst/workspace_group_plus/object_locus_joint_v1/{control,joint}`。临时smoke/M2在报告根`validation/`，不混入正式checkpoint。

checkpoint保留epoch0/8/16/32/64与最新非评估恢复点；每epoch写model、optimizer、RNG、配置、arm、完整SHA、data/plan/init hashes、completed updates、next position、exposure counts。新文件完整落盘后仅滚动删除本run上一非注册恢复点；不得清理旧实验。本轮不追加清理任务。

基础设施失败允许原样恢复model/optimizer/RNG/position，不额外更新、不重置optimizer；nonfinite STOP，不自动续训。报告打包失败仅修packaging并重做打包，不重训。

最终提供一个≤28MiB ZIP，含spec、决策表、合规映射、manifest/plan/init/transfer/optimizer报告、M1噪声包络与差异、M2全部固定探针、训练曲线、逐scene官方来源与bootstrap、完整任务表、固定图、关键日志和源码diff；不含数据集/checkpoint/大量official PNG。checkpoint保留独立路径。ZIP实际解包验证一次。

最终回复逐项报告：基线/实现/最终SHA与push；修改文件；450/78/4 transfer；参数与LR归组；forward顺序和shapes；M1包络/M2活动结果；3090allocated/reserved；两个job IDs、完成更新与exposure；checkpoint/eval节点；两臂mIoU/PQ/mAP/AP50/PSNR及paired CI；A/B/C判定；是否有效object masks、是否保持重建、是否有未见场景证据；未执行任何未注册科学改动。

## 11. 决策表

所有科学决策均已固定；没有留给执行者的科学选择。

| 决策项 | 取值 | 依据 | 影响 | 若取错的后果 |
| --- | --- | --- | --- | --- |
| 基线 | b00b94f完整SHA、独立新worktree | 用户指定V3 Evalfix | 同一科学基座 | 在落后main实现错误模型 |
| pretrained SHA纠正 | 5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f | V3 runtime/spec真实值；纠正用户笔误 | 唯一47500权重 | hash失败或加载错误权重 |
| 训练起点 | reconstruction fresh transfer，object/injection fresh | 用户指定 | 比较生成耦合而非旧理解能力 | 续训曝光不公平 |
| strict transfer | 450 reconstruction+78原object+4新W | 原参数结构及4线性投影 | 保持旧权重与新参数边界 | 漏参数、静默错误加载 |
| 构造与seed | global42；单CPU fork_rng seed31415，object先、injection后 | 保留V3初始化 | 公共78tensors相同 | 注入seed扰动公共初始化 |
| 注入参数 | 四Linear(256,1024,bias=False)，全零 | 全零且不产生多层dead network | 初始严格无干预 | 零MLP无梯度或step0不等价 |
| 注册层 | L6/L8/L10/L12，各自W | 用户要求和V3states | 四层生成通道 | 层间通路错位 |
| 注入位置 | 本层μ/ρ及object更新后、state/head前 | H3、原decoder顺序 | 下层cross前及L12head生效 | 事后加法不影响生成 |
| 写入张量 | tokens only；μ/ρ直接残差0 | 隔离token生成耦合 | head/后续层间接几何变化 | 多变量几何干预无法归因 |
| 次层消息 | 已加进tokens，不再pending加一次 | 防重复注入 | 单次残差 | 注入尺度翻倍 |
| 注意力偏置 | 不新增 | 收紧唯一变量 | 原ray bias/PE保留 | 引入第二通道归因污染 |
| 路由 | normalize feature dot/0.1 + -log1p(d2)，softmax102轴 | H4 | 每anchor消息尺度稳定 | 误用R列和导致密度漂移 |
| 原geometry bias | evidence原样；新route不复用clamp | 避免远场固定地板 | Student软尾、无hard crop | 远场均匀地板被误称排除 |
| stuff | 路由geo=0，无c/s | 原stuff定义 | 全局support | 把墙地局限成物体 |
| 残差 | β*0.1*rms(h).detach()*tanh(Wu) | H5及有限相对幅度 | 范数相对上界0.1β | 无界幅度损害重建 |
| β | control0；joint min(t/200,1) | 固定warm-up | 前200平滑启动 | eval step0导致长期不活动 |
| 新尺度 | §3.3八项固定值 | 全部登记 | 可复核 | 执行者隐式调参 |
| LR/GC | injection属object1e-4、WD.05、不受GC.01 | 注入可学习、非重建参数 | 防1e-6+GC双重抑制 | 无有效干预 |
| loss | 原final set loss与重建objective，不新增 | 保留V3成功基线 | 只改变生成通路 | 收益来自监督变化 |
| trainability | 全部可训练，冻结0 | 加载≠冻结 | 双任务联合学习 | 下游冻结实验冒充joint |
| 数据与曝光 | 原56窗64ep/3584，每窗64 | 用户要求 | 两臂严格匹配 | 数据量混杂 |
| optimizer | AdamW .9/.95 eps1e-8，WD矩阵.05/其余0，clip1 | V3真实runtime | 复现配方 | 不同优化过程 |
| schedules | LR200warm→cosine.1；w_U=min(t/200,1) | V3正式train函数 | 避免旧runtime默认schedule | 控制臂不再V3配方 |
| M1 | 原模型重复包络，无floor/倍率 | 用户要求 | 可证关闭等价 | 交错前向污染配对 |
| M2 | 40步、20/40日志、固定40终点反事实 | 用户20–50要求内唯一值 | 证明有干预 | 零注入被误判无价值 |
| 直接Δμ为0 | 不纳入非零活动要求，最终child XYZ纳入 | token-only数学事实 | 正确判定几何通路 | 对不存在的输出设不可能gate |
| 正式活动性 | 每20步只记录，不门控 | H2/H6；M2一次性例外 | 不陷入机制审计循环 | 按诊断自动调参 |
| 评估 | 四splits、五nodes、三scopes全注册 | P3 | 消除历史MISSING歧义 | scope混用 |
| 主统计 | pooled AP50+paired per-scene macro CI，10000次95% | 8scene相关性与AP非线性 | 两种估计清楚区分 | 平均AP冒充pool AP |
| B阈值 | train CIlo>0且poolΔ>0；holdΔ/CIlo≥-.01；PSNR容差.5dB | 预注册任务收益与保护 | 防事后改口 | 挑节点或损重建换指标 |
| P2代理 | 测最终生成与object-specific干预；不测RGB方差下降 | 多材质物体不应被过平滑 | 可证伪工程/任务预测 | 错把平滑当组织 |
| 局部性主张 | 本轮不宣称严格实例局部 | H8 | 共同conditioning不自证 | 循环论证 |
| GPU | 13号RTX3090排队、control后joint | 用户优先与并行任务隔离 | 不抢Expanded资源 | 改节点或干扰运行 |
| SLURM GPU index | 留给SLURM分配；记录CUDA_VISIBLE_DEVICES | 编号属于调度资源，不是科学决策 | 避免硬编码抢GPU | 与别的job冲突 |
| Git | M1/M2→commit→非force merge/pushmain→正式 | 用户顺序与远端新历史 | 保持可复现提交 | 先训练未push或覆盖main |
|失败 | 仅修明确实现错误；指标差照报 | 用户禁止自由发挥 | 固定实验完成 | 自动换结构重训 |

## 12. 第二轮评审合规映射表

| 评审条目 | 提示词小节 | 采纳状态与具体落实 |
| --- | --- | --- |
| H1 控制臂可证等价 | §1.2、§2、§6、§10 | 采纳；新子类交错前向、原模型重复包络、step0两臂等价 |
| H2 活动性记录 | §7、§8 | 采纳记录要求；用户新增M2一次性go/no-go，正式日志仍不门控 |
| H3 注入顺序/写入 | §2.2–2.3、§3.3 | 采纳；tokens-only、无attention bias、下一cross前生效 |
| H4 路由归一化/clamp | §3.1、§8 | 采纳；独立102轴softmax、不复用R、不复用-20地板 |
| H5 注入限幅 | §3.2–3.3 | 采纳；tanh、0.1β相对范数上界、尺度登记；不直接写μ/ρ |
| H6 逐层梯度实测 | §7、§8 | 采纳；四W梯度、逐层leave-one-out最终几何影响，无新增aux |
| H7 A/B/C结论 | §9.2–9.3 | 采纳；预注册CI/holdout/PSNR、非活动不可判定 |
| H8 不自证局部性 | §1、§9.1 | 采纳；明确conditioning≠严格局部，不用T构造GT |
| P1 历史负结果作为先验 | §1、§3、§9.1 | 采纳；不引入严格继承/103竞争，不把旧配置低分当容量否证；不复跑旧实验 |
| P2 可证伪预测 | §7、§9.1 | 采纳预测要求；未采纳“物体内外观方差必须下降”，原因是材质差异与过平滑混杂；改为object-specific生成干预+固定任务预测 |
| P3 预注册清单 | §5.1、§9、§10 | 采纳；splits/scopes/nodes、per-sceneCI与阈值全部写死 |
| 第4部分建议1 交错实现 | §2 | 采纳；新decoder子类，参考forward_stateful、逐项保留原decoder顺序 |
| 第4部分建议2 等价smoke | §6.1 | 采纳；实际同臂重复噪声包络，无人为容差 |
| 第4部分建议3 点位/张量/尺度 | §2.2–2.3、§3.3 | 采纳；L6/8/10/12、tokens-only、所有新尺度固定 |
| 第4部分建议4 路由/几何/熵 | §3.1、§8 | 采纳；102归一化与soft-tail bias，路由质量/熵只记录 |
| 第4部分建议5 活动/梯度记录 | §7、§8 | 采纳；正式不设门限；M2为用户指定一次性例外 |
| 第4部分建议6 三档/指标/CI | §5.1、§9.2–9.3 | 采纳；epoch64判定、paired macro CI、holdout .01/PSNR .5dB |
| 第4部分建议7 不活动不可判定 | §7、§9.3 | 采纳；固定书面措辞“未形成有效干预” |
| 第4部分建议8 finite与捕获 | §8、§10 | 采纳；保留守卫/现场/原样重放，不用历史nonfinite归因结构 |

以上所有决策构成执行合同。科学结果差只报告，不自动改结构、不调参、不追加实验。

## 13. 补充约束（核查增补，优先级等同正文）

13.1 新 decoder 的挂载（必须）
ObjectLocusJointAnchorDecoder 实例必须赋给 self.anchor_decoder，覆盖
LocusGSRecon.__init__（canonical_recon_models.py:131）建立的
LocusGSAnchorDecoder 实例；不得放在新属性上，否则 state_dict 会同时含两套
decoder，§1.2 的 450-tensors 契约当场破裂。该子类不得新增、删除或重命名任何
参数（decoder 侧零新增参数）；4 个注入权重只存在于模型级
object_locus_joint_injection 下。现成范式：instance_state_locusgs.py 中
self.anchor_decoder = InstanceStateDecoder(opt, self.enc_dec_backbone.decoder_blocks)。

13.2 交错循环的 attention 调用（必须）
新循环体内 cross-attn / self-attn / MLP 的调用与 scale 包装必须与
LocusGSAnchorDecoder.forward 逐字相同，不得改写为手写 SDPA 路径。若确需触碰
attention，必须照 instance_state_locusgs.py 加"已验证 SDPA 路径"断言
（no rope / no flex score_mod / no flex block_mask / fused_attn=True）。

13.3 不使用单层 hook（必须）
不传 LocusGSAnchorDecoder.__init__ 的 token_update_hook / token_update_layer；
交错注入由新子类自行实现，不与该 hook 并存。

13.4 M1 失败诊断（必须）
M1 失败时按层序输出首个发散张量：层号、张量名、
max|X_control - X_V3_run1|、该张量的 E_X。不得直接扩大容差。

13.5 说明（不必改代码）
object_locus_init_seed 在 options.py 中不存在，
getattr(opt, "object_locus_init_seed", 31415) 恒为 31415；固定 31415 与 V3
等价，不新增该 option。

13.6 报告 caveat（仅措辞）
每臂仅一次训练 run，训练层 run-to-run 方差未估计；B 档判据是端点差值 +
per-scene paired CI，不声称覆盖训练方差。

## 执行记录

独立 worktree 从指定 b00b94f 创建；无关历史展示资源 LFS 404，使用 GIT_LFS_SKIP_SMUDGE=1。

### 2026-10-02 执行状态：M1 STOP，未进入正式训练

- 基线：`b00b94fdc45af6d2f78c55f05671aaa75906204f`；独立分支与 worktree 已建立。
- 文件边界：新增规格允许的10文件；仅修改 models/__init__.py 与 options.py 注册项；旧科学文件未修改。
- CPU contracts：5项通过。实测 checkpoint/manifest SHA 匹配；450/78/4；公共权重 exact；4个W全零；新增参数1,048,576；冻结0；optimizer覆盖且注入不受reconstruction GC。
- SLURM M1/完整smoke/M2验证 job：58067，RTX3090 / 3dimage-13。第一个 M1 窗口 scene0018_00 context[63,75] novel[69,72] 的control失败后停止，完整一步smoke和M2均未运行。
- 首个发散项：L6 `loss_layer6`，difference=3.170222043991089e-6，E_X=8.23289155960083e-7。`loss_gaussian_visibility_layer6` difference=3.169705451000482e-6，E_X=8.231727406382561e-7。未改变容差。
- 初次M1中所有12层 tokens/mu/rho/radii、4层共同object字段、Gaussian、RGB/alpha、membership、logits与semantic输出均符合原两次包络；不得据此把完整M1记为通过。
- 失败诊断 job：58068，仍使用同一固定M1第1窗及未修改V3。原V3两遍decoder states和最终Gaussian逐项exact；RGB/alpha/depth exact；L6 means2d_pred仍有69,572个元素不同，最大绝对差0.5350165367126465。final forward means2d有87,378个差异元素，最大差25.565210342407227。
- 安装gsplat源码显示 means2d 使用 at::empty；被裁剪Gaussian提前返回，未写means2d；原canonical visibility loss读取全部means2d。此为有证据支持的原因，但缓存CUDA二进制的构建来源未建立，不把源码核查当作二进制证明。
- 遵循§1.1、§6.1、§10：未修改renderer/loss，未扩大噪声包络，未重新抽窗或调整科学配置；未运行M2/正式control/joint，未commit/push。正式updates=0/0；工程C、任务C。未形成有效干预尚未测试，不能声称耦合无价值。
- 原V3 Expanded保持运行。历史V1 step4090根因仍UNKNOWN。
- 当前代码是可审阅实现，GPU训练、评估和报告入口的完整通过状态尚未建立。首个失败验证job未保存allocated/reserved峰值，不能补造该数字。
- 证据根：`/space/mawb/ssst/group_plus/object_locus_joint_v1/validation/`；配对STOP报告与ZIP：该根的`pair/`。

继续正式阶段须先解决当前合同与原renderer/visibility行为之间的阻塞。现有规格禁止修改原renderer和原loss；本执行没有越过该限制。


## M1修复续接合同

# Object-Locus Joint V1：M1投影正确性修复与继续执行指令

这是原《Object_Locus_Joint_V1_Codex_Implementation_Spec.md》的补充执行合同。冲突处以本文件为准；未被本文件明确修改的设计、数据、优化、M2与正式配对协议全部保留。Codex只执行，不自行调参或追加实验。

## 1. 当前证据与本轮任务

已交付的job58067/58068说明：

- 第一个固定窗口为 `scene0018_00`，context `[63,75]`，novel `[69,72]`。
- control的M1只失败于 `loss_layer6` 与 `loss_gaussian_visibility_layer6`；差异约3.17e-6，原同臂包络约8.23e-7。
- 已测Gaussian、decoder科学状态、RGB、alpha与depth相同；`means2d_pred` 在相同原V3输入下也会变化。
- 原 `canonical_recon.py::canonical_layer_loss` 的Gaussian visibility直接读取全部renderer投影坐标，没有区分未有效写入的点。
- 仓库已有确定的解析投影 `project_points_means2d` 和处理near-plane validity的 `visibility_loss_from_points`；普通TokenGS loss已使用解析路径，但canonical Gaussian visibility仍在使用旧路径。

这些证据足以实施一个窄范围的投影正确性修复；不要求证明当前缓存CUDA二进制的完整构建来源，不要求重编译gsplat。不能把历史step4090 nonfinite或旧泛化失败归因于这一缺陷。

任务是：修正两臂共用的Gaussian visibility输入及renderer输出坐标定义，恢复M1→完整一步smoke→M2→commit/push→正式control/joint，不重新设计Joint模型。

## 2. 显式扩大修改白名单：仅两个共用文件

本次授权在独立 `/space/mawb/ssst_object_locus_joint_v1` worktree内额外修改：

1. `tokengs/rendering/gs.py`：仅修正 `GaussianRenderer.render_standard` 的 `means2d_pred` 输出来源。
2. `tokengs/models/canonical_recon.py`：仅修正 `canonical_layer_loss` 的Gaussian visibility分支。

这是共同基础实现修复，不是新增loss或改变loss权重。必须向reference V3、control、joint同时应用，不能只修实验臂。

不要在正在运行的V3 Expanded worktree或旧主worktree里改上述文件。新commit push不改变旧worktree已检出的文件。旧实验产物不重评、不重训、不覆盖。

原protected-file检查对这两个文件增加“本补充授权的固定patch”例外：记录修改前后SHA和diff；其他受保护文件仍必须与基线相同。禁止通过删除整个检查绕过限制。

## 3. renderer坐标输出：对全部Gaussian做确定解析投影

`GaussianRenderer.render_standard` 中，保持原 `rasterization` 调用及所有参数不变，RGB、alpha、depth的提取、排列与shape不变。

停止把 `info['means2d']` 作为向模型/loss返回的投影坐标。循环中不再读取、收集或stack该字段；从同一Gaussian centers和camera计算：

```python
# 放在方法内导入，避免新增模块级循环依赖。
from tokengs.models.canonical_recon import project_points_means2d

analytic_intrinsics = torch.stack(
    (Ks[..., 0, 0], Ks[..., 1, 1], Ks[..., 0, 2], Ks[..., 1, 2]), dim=-1
)
means2ds = project_points_means2d(
    means3D,
    viewmat.transpose(-1, -2),
    analytic_intrinsics,
)
```

说明：`render_standard`收到的viewmat已经是world-to-camera的renderer布局；现有helper内部再次transpose。因此这里传入其transpose，确保helper最终用到的矩阵恰好是现有renderer的viewmat。

返回原key `means2d_pred`，shape `[B,V,N,2]`，FP32、原device、保持autograd。不detach，不用零初始化CUDA buffer，不把culled点统一填0，不用nan_to_num，不修改kernel。

解析坐标对所有点有定义，包括屏幕外和被裁剪点；它不表示该点实际参与渲染。near-plane有效性由§4明确处理。

只修当前任务实际使用的standard路径；`deferred_bp=False`继续固定，不修改未使用的DeferredBP实现或feature-channel renderer。

## 4. canonical Gaussian visibility：复用已有正确函数

将 `canonical_layer_loss` 原Gaussian visibility分支中从renderer means2d构造uv/relu/min/clamp/mean的代码替换为：

```python
if opt.canonical_gaussian_visibility_weight > 0 and camera is not None:
    gaussian_visibility = visibility_loss_from_points(
        gaussians[..., 0:3],
        camera,
        intrinsics,
        img_size,
        clamp_max=visibility_clip,
        znear=float(getattr(opt, 'znear', 0.0)),
    )
```

沿用函数现有实现，不再改写一套数学：

- 以真实Gaussian XYZ解析投影。
- `z_cam > znear`为有效；当前znear=0.025。
- 对有效点，原normalized-image-coordinate越界penalty不变。
- near-plane无效点设现有最大penalty。
- min over supervision views、clamp与mean沿用现有函数。
- 当前visibility_distance_threshold=1.0，Gaussian visibility weight=1.0，anchor visibility weight=0.1，均不改变。

不能以 `radii==0` 或“实际被renderer culled”作为所有visibility点的IGNORE：屏幕外点仍需要真实投影越界惩罚，不能被过滤掉。不要新增far-plane规则、GT depth gate或新loss。

不改RGB/SSIM、anchor visibility、reconstruction supervised layers/weights、understanding losses或任何学习率。

## 5. 必要验证与M1重新定义

### 5.1 只补必要正确性测试

在现有Joint contracts测试文件中新增解析投影/visibility测试，不新增大型审计框架：

1. 单位camera、fx=fy=100、cx=cy=128、image256×256，XYZ=(0,0,1)投影(128,128)、visibility=0。
2. XYZ=(2,0,1)投影(328,128)，normalized越界penalty=0.5625，visibility=0.5625；同测试另取XYZ=(3,0,1)，penalty=1.34375，经现有clamp1后visibility=1。
3. XYZ=(0,0,0.01)，因znear=0.025无效，visibility=1，不能将主点坐标解释为可见。
4. 两视图一视图可见、一视图越界时，min-over-views结果为0。
5. 使用至少一个平移camera，独立按 `R*XYZ+t` 和intrinsics计算投影，对照helper与renderer包装，防transpose用错。
6. 有效、非边界的屏幕外点上backward，XYZ梯度finite；测试点选XYZ=(1.5,0,1)，penalty=0.171875，避免clamp饱和导致预期梯度为0。

真实第1固定batch，使用同一个Gaussian tensor分别做原rasterization图像提取和修复后包装：RGB/alpha/depth必须一致，证明只改坐标/visibility正确性，没有改图像渲染。该检查只做一次，不另开实验。

### 5.2 修复后的M1参考

M1 reference仍是原V3模型类和原controller，但使用上述共同投影修复。不是拿修复后的control对比未修复、包含未定义坐标的旧loss。

明确报告名称：`V3 control + common analytic visibility correctness fix`。不得宣称它与历史V3训练loss逐位等价；共同基础修复会改变旧错误visibility及其梯度。旧V3指标只作为历史背景，新配对归因以本次control为准。

未改模型顺序、loss权重或readout；Gaussian/RGB等不应因该共同修复改变。沿用原M1两个reference重复的同臂包络，不加容差floor、不扩大倍率、不删loss比较；解析means2d也继续比较。

使用原计划前3真实窗口完成reference重复、control、joint step0对照。三个窗口全部M1通过后立即做原完整一步smoke与原40步M2。

如果仍有差异，查看首个具体不同字段并修明确实现错误；不要默认归因于CUDA噪声，不追加多轮诊断作业。原58067/58068失败JSON和日志原样保留在独立历史attempt目录，不用新PASS覆盖证据。

## 6. 允许11号：本任务固定改用3dimage-11单张3090

本补充将GPU节点从13号改为 `3dimage-11`，partition仍3090，`--gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00`。11号有8张3090，不意味着本轮改为8卡DDP。只申请1张，GPU编号交给SLURM，不硬编码物理GPU，不抢占其他任务。

在新Joint runtime提供自己的 `assert_joint_gpu()`，替换Joint smoke/train/replay对原V3 `_gpu_assert` 的导入；不修改旧V3 `_gpu_assert`。

要求：hostname以3dimage-11开头；进程可见CUDA GPU恰好1张；型号NVIDIA GeForce RTX3090；24GB级显存。记录CUDA_VISIBLE_DEVICES、节点、Torch/CUDA版本和实际加载的gsplat扩展路径/文件SHA；不要求额外重编译或更换库版本。

smoke/M2和正式两臂均用11号。本次正式control、joint放在同一个单GPU SLURM allocation中依次运行，保证同一物理GPU与环境：

```text
python -u scripts/train_object_locus_joint_v1.py --arm control
python -u scripts/train_object_locus_joint_v1.py --arm joint
```

两条命令由set -e的wrapper顺序执行；control失败不能启动joint。各臂仍fresh init/optimizer/RNG、各3584 updates，不相互续训。允许新增submit phase `formal_pair`，保留已有独立arm入口用于原样恢复。

13号V3 Expanded继续运行，不干扰、不迁移。11号忙则排队，不改变训练配方或GPU数量。

## 7. 继续顺序与禁止事项

当前代码尚未commit/push，正式更新0/0。沿用该实现继续，不能重写整个模型。

```text
上述窄修复 → 必要CPU tests → 新M1 → 完整一步3090 smoke
→ 原40步M2 → 丢弃全部临时训练状态 → commit
→ 非force合并远端新历史并push origin/main → verify SHA
→ fresh control3584 → fresh joint3584 → 原配对评估和报告
```

原M2判据、loss、β、注入限幅、optimizer/GC、seed、data order、epoch节点、splits、bootstrap、A/B/C判据不变。本补充不授权放宽M2或根据效果选择其他超参数。

合并远端时，两项共同修复作为明确登记patch保留；不能被原“protected files与b00b94f全等”检查误拒。若远端存在冲突的科学修改，按原规范STOP，不自行选一边。

禁止：加loss；放大M1容差；跳过visibility项；关闭Gaussian visibility weight；把NaN坐标替换0；改变官方evaluator/导出；启用DDP；从M2状态正式续训；先启动训练后push；自动扩数据/调参/追加实验。

完成后报告：共同修复两文件diff与SHA、CPU测试、M1包络、完整smoke和40步M2、显存、commit/push、11号两个arm所属job与物理GPU记录、正式更新与任务结果。若再次STOP，只报告首个明确阻塞点与证据，不新开一长串机制审计。


### 续接修复验证 PASS（2026-10-02）

额外白名单仅 canonical_recon.py 与 rendering/gs.py；固定共同解析投影正确性修复已登记SHA/diff。7项CPU测试通过。11号单张3090验证job58069完成：三个窗口×两臂M1全PASS，所有包络及差异均0；RGB/alpha/depth独立原rasterization对照exact；完整一步smoke通过；M2固定40updates、四层所有活动性判据与object-mean均通过，peak allocated/reserved=7.222070693969727/7.60546875 GiB。reference名称为V3 control + common analytic visibility correctness fix，不声称等价于历史错误visibility loss。旧58067/58068证据保存于validation/historical_attempt_58067_58068。未重训、重评旧实验。后续按合同commit、合并远端、CPU/M1最终树验证、push verify后正式两臂。


### 正式启动的基础设施修复

实现SHA308c069b5795ae4327fcce8080ae2ae2e3d631bc；第一次最终SHA5ff1c0adfba28d6f06eeeaf88a336520cf145076已push并在登录节点ls-remote核验。最终树CPU7项与M1+完整smoke job58070通过。正式启动58071/58072/58073均在Git远端请求前向前失败（localhost代理），58074清除代理后直连阻塞，未建立模型、未执行任何更新，取消释放自己的资源。修正正式gate：使用已经由登录节点真实ls-remote生成、与验证tree及HEAD绑定的remote_verified_sha；仍要求clean worktree、M1/M2通过、实现hash及完整SHA一致。不在计算节点重复联网，不更改任何模型、loss、优化器、schedule、GPU或数据。该实现修复按原顺序重新验证同一个fresh40步探针、提交推送、验证，再正式启动；不作为调参或新实验。

启动入口修复验证job58075：CPU7项、3窗M1 exact、完整smoke与同一个fresh40步M2全部PASS。模型、共同两项修复和所有科学配方未改变。临时状态全部丢弃，正式更新仍0/0。
