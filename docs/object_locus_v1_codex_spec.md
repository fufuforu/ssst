# Object-Locus V1：交给 Codex 的最终中文执行提示词

你是执行者。请在 /space/mawb/ssst 实现并训练本规格唯一注册的 Object-Locus V1。不得提出架构备选，不得自行调参，不得根据训练效果更换结构。用户已授权必要实现、检查、commit、push 和指定的 5000-step 正式训练；不要重复询问这些事项。

本任务的唯一主线：scene-conditioned object states + spatially guided evidence reading + 独立 ownership competition + residual object decoder。全程 joint optimization，第一阶段没有 object→anchor injection。optimizer-only FQ 不属于本任务。

## Assumptions fixed for Object-Locus V1

下面是设计者现在固定的选择；它们不是声称已由实验证明的最优超参数：

1. 起点是 fufuforu/ssst 的 main，精确基线 cf63b6ffb548f190136a64c8ebb4813d18ebdc80；代码版本不得用缩写代替。
2. 新模型类 LocusGSObjectLocusV1Recon，registry key siu3r_object_locus_v1，architecture_name LOCUSGS_OBJECT_LOCUS_V1；新增模块参数前缀 object_locus.。
3. 100 thing states、2 stuff states、1非query void channel。Object feature D=256；evidence attention 8 heads，head_dim=32；ownership/identity dimension=16。
4. 4个注册层共享同一个 object controller 的参数；候选选择使用本规格定义的空间/特征联合最远点算法；初始化 neighborhood K=16。
5. evidence 与 ownership 有独立 projections、独立归一化方向。两个 head 都使用 soft geometric bias；只有 thing 使用 bias，stuff不使用。
6. object decoder 使用 post-norm、FFN expansion=2、GELU、dropout=0。c/s 每层更新，ownership 使用更新后的 c/s。
7. auxiliary weight=0.25，L6/L8/L10 的 auxiliary loss取平均；不进行中间层 understanding pixel rendering。
8. visibility threshold 使用相对 GT depth tolerance 10%，不用预测深度、不新增 depth loss；单个可信view可以提供监督，多个可信view必须一致。
9. 新专用 provider 输出 depth_gt_m_all、depth_gt_valid_all、depth_gt_scene_all；原 Provider 和 SIU3RProcessedProvider 文件保持不变。
10. FP32训练，无autocast/GradScaler，B=1，无梯度累积。保持现有128 scenes/1024 windows、5000-step plan；正式阶段直接跑完整5000步，不设置短阶段指标gate、不自动延长。
11. checkpoint/eval节点固定为0/200/500/1000/2000/3500/5000。任务指标由未修改的SIU3R official evaluator计算；分别导出target-all与novel-only两个视图集合，避免把混合target指标称为novel。
12. 重建保持基线的 bounded delta、固定Gaussian decode radius=0.15、可训练anchor support radius。这里“固定decode radius”不等于冻结geometry。
13. 第一阶段不得加入新loss family。保留既有identity loss，不添加额外contrastive/triplet/diversity/compactness/depth loss。
14. 所有正式结果只报告，不自动执行第二阶段object→anchor feedback。

## 0. 已核对的仓库事实

本规格依据上面的精确基线源码制定，以下路径和函数是真实存在的：

- tokengs/models/anchor_group_locusgs.py：
  AnchorGroupController.encode_token、assign_group、update_group_states；
  LocusGSAnchorGroupRecon._group_readout、step_loss、forward_instance_state。
- tokengs/models/anchor_group_loss.py：
  project_points、resolve_anchor_observations、pairwise_anchor_bce_cost、
  build_anchor_targets、unified_hungarian、anchor_group_losses。
- tokengs/models/instance_state_loss.py：
  _check_labels、_flat_regions、_linspace_indices、_matching_cost、
  stuff_loss、semantic_loss、identity_loss。
- tokengs/models/canonical_recon_models.py：
  LocusGSRecon、patch_plucker_rays、_full_supervision、LocusGSRecon._layer_objective。
- tokengs/models/locusgs_recon.py：
  LocusGSAnchorDecoder、LocusGSGaussianHead。
- scripts/anchor_group_v1.py：
  build_options、build_optimizer、set_optimizer_lr、evaluate_anchor_group_all。
- scripts/anchor_group_v1_gc.py：
  backward_gradient_controlled；它按anchor_group.前缀区分参数，不可直接用于object_locus.而不适配。
- scripts/instance_state_generalization.py：
  _batch_for、_seen_classes、layered；旧_batch_for使用旧provider，不能为新visibility直接提供不存在的depth字段。
- tokengs/data/siu3r_processed.py：
  SIU3RProcessedScanNet.get_data、SIU3RProcessedProvider、pin_pair、
  validate_batch_frame_order；dataset读取uint16毫米depth PNG并除以1000得到米。
- tokengs/data/provider.py：
  Provider._preprocess；当前返回RGB、camera、labels，但没有depth输出。
- scripts/eval_instance_state_v1.py：
  evaluate_windows、_masks、_semantic_confusion、_instance_metrics。
  该文件只输出局部诊断，不能拿其local_panoptic值冒充official PQ，也没有mAP/AP50。
- scripts/export_instance_state_official.py：
  export_windows、write_official_pair、panoptic_and_semantic，提供格式参考。
  不直接调用旧exporter：它的best初始化shape、label_id偏移和void转换不满足本规格，改动放在新adapter。
- scripts/invoke_siu3r_official_evaluator.py：
  evaluate与CLI，固定SIU3R_COMMIT=8ea80166be76854f938e90521f1a5b688b755c87。
- 官方源码/space/mawb/SIU3R/src/evaluator.py：
  Evaluator.process_segmentation、setup、evaluate；
  读取pred.json的label_id后减1；context_map/target_map返回map与map_50。

既有final matching cost：
-P(gt_class) + 5*pixel_BCE + 5*pixel_Dice + 2*anchor_BCE + 2*anchor_Dice。
既有final understanding loss：
0.1*Lthing2d + 0.1*Lstuff + 0.1*Lsemantic + 0.01*Lidentity + 0.1*Lgroup。
no-object分类权重=0.1。
这些现有权重保持不变。

## 1. Git基线、工作区保护与文件边界

首先在原仓库只读记录：git status --porcelain=v1、git rev-parse HEAD、git diff --stat、当前branch、origin URL。
把该记录保存到本任务provenance。不得git reset --hard、git clean、自动stash、删除或改写已有未提交文件。

无论原工作区是否干净，执行：
- git fetch origin main；
- 验证origin/main精确等于cf63b6ffb548f190136a64c8ebb4813d18ebdc80。
- 从该commit建立干净linked worktree：
  /space/mawb/ssst_object_locus_v1
  branch：object-locus-v1-exec。
- 若该路径或branch已存在，仅在它有本任务provenance且基线一致时续用，否则STOP，不覆盖。
- 原/space/mawb/ssst中的Anchor-Group/FQ修改、未跟踪实验、用户文档和已有删除一律保留原样。
- 若remote main已前进，STOP并报告真实refs；不得强推、不得悄悄换基线或夹带新的main代码。

正式数据、pretrained、历史manifest使用下面的绝对路径；新源码在worktree运行。每个新脚本的REPO必须由Path(__file__).resolve().parents[1]得出，不能写死旧仓库导致import旧版新模型。

允许修改的已有源码只有：
1. tokengs/models/__init__.py：新增import/registry/__all__项目，不改旧registry条目。
2. tokengs/options.py：新增train_siu3r_object_locus_v1 preset及说明，不改旧preset；新增preset必须出现在AllConfigs构造之前。

允许新增：
- tokengs/models/object_locus_v1_controller.py
- tokengs/models/object_locus_v1.py
- tokengs/models/object_locus_v1_loss.py
- scripts/object_locus_v1_runtime.py
- scripts/train_object_locus_v1.py
- scripts/eval_object_locus_v1.py
- scripts/export_object_locus_v1_official.py
- scripts/smoke_object_locus_v1.py
- scripts/submit_object_locus_v1.sh
- tests/test_object_locus_v1_contracts.py
- docs/object_locus_v1_codex_spec.md

新报告仅写：
/space/mawb/ssst/group_plus/object_locus_v1/
新checkpoints仅写：
/space/mawb/ssst/workspace_group_plus/object_locus_v1/
不得覆盖同名历史结果；已存在且无本任务manifest时STOP。

禁止修改：
旧Anchor-Group/SM-RU/FQ、instance_state、group_locusgs、object_locusgs；
LocusGS reconstruction decoder/head、canonical_recon loss、encoder、renderer；
旧provider/data类、旧eval/export脚本；
/space/mawb/SIU3R内任何源码、config、数据和evaluator。
禁止无关重构、全仓库格式化、替换依赖、修改CUDA extension。

复用规则：
- 新模型直接继承LocusGSRecon，不继承会创建旧controller的LocusGSAnchorGroupRecon。
- 复用canonical encoder/decoder/head、patch_plucker_rays、_full_supervision、canonical reconstruction objective。
- 在新loss中复用旧纯函数_matching_cost、pairwise_anchor_bce_cost、project_points及三项stuff/semantic/identity loss。
- 新final loss从旧anchor_group_losses逐项迁移并因式分解，使pairs显式传入；不可调用旧anchor_group_losses导致第二次Hungarian。
- 新controller不实例化GRU、SM-RU或旧未使用的feedback projections。
- GC公式迁移到新runtime，按object_locus.识别新分支；旧GC文件不改。
- 复用RNG capture/restore语义，但不得调用旧脚本的train/audit主流程。
- 新eval/runtime需要_seen_classes和layered时，在新文件迁移这两个纯函数的原实现；不import scripts.instance_state_generalization或run_instance_state_v1，因为它们会将旧/space/mawb/ssst加入sys.path优先位置。新源码的import路径必须来自worktree；数据/pretrained资产仍使用规定的旧仓库绝对路径。smoke在provenance记录新model、controller、loss、runtime四个模块的__file__，必须位于worktree。

## 2. 新配置与pretrained初始化

新增preset以config_defaults["train_siu3r_anchor_group_v1"].evolve(...)建立，仅覆盖：
model_type="siu3r_object_locus_v1"
workspace="/space/mawb/ssst/workspace_group_plus/object_locus_v1"
experiment_name="siu3r_object_locus_v1"
batch_size=1
gradient_accumulation_steps=1
num_workers=0
seed=42
num_input_views=2
num_views=4
mixed_precision="no"
random_reflect=False
reconstruction_only=False
instance_state_coupled=False
instance_state_local3d=False
instance_state_layers=(6,8,10,12)
init_checkpoint=None
dataset_kwargs={"data_root":"/space/mawb/SIU3R/data/scannet"}

其余基线结构参数不变：
H=W=256，encoder/decoder token dim=1024，
enc_depth=3、dec_depth=12、patch_size=8、dec_patch_size=8，
1024 tokens，每anchor生成64 Gaussian，总65536 Gaussian。
相机normalization=first_cam、scale_method=constant，scene scale=0.15。
locusgs_bound_delta=True、locusgs_freeze_decode_radius=True、
decode radius=0.15，Gaussian z offset=0。
reconstruction supervised_layers=(6,12)，权重=(1/3,2/3)。

唯一pretrained：
/space/mawb/ssst/workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt
SHA256：
5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f
源step=47500。

只加载canonical reconstruction全部name/shape一致的tensors；排除object_locus.。
用独立canonical LocusGSRecon strict=True验证source，要求非object分支keys完整一致。
不加载旧理解checkpoint，不加载旧query_init，不加载旧optimizer state。
新分支在torch.random.fork_rng(devices=[])内manual_seed(31415)初始化。
全局Python/NumPy/torch/CUDA seed=42。
Linear weights Xavier uniform，bias=0；LayerNorm weight=1、bias=0、eps=1e-5。
MultiheadAttention的in_proj_weight和out_proj.weight也显式Xavier uniform，全部bias=0；不依赖不同PyTorch版本的默认初始化差异。
stuff_seed[2,256]唯一indexed learned seeds：normal std=0.02，无weight decay。
c/s residual heads weight/bias=0，详见第7节。
新分支不包含100个learned query_init，不包含每slot learned positional embedding。
所有reconstruction和object_locus参数requires_grad=True；canonical参数别名去重，不遗漏。
移除继承结构中的“reconstruction_only=True”语义覆盖，公开类属性也必须False。

## 3. Forward输入与严格执行顺序

模型encoder只能读取前2个context RGB/rays/cameras。初始化和attention不得访问GT语义、实例、depth或novel RGB。GT depth只在loss/监督构建中使用。

canonical decoder输出states[0..11]，注册层状态：
h_l=states[l-1]["tokens"]：[B,1024,1024]
mu_l=states[l-1]["mu"]：[B,1024,3]
r_l=states[l-1]["radii"]：[B,1024]。

全部模型浮点计算FP32，indices int64，validity bool，label int64。
B正式固定1；controller shape代码支持B>=1，既有identity loss正式B=1。

Object controller anchor encoding每个注册层执行：
g_i=[mu_i/ell, log(clamp(r_i/ell,1e-4,1e4))]：[B,1024,4]
a_l=LN_x(W_h LN_h(h_l)+W_m g_i)：[B,1024,256]。
LN_h=1024维，W_h=Linear(1024,256)，W_m=Linear(4,256)，LN_x=256维。
保留encode_token原有公式，但模块归属object_locus.。

L6计算固定scene normalization：
o=stopgrad(mean_i(mu6_i))：[B,1,3]；
ell=stopgrad(clamp(sqrt(mean_i(||mu6_i-o||²)),min=0.05))：[B]。
o/ell在本forward所有注册层固定，下一scene重新计算。

thing object：
q_t：[B,100,256]；
c：[B,100,3]；
s：[B,100,3]，各轴均为正。
stuff：
q_s：[B,2,256]，无c/s、不定义紧凑support。
拼接q_all=[q_t,q_s]：[B,102,256]。
void不是state，不参与self-attention。

每层顺序不得调整：
1. 读取本层canonical h/mu/r，不改写它们。
2. encode得到a_l。
3. 仅L6执行第4节初始化；其余层输入上一注册层q/c/s。
4. 用旧q/c/s计算第5节evidence logits、R和z。
5. cross-attention residual+norm。
6. 102个thing/stuff一起self-attention residual+norm。
7. FFN residual+norm，得到q_new。
8. 用步骤4的R和步骤7的q_new更新thing c/s。
9. 用a_l、q_new及更新后的c/s计算独立ownership A_l。
10. 从LN_cls(q_new[:,:100])计算本层class logits。
11. 保存本层metadata；q_new/c_new/s_new送入下一注册层。

不得额外初始化L8/10/12，不得使用A_l作为下一层hard evidence mask。
第一阶段没有feedback，因此允许先执行一次完整canonical decoder，再按6→8→10→12遍历保存的canonical states；这是唯一对象更新顺序，不能让四层并行独立更新。
canonical reconstruction张量不被object分支原地修改。

状态metadata固定：
q、c、s、anchor_embedding、evidence_attention、
anchor_assignment、thing_logits19、ell、scene_origin、seed_indices。
q是102 states，c/s仅100 thing。
正式输出还保留states原有12项及beta=0.0。
不为兼容旧diagnostics伪造A_pre等于evidence；新诊断读取新字段。

## 4. Scene-conditioned initialization：唯一算法

4.1 100候选的确定性联合最远点选择

使用mu6.detach()、h6.detach()，在CPU float64计算离散选择，防止GPU tie选择不稳定；选完indices转回原device。
f_i=normalize(LayerNorm无affine(h6_i),dim=-1,eps=1e-6)。
x_i=(mu6_i-o)/ell。
联合距离：
d_sp(i,k)=min(||x_i-x_k||²,4)/4。
d_feat(i,k)=clamp((1-f_i·f_k)/2,0,1)。
d_joint=0.5*d_sp+0.5*d_feat。

第一seed：argmin_i ||x_i||²。
后续每次选argmax_i min_{k已选} d_joint(i,k)。
所有tie均取最小原anchor index；已选候选排除。
精确选100个不同indices；零feature使用normalize的零向量输出；全部距离为0时仍按最小未选index填满100。
NaN/Inf输入是实现/数值错误，STOP，不静默替换。
不得用GT筛thing，不得预先过滤wall/floor，不得加入未经注册的objectness head。
此算法只生成hypotheses，绝不删除任何anchor。

4.2 每个seed的K=16空间neighborhood

CPU float64按||mu_i-mu_seed||²排序，tie最小index。
固定包括seed本身，其余15个取最近的其它anchors。
1024 anchors保证K=16；单测N<16时使用所有N，不padding复制；正式N!=1024报错。
σ=max(第16近anchor到seed的欧氏距离,0.05*ell)，离散距离/σ用于初始化pool weights时detach。
w_i=exp(-||mu_i-mu_seed||²/(2σ²))，仅16邻居；
w_i/=sum(w)+1e-12。
这16项指数范围可正常归一化；若sum<=1e-12，使用包含seed的one-hot fallback。

gather的a6和mu6不detach：
a_pool=sum_i w_i*a6_i：[B,100,256]。
c0=sum_i w_i*mu6_i：[B,100,3]。
s0_d=clamp(sqrt(sum_i w_i*(mu_i,d-c0_d)²+(0.05ell)²),
             0.05ell,2ell)。
p0=[(c0-o)/ell,log(s0/ell)]：[B,100,6]。
q_t0=LN_init(W_init*a_pool+W_pose*p0)。
W_init=Linear(256,256)，W_pose=Linear(6,256)，LN_init=256维。
c0/s0从可微gather计算，selection indices/初始w不求导。
不得添加100×256的trainable种子表。

4.3 stuff初始化

a_global=mean_i a6_i：[B,256]。
q_s0=LN_stuff(W_stuff*a_global[:,None,:]+stuff_seed[None,:,:])。
W_stuff=Linear(256,256)，stuff_seed[2,256]依次wall、floor。
不读取GT比例，不按GT mask pool，不设置stuff c/s。
后续stuff全局读取完整1024 anchors，geometry bias=0。

## 5. Evidence cross-attention：与ownership完全独立

共享跨层但独立于ownership的模块：
LN_ev_q、LN_ev_a；W_Q/W_K/W_V/W_O，均Linear(256,256)、bias=True。
8heads，每head32维。

Q=reshape_heads(W_Q LN_ev_q(q_all))：[B,8,102,32]。
K=reshape_heads(W_K LN_ev_a(a_l))：[B,8,1024,32]。
V=reshape_heads(W_V LN_ev_a(a_l))：[B,8,1024,32]。

thing geometry：
b_ji=clamp(-0.5*sum_d((mu_i,d-c_j,d)/(s_j,d+1e-6))²,-20,0)。
stuff j=100/101：b_ji=0。

L_bhji=(Q_bhj·K_bhi)/sqrt(32)+b_bji。
不用cosine normalization，不加额外temperature；scale唯一1/sqrt(32)。
不使用GT mask、depth mask、ownership mask或hard radius crop。
正式1024个anchors全部有效；非finite必须报错，不能把异常当masked token。
R=softmax(L,dim=-1)：[B,8,102,1024]，沿1024 anchors归一化。
z_hj=sum_i R_hji*V_hi。
z=W_O(concat_heads(z_h))：[B,102,256]。
R_bar=mean_heads(R)：[B,102,1024]；用于空间更新及诊断，梯度保留。

low-support明确处理：
没有“ownership mass低就跳过”的规则；softmax总有权重。
bias下限-20确保远处anchors仍有可读概率。
若全部anchors离某thing很远，照常softmax并执行受限c/s更新，不hard reset、不新建query。
有效anchor数量为0在本结构不可能；输入N=0或nonfinite为contract error。
不得复用ownership projection、ownership概率，或用A_l反向屏蔽R。

## 6. Residual object decoder

固定post-norm、D=256、8heads、dropout=0、FFN hidden=512、GELU。
L6/8/10/12共享同一个block；不为各层分别初始化参数。

cross：
q1=LN_cross(q_all+z)：[B,102,256]。

self：
使用nn.MultiheadAttention(embed_dim=256,num_heads=8,dropout=0,
                           bias=True,batch_first=True)。
t=SelfAttn(q1,q1,q1,need_weights=False)[0]。
q2=LN_self(q1+t)。
thing与stuff共102states一起self-attention；void不参加。
不加入ownership attention bias，不加入learned slot ID embedding。

FFN：
q_new=LN_ffn(q2+W2 GELU(W1 q2))。
W1=Linear(256,512)，W2=Linear(512,256)。
没有GRU、没有candidate norm matching、没有gamma=0.1 interpolation。
三个residual输出不得detach。
每个LayerNorm eps=1e-5，所有输出FP32。

## 7. 动态c/s：每层执行且参与梯度

只更新100个thing，使用本层R_bar[:,:100]和q_new[:,:100]。
不得使用ownership A_l求统计，也不得先算A再用它更新c/s。

evidence center：
c_ev_j=sum_i R_bar_ji*mu_l_i。
evidence support：
s_ev_jd=clamp(sqrt(sum_i R_bar_ji*(mu_l_i,d-c_ev_jd)²+(0.05ell)²),
               0.05ell,2ell)。

当前object残差：
d_c=0.25*(c_ev-c)+0.05*ell*tanh(W_c LN_geom(q_new_t))。
W_c=Linear(256,3)，全零初始化；
LN_geom=256维，供W_c/W_s共享。

L2位移限幅，最大单层位移0.25ell：
scale=min(1,(0.25ell)/(||d_c||_2+1e-6))。
c_new=c+scale*d_c。
不限制每个坐标绝对位置，不向object中心移动anchor/GS。

support使用对数正值参数化：
v=log(s/ell)。
d_v=0.25*clamp(log(s_ev/s),-log(2),log(2))
    +0.1*tanh(W_s LN_geom(q_new_t))。
W_s=Linear(256,3)，全零初始化。
v_new=clamp(v+d_v,log(0.05),log(2.0))。
s_new=ell*exp(v_new)。

每轴lower=0.05ell，upper=2ell。
不在object state之间normalize support，不用负半径，不用hard ownership统计替换。
除o/ell与离散初始化selection/w之外：
mu_l、R、q_new、c_ev、s_ev、c/s更新均保留autograd。
c/s是动态activation，不是新的per-scene持久nn.Parameter；其梯度通过初始化、evidence、W_c/W_s和ownership传递。
W_c/W_s尽管零初始化，L12 ownership使用c_new/s_new，因此final loss能监督这两个head。
stuff无紧凑c/s，geometry bias始终0。
不要加入center/size监督loss或geometry shrink loss。

## 8. Independent ownership head

独立模块：
LN_own_a、W_own_e=Linear(256,16)；
LN_own_q、W_own_u=Linear(256,16)；
W_void=Linear(256,1)，weight/bias零初始化。
这些参数不得与W_Q/W_K/W_V/W_O共享；evidence与ownership的LN也独立。

e=normalize(W_own_e LN_own_a(a_l),dim=-1,eps=1e-6)：[B,1024,16]。
u=normalize(W_own_u LN_own_q(q_new),dim=-1,eps=1e-6)：[B,102,16]。
feature logits F_ij=(e_i·u_j)/0.1。

thing ownership geometry b_ji按第5节同一公式重新计算，但使用更新后的c_new/s_new。
thing logit=F_ij+b_ji。
stuff logit=F_ij，不加geo。
void logit=W_void(a_l)：[B,1024,1]。
concatenate顺序：
0..99 thing、100 wall、101 floor、102 void。
A_l=softmax(logits,dim=-1)：[B,1024,103]。
该softmax沿103 ownership channels归一化。
geometry公式可以相同，feature head与probability表必须独立。
温度0.1固定，不learn logit_scale，不添加other entropy正则。

L12 ownership严格继承到Gaussian：
A_g=A12[:,:,None,:].expand(B,1024,64,103).reshape(B,65536,103)。
Gaussian顺序与activation_head原flatten顺序一致。
不做Gaussian-level reassignment，不新预测GS mask logits。

## 9. 分类、identity与readout接口

thing类别：
Z19=Linear(256,19)(LN_cls(q_new[:,:100]))：[B,100,19]。
LN_cls独立于其它LN，eps=1e-5。
0..17映射provider semantic2..19；18为no-object。
P=softmax(Z19,-1)。
公开thing_class_logits=[两列-1e4 pad,Z19]：[B,100,21]；
公开p_class：[B,100,19]。

identity沿用旧_group_readout数学结构，权重保持旧identity loss：
- parent e使用本节ownership的e；
- de=Linear(1024,64*16)(LN_de(h12))，零初始化；
- off=(GS_xyz-mu12)/0.15，逐轴clamp[-2,2]；
- W_off=Linear(3,16)，Xavier；
- eg=normalize(e_parent+0.25*tanh(de)+0.1*tanh(W_off(off)),eps=1e-6)。
这里的de只影响identity channels，不能修改ownership。
不添加旧proj_wh/proj_wmu/proj_wr/proj_wgs等未使用模块。

最终只把[A_g,eg]共119channels送入未修改的：
self.gs.render_feature_channels(gaussians,features,decoder.cam_view,decoder.intrinsics)。
M=前103channels：[B,V,103,256,256]；
E=后16channels：[B,V,16,256,256]；
alpha：[B,V,1,256,256]。

semantic scores：
S[:,:,0]=M[:,:,100]，S[:,:,1]=M[:,:,101]。
S[:,:,k]=sum_j M[:,:,j]*P[j,k-2]，k=2..19、j=0..99。
S_void=(1-alpha)+M[:,:,102:103]
       +sum_j M[:,:,j]*P[j,18]，保持channel维。
S为[B,V,20,256,256]，不额外softmax、不除alpha；
semantic_loss使用旧绝对渲染mass NLL。
void是未覆盖opacity、显式void ownership、thing no-object mass之和；不是query。

公开forward：
forward_object_locus(model_input,*,render_decoder_input=None,
                     context_decoder=None,coupled=False,step=None)
forward_instance_state(...)完全同签名并代理到它。
coupled=True明确raise；beta常数0.0，不依step变化。
step_loss(batch,*,step,phase="train",coupled=False)返回({"prediction":pred},metrics)。

prediction必须包含：
reconstruction、gaussians、render、states、beta、
assignment=A_g、anchor_assignment=A12、region_mass=M、
semantic_scores=S、pixel_void_mass=S_void、identity_render=E、
alpha、p_class、thing_class_logits。
gaussians保持旧[B,65536,14]，不得改变XYZ/opacity/scale/rotation/RGB顺序。
训练RGB渲染4views，理解feature渲染只2context views。
eval按请求decoder渲染context2或全部4/6views。
L6/L8/L10的aux不渲染任何understanding pixel features。
canonical L6 reconstruction RGB监督仍然保留；这不属于aux understanding rendering。

## 10. 数据接口与GT depth visibility

10.1 实际depth接口：新增专用provider，不改旧provider

在scripts/object_locus_v1_runtime.py新增：
class ObjectLocusV1Provider(SIU3RProcessedProvider)。
_override _preprocess使用与基线Provider._preprocess一致的完整签名，额外字段通过返回dict添加。

进入_preprocess时保存raw_depth_m=depths.clone()，它来自SIU3R PNG/1000。
调用super()._preprocess完成既有RGB/labels/cameras/rays链。
随后用本次self.image_transform.crop_transform(raw_depth_m)及
F.interpolate(...,size=output["images_all"].shape[-2:],mode="nearest")，
构造与RGB/labels对齐的GT depth。不得用preprocess_images对depth作bilinear resize。
valid=isfinite(depth_m)&(depth_m>0)；invalid置0。
输出：
depth_gt_m_all：[V,1,H,W] float32；
depth_gt_valid_all：[V,1,H,W] bool；
depth_gt_scene_all=0.15*depth_gt_m_all：[V,1,H,W] float32。
default_collate后分别[B,V,1,H,W]。
assert camera_scale_method=="constant"、camera_normalization_method=="first_cam"、
scene_scale==0.15、random_reflect==False。
这两个first_cam/constant设置已在本preset固定，无其它camera scale fallback。
depth与camera-z同为z-depth，不使用ray range代替。
字段不存在/shape不符为实现错误；禁止降级到predicted depth或全零GT。

新build_batch(opt,window,device)复制旧_batch_for的取scene/root/pin_pair/default_collate/move语义，
仅provider改成ObjectLocusV1Provider。
TRAIN_ROOT=/space/mawb/SIU3R/data/scannet/train；
VAL_ROOT=/space/mawb/SIU3R/data/scannet/val。
frame_ids必须严格等于context+novel。
GT只用于loss/评测，不传入controller/encoder或proposal selection。

10.2 每个注册层的visible anchor targets

新函数build_visible_anchor_targets(mu,batch)。
GT实例列表、classes、pixel masks继续按旧build_anchor_targets从前2context labels建立，
即使某实例没有trusted anchors也保留pixel GT。
每层使用自己的mu_l.detach()构建该层targets，targets不求导。

每view按旧project_points convention：
c2w=inverse(cam_view.T)；
x_cam=(mu-c2w[:3,3]) @ c2w[:3,:3]；
z=x_cam[:,2]；
u=fx*x/z+cx、v=fy*y/z+cy。
必须finite，z>0，0<=u<W、0<=v<H。
采样pixel ix=floor(u)、iy=floor(v)，不round。
D=depth_gt_scene_all[b,v,0,iy,ix]；
仅depth_gt_valid_all=True且D>0的pixel可检查。
depth consistency：
abs(z-D)<=0.10*D。
唯一相对tolerance10%，没有绝对额外项。
z>D*1.10为occluded，z<D*0.90为untrusted/off-surface；
二者均不给该view observation，不标void、不标negative。
invalid depth、out-of-frame、behind-camera、invalid semantic也不给observation。

可信observation：
semantic0→WALL；
semantic1→FLOOR；
semantic2..19且instance>0→THING(class,scene-global iid)；
semantic255或thing instance0→不提供label。
two-view consensus沿用resolve_anchor_observations：
- 无可信observation→IGNORE=-1；
- 仅一个可信observation→使用该label；
- 多个可信observation全部同wall→wall；
- 全同floor→floor；
- thing必须semantic与instance id全一致→该thing；
- 任何冲突→IGNORE。
visibility失败的view被排除，不能否决另一个可信view。
IGNORE不参与anchor cost/CE/Dice，不监督到void。
pixel-domain BCE/Dice/semantic/identity仍按旧规则使用context labels，不受anchor visibility gate影响。
GT semantic255始终IGNORE；GT stuff iid导出为0。

## 11. 唯一final Hungarian与loss

在新object_locus_v1_loss.py实现：
build_visible_anchor_targets、
final_hungarian、
loss_with_pairs、
aux_with_pairs、
object_locus_v1_losses。

每个forward只L12进行一次Hungarian solve（B=1）：
1. 用L12 mu和batch构建targets12。
2. 像旧unified_hungarian一样做deterministic pixel采样最多4096点；
   用_flat_regions保证[B,V,Q,H,W]先permute再flatten。
3. cost_class=-P(gt_class)。
4. pixel cost用旧_matching_cost，包括5*BCE+5*Dice；
   mask probability p clamp[1e-6,1-1e-6]后logit，不能把mass当logit。
5. anchor BCE用pairwise_anchor_bce_cost；
   anchor Dice=1-(2 pa@ya.T+1)/(sum(pa)+sum(ya)+1)。
   仅trusted anchor_valid；无anchor support的GT column的两个anchor cost置0。
6. total cost：
   1*class+5*pixel_BCE+5*pixel_Dice+2*anchor_BCE+2*anchor_Dice。
7. detach cost后SciPy linear_sum_assignment；
   qi/ki int64，按统一GT instance id对应。
8. 无GT给empty pairs；GT数量>100报错。
不要按query频率惩罚matching，不改变matcher类别softmax项，不再solve其它层。

final loss逐项沿用旧anchor_group_losses：
Lthing2d=2*CE19+5*BCEpixel+5*Dicepixel。
CE19：100thing全部监督；matched target=semantic-2，unmatched target=18；
class weight前18=1，no-object=0.1，
F.cross_entropy(...,weight=weights,reduction="mean")，
保留该函数weighted-mean denominator，不除matched count。
unmatched_noobj_scale=1.0，不做no-object ablation。
BCEpixel/Dicepixel仅matched pairs，按旧valid pixel规则/mean。
Lstuff=5*BCEstuff+5*Dicestuff，两个stuff平均，调用旧stuff_loss。
Lsemantic调用旧semantic_loss，0..19 valid pixels，255忽略。
Lidentity调用旧identity_loss，max_points=64、alpha_min=0.05、push_margin=0.2；
这是已存在的loss，不新增对比目标。
Lgroup=anchor_CE+anchor_Dice：
anchor_CE是在trusted anchors对应matched query/stuff channel上的-log(A)，mean；
anchor_Dice只对有anchor support的matched GT，沿用旧平滑+1，mean。
无有效anchor/无supported matched GT：相应项为图连接的0，不NaN。
无pixel valid满足旧semantic_loss的empty条件：真实数据错误并报告，不凭空制造标签。

Lfinal=0.1 Lthing2d+0.1 Lstuff+0.1 Lsemantic+0.01 Lidentity+0.1 Lgroup。

aux：L6/L8/L10分别使用本层visible targets，
GT全集排序按same scene-global iid对齐L12，pairs通过iid映射；
不得依本层support重排GT，不得重新Hungarian。
各层：
Laux_l=0.2*CE19_l+0.1*(anchor_CE_l+anchor_Dice_l)。
只含分类和anchor CE/Dice，anchor CE包含wall/floor。
没有aux pixel BCE/Dice、aux semantic、aux identity、aux RGB新目标。
三层平均：
Laux=(Laux6+Laux8+Laux10)/3。
Lunderstanding=Lfinal+0.25*Laux。
不得把三个aux直接相加；不得乘两次0.25。
所有既有IGNORE与probability clamp规则保留。

logged metrics同时报告每项unweighted与最终weighted值；
metrics["loss_understanding"]保留autograd，不能float/detach后反传；
metrics["loss_recon"]保留autograd；
metrics["loss"]=Lrecon+w(t)*Lunderstanding作为日志总目标，
实际backward按下一节GC，不直接对metrics["loss"] backward替代GC。

## 12. Joint reconstruction、GC与warm-up

reconstruction objective直接调用LocusGSRecon._layer_objective，
L6/12 canonical loss权重1/3、2/3，不改变：
每层Lrecon_l=MSE+0.2*(1-SSIM)/2
              +1.0*Lvis(Gaussian)+0.1*Lvis(anchor)。
LPIPS=0；GT depth不加到重建loss。
不冻结encoder、decoder、mu/refine_mu、rho/refine_rho、ray bias、activation_head。

understanding warm-up：
w(t)=0，0<=t<=200；
w(t)=(t-200)/800，200<t<1000；
w(t)=1，t>=1000。
aux已包含在Lunderstanding内，使用同一个w(t)，无独立aux schedule。
没有beta schedule，beta=0始终；不复用旧coupled=True路径。

GC：
model.zero_grad(set_to_none=True)。
Lrecon.backward(retain_graph=True)。
仅understanding backward期间，为所有非object_locus.的requires_grad参数注册hook grad→0.01*grad。
(w(t)*Lunderstanding).backward()。
finally移除hooks。
w=0跳过understanding backward，不要求object branch这时有非零grad。
结果：
reconstruction/shared：gR+0.01*w*gU；
object_locus：w*gU。
第一阶段没有object→reconstruction路径，因此gR不应贡献object_locus参数；
不detach a/h/mu破坏understanding→shared梯度。
不缩放完整total loss，不缩放已有gR，不把0.01再次乘到新branch。
clip_grad_norm_(全部model.parameters(),1.0,error_if_nonfinite=True)。
optimizer.step()。
下一步由model.zero_grad清理全部grad。

## 13. Optimizer、LR、precision和训练资产

四个optimizer groups：
object_locus_decay、object_locus_nodecay、
reconstruction_decay、reconstruction_nodecay。
object_locus前缀判定分支，其余canonical参数归reconstruction。
named_parameters去重后每个trainable param恰好出现一次。
no_decay条件：
p.ndim==1 OR name.endswith(".bias")
OR name.endswith("stuff_seed") OR getattr(p,"_no_weight_decay",False)。
其余matrix weights WD=0.05，nodecay WD=0。
AdamW betas=(0.9,0.95)、eps=1e-8、amsgrad=False、
foreach=False、fused=False，防止CPU/smoke与正式optimizer实现漂移。
object peak LR=1e-4，reconstruction peak LR=1e-5。

step前设置LR：
m(t)=t/200，1<=t<=200；
m(t)=0.02+0.98*0.5*(1+cos(pi*(t-200)/4800))，200<t<=5000。
object LR=1e-4*m(t)，reconstruction LR=1e-5*m(t)。
t=5000分别2e-6、2e-7。
5000是唯一scheduler总步数，不在1000步重启cosine。
FP32，batch1，accumulation1，无DDP、多GPU、autocast或GradScaler。

锁定训练manifest：
/space/mawb/ssst/group_plus/instance_state_v1_generalization/train128_windows1024.json
SHA256=1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483。
必须128个不同scenes、1024 windows。

锁定plan：
/space/mawb/ssst/group_plus/instance_state_v1_generalization/plan_C_frozen_5000.json
SHA256=ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323。
顺序step1..5000，scene/context/novel与manifest索引一致，原样遍历。
不得重采样、重做manifest、更换数据顺序、扩大数据集或改seed。

monitors原样复制到本任务reports：
monitor_train16.json，windows key，SHA256
133811557d13dca81864b9843b831f89df9226a899c381bfb79e0131991a5961；
monitor_8pairs.json，pairs key，SHA256
5dae7077779dd398ba97851f0259aef2df7de083df80f5a6bea4b5f5e731c321；
monitor_32pairs.json，pairs key，SHA256
af51dfe52d8cf31140f028a5c3b1d1bc5402fd524805d241940f314fad87bd36。
源dir同manifest父目录。
train128_class_coverage.json原样复制，_seen_classes必须得到set(range(20))；
不得屏蔽wall/floor或用旧错误coverage。
asset SHA不符STOP，不修改文件以迎合hash。

## 14. 新export adapter与真实SIU3R任务评测

不修改scripts/eval_instance_state_v1.py或任何SIU3R evaluator。
新模型兼容evaluate_windows；用arm="C"调用，batch_builder=新build_batch。
旧scope="target"包含全部请求views，不等于novel；保留其local结果但正确标为target-all。

新scripts/export_object_locus_v1_official.py按旧export layout写入：
<root>/<scene>_context<c0>_<c1>/
context_seg_pred、context_seg_gt、target_seg_pred、target_seg_gt。
只创建这些官方seg目录；PSNR直接用float rendering另存metrics，不为了seg eval生成假的depth。

GT-free panoptic assembly唯一规则：
semantic_raw=argmax_k S_k where alpha>0.05，否则internal void20。
thing_score_j=sum(P[j,:18])；thing_class_j=argmax(P[j,:18])+2。
eligible thing：score>=0.5、M_j>0.5、alpha>0.05。
每pixel在eligible queries中最大score_j*M_j者获胜；tie最低query index。
best初始化为zeros_like(M[v,0])：[H,W]，不能zeros_like(score_q)。
winner semantic=thing_class_j，instance=j+1，跨views同query同id。
无winner且semantic_raw为wall/floor且alpha>0.05：保留stuff semantic，instance=0。
其它无winnerpixel：internal void20、instance=0。
不以GT匹配决定pred labels/instance id，不添加NMS、面积filter或自调threshold。
local evaluate_windows原规则不改；新adapter规则按上面执行，差异如实记录。

official类别转换：
internal0..19→official1..20；
internalvoid20→official0，使用torch.where(valid_sem,sem+1,0)，不可clamp_max(20)。
PNG packed=official_semantic*1000+instance，
R=packed%256，G=(packed//256)%256，B=packed//65536。
每帧命名<scene>_pred<frame>.png，GT对应<scene>_gt<frame>.png。
GT255→official0且instance0；GTwall/floor→instance0；
GTthing instance0仍按原GT内容，不生成实例。
每个pred目录pred.json条目：
{"id":j+1,"label_id":thing_class_j+1,"score":thing_score_j}。
thing_class provider2→official label_id3，不得旧exporter的+2偏移。
每个PNG中非零pred iid必须有且仅有匹配pred.json条目；
stuff不产生thing JSON条目。
保存class probabilities与mask panel用于检查，不以check结果改变模型。

每个split、每个eval节点输出两个独立seg根目录：
official_all：context目录放前2context；target目录放原context+novel全部4/6views，
            保留当前target-all导出集合语义；
official_novel：context目录同前2context；target目录仅novel2/4views。
官方evaluator每个根目录分别运行；第二组明确叫novel-only subset。
不得把novel-only结果称为官方完整val_pair总表；
不得把train16/val8/val32 monitor结果称为完整1860-pair官方benchmark。

官方调用原脚本，不改参数实现：
/space/mawb/SIU3R/.venv_gpu_v4/bin/python
/space/mawb/ssst_object_locus_v1/scripts/invoke_siu3r_official_evaluator.py
--eval-path <上述根目录> --output <本节点官方结果JSON>
--device cpu --no-image-depth

CPU运行seg metrics，避免训练model/optimizer驻留GPU时官方metric再次占GPU。
官方repo HEAD必须8ea80166be76854f938e90521f1a5b688b755c87；
只读检查，不能checkout/reset用户SIU3R仓库。
seg eval必须同时启用semantic/PQ/mAP，不能--semantic-only或--recon-only。
读取原始result：
context_miou、context_pq、context_map["map"]、context_map["map_50"]；
target_miou、target_pq、target_map["map"]、target_map["map_50"]。
如果official返回缺字段、NaN或-1，记录原值及N/A原因，不伪造0、不用recall替代AP。
如无任何有效预测而官方metric返回无定义结果，这是模型表现边界，记录后继续训练。
运行错误/依赖缺失为评测基础设施错误，保留checkpoint并报告，不改evaluator。

PSNR：
context与novel分别在float RGB上逐window计算
-10 log10(mean((pred-gt)^2).clamp_min(1e-12))；
分别对windows取平均。另存target-all PSNR。
不把novel与context混合PSNR标为novel。
表格metric数值保留0..1，并注明不是百分比。

比较口径必须记录：
ssst输入使用GT camera poses；SIU3R本身是unposed。
相同evaluator不代表相同输入条件。
本轮5000-step128-scene结果是结构验证，不宣称已追平完整SIU3R指标。

## 15. 必要CPU contracts：限定范围

仅新增tests/test_object_locus_v1_contracts.py。
controller文件可独立import，不强制CPU加载CUDA renderer。
执行：
/space/mawb/anaconda3/envs/tokengs/bin/python -m pytest
tests/test_object_locus_v1_contracts.py -q

必须覆盖：
1. q/c/s、R、A、class logits shape与dtype。
2. 同输入重复初始化seed indices、neighbors、q/c/s相同；tie/重复坐标仍100distinct seeds。
3. 初始化使用16neighbors，但evidence后续N始终1024。
4. R沿anchor sum=1，A沿103channel sum=1，atol=1e-5。
5. evidence/ownership参数对象无共享，probability表shape/归一化不同；
   扰动ownership head不改变同输入evidence，扰动evidence head不直接替换ownership公式。
6. c/s finite，s bounds，各层c位移<=0.25ell+1e-6。
7. c/s由R和q_new更新，不依赖A；改变当前ownership head不会改变本层c/s统计更新。
8. detach只在指定位置；tiny synthetic graph下W_c/W_s/evidence/ownership/class heads有finite gradient。
9. GT visibility：behind/out-of-frame/invalid depth/±10%外→无observation；
   单view可信可用、两view冲突IGNORE、无观测IGNORE；
   所有ignore不监督void，GT实例全集不被anchor filtering删除。
10. L12 final Hungarian只调用一次；aux复用pairs/id映射；
    _flat_regions order正确，无support时anchor cost0。
11. losses权重/aux平均/warm-up公式正确、empty anchor loss可微0。
12. export shape、void0、thing类别+1、pred.json coverage、context/novel frame集合正确。
13. optimizertrainable参数无重复/遗漏，四group LR/WD正确。

不要新增PR threshold、cosine threshold、effectiveQ threshold、mask质量threshold；
不要跑历史长audit，不跑全tests目录，不跑1024windows机制审计。
CPU测试中小张量维度允许用controller内部helper；正式模型常量不得变。
测试失败修实现，不改正确参考值、容差或合同逃避问题。

## 16. RTX3090 smoke：固定真实batch和生产训练路径

使用单张NVIDIA GeForce RTX 3090，24GB级，
torch>=既有cluster环境，调用prepare_runtime(worktree)加载现有gsplat extension。
不改装依赖、不改renderer、不用4090结果冒充3090。

新smoke：
python -u scripts/smoke_object_locus_v1.py --device cuda
必须调用和正式train完全相同的build_model/build_batch/build_optimizer/train_one_step。
不另写“smoke简化模型”。

固定训练batch：
locked plan entries[999]，step1000；
window_index=241；
scene=scene0016_00；
context=[1506,1517]；
novel=[1509,1516]；
frame_ids=[1506,1517,1509,1516]。
该数据不存在或不一致STOP，不换简单scene。

smoke分为同一新鲜初始model上的必要检查：
A. 检查pretrained SHA、strict transfer、所有trainable参数/optimizer。
B. real forward与step_loss(step=1000)，w=1，记录loss各项。
C. 用同一个forward graph，先用autograd.grad(...,retain_graph=True,allow_unused=True)
   检查understanding到代表shared feature/geometry的梯度；这些检查不写入.grad。
   选定encoder/late-decoder、anchor_decoder.mu、activation_head有效参数；
   reconstruction梯度也检查graph connected。
   记录允许自然为0的分支；要求至少一个shared decoder understanding gradient非零、
   anchor geometry understanding gradient非零以及RGB decoder/head gradient非零。
D. model.zero_grad，然后生产GC backward、clip、optimizer.step。
   所有出现的grad finite；evidence/ownership/class/c/s residual heads必须至少各一个非零gradient。
   未使用参数不得出现：发现注册模块没有forward消费者应修复实现。
E. step后全部params/optimizer state finite，确认reconstruction与object参数实际有更新。
F. forward中R/A/q/c/s/GS/所有reconstruction及readout tensors finite，
   s bounds、位移限制、sum normalization满足合同。
G. 同模型eval官方val32第一pair：
   scene0011_00，context=[68,87]，novel=[69,70,71,72]，
   frame_ids=[68,87,69,70,71,72]。
   释放训练forward/graph，仅保留model/optimizer后no_grad推理；
   调用evaluate_windows一window的context/target两个scope；
   新official adapter导出all/novel，并运行未改official evaluator最小一次all seg eval。
   返回指标为0或无定义不属于smoke失败，运行/shape/接口失败才失败。

torch.cuda.synchronize后记录：
GPU name、total_memory；
before allocated/reserved；
训练forward/backward/step的max_memory_allocated/max_memory_reserved；
eval peak allocated/reserved；
总体peak，均bytes与GiB；
OOM与阶段、所有关键gradient名称/norm/finite、
grad clip pre-norm、四组LR、数值合同、evaluator状态。
reserved大于allocated正常，不据此宣称泄漏。

smoke optimizer一步后丢弃整个smoke model/optimizer；
正式训练必须重新seed、重新初始化、重新加载pretrained，不从smoke checkpoint续训。
smoke报告含代码文件hash，此时尚无正式commit，用hash attestation，不能虚构SHA。
OOM或nonfinite修正仅限本规格内实现/内存生命周期；
不得自动改precision、batch、anchor/query数量、loss、结构或LR。
若同一实现仍无法在3090完成，STOP并报告，不偷偷转4090。

## 17. Git顺序与正式作业

严格顺序：
implementation → CPU contracts → RTX3090 smoke → commit
→ push origin/main → fresh正式训练5000steps。

实现期间旧CPU/GPU合同已满足后不要启动旧long audits。
先将spec精确复制到docs/object_locus_v1_codex_spec.md。
源码commit只stage第1节白名单files，不能git add .。
CPU/smoke报告保存在reports，并记录checksum；
docs中提交精简结果引用与文件hash，不提交大规模PNG/JSON或checkpoint。
smoke的source attestation覆盖本任务Python/shell源码、registry和options；docs提示词在附加检查结果引用之前的规格内容另存spec SHA256。追加文档引用不改变已验收源码hash。正式checkpoint同时记录code hash、spec hash与实现commit。
commit message：
Implement Object-Locus V1 scene-conditioned instance states

push前再次fetch origin main，必须仍等于指定base；
git push origin HEAD:main；
不force、不绕过protected-branch拒绝。
push成功后git ls-remote origin refs/heads/main验证等于新commit SHA。
只有verified push成功才能sbatch正式train。
失败则STOP；不要先训练、稍后补push。

原main工作区未提交修改依旧原样保留；worktree branch保留。
正式driver启动也必须校验自身HEAD、新commit hash、源码白名单SHA256，
并检查remote main包含本commit；不得直接运行原dirty工作区中的新模型。

新增submit script SLURM：
--partition=3090
--nodes=1
--ntasks=1
--gpus-per-task=1
--cpus-per-task=8
--mem=64G
--time=24:00:00
--exclude=3dimage-13
输出/错误：
/space/mawb/ssst/group_plus/object_locus_v1/slurm-%j.out/.err。
不固定node3dimage-11，不允许改为多GPU。
固定python：
/space/mawb/anaconda3/envs/tokengs/bin/python。
working directory=/space/mawb/ssst_object_locus_v1。

支持两个明确phase：
sbatch scripts/submit_object_locus_v1.sh smoke
sbatch scripts/submit_object_locus_v1.sh train

smoke phase仅运行smoke脚本；正式train phase调用：
python -u scripts/train_object_locus_v1.py --device cuda
--reports /space/mawb/ssst/group_plus/object_locus_v1
--run-root /space/mawb/ssst/workspace_group_plus/object_locus_v1
--until-step 5000

CLI不得暴露可自由覆盖的architecture/loss/LR/query count/schedule参数。
仅device/reports/run-root/until-step；until-step首次必须5000。
slurm job time耗尽时允许恢复同一run到5000，不允许重置optimizer/RNG/plan或改变总步数。
记录sbatch返回job ID、实际GPU与节点。

## 18. 正式训练5000步、checkpoint/eval/日志

直接完整5000steps；无1000-step质量gate，无PR gate，无自动第二轮。
step0初始化后eval并保存initial checkpoint；这个step0位于push成功之后。
checkpoint与eval节点全部固定：
0、200、500、1000、2000、3500、5000。
每个节点checkpoint先写成功，再eval；节点后继续下一plan entry，不因指标差停机。

checkpoint路径：
<run-root>/checkpoints/step_00000000/train_state.pt，
其余使用8位step，目录有COMPLETE。
atomic写临时文件/dir后rename；已有完成checkpoint不得覆盖。
checkpoint必须包含：
model state、optimizer state、step、plan_position、
total_steps=5000、warmup=200、config与本spec版本、
Python/NumPy/torch/CUDA RNG、git_commit、source_file_hashes、
pretrained/manifest/plan/monitor hashes、architecture_name、
four optimizer group metadata、GC alpha。
恢复时严格核对，不缺字段静默fresh-start。
progress interruption是恢复，不是重新训练另一结构。

每次eval：
train16、val8、val32三组；
context、target-all local evaluate_windows；
新official_all和official_novel各一次official seg eval，
任务表取all的context metrics和novel根的target metrics，
并保留all的target指标原始JSON。
eval前后capture/restore全部RNG与model.train/eval模式，不改变训练随机流；
optimizer/parameters不可被evaluation改变。

log每100steps：loss每项、w、LR、GC、clip norm、显存、时间。
每个eval节点：
- official context/novel mIoU、PQ、mAP、AP50；
- float context/novel/target-all PSNR；
- thing/stuff mIoU附加用旧layered混淆矩阵计算，标local；
- 有效GT数、class-aware TP/FP/FN、raw recall50，标diagnostic；
- matched classification accuracy及正确类别margin，基于final pairs；
- 每query类别、score、mask area。
0与5000保存完整任务汇总，不能只输出提升最大的指标。

固定qualitative panels：
train16 positions=[0,5,10,15]；
val32 positions=[0,5,10,15,20,25,30,31]；
所有eval节点保存RGB GT/pred、semantic GT/pred、
panoptic/instance GT/pred、top5 score queries及其class/score/masks。
明确列出实际选择的window scene/frame IDs。
不得人工换scene以挑最好结果。

诊断只利用上述eval forwards与online训练matches：
PR/cosine(q和投影u)、evidence overlap、
ownership mass/concentration、GT best Dice、slot utilization。
PR计算对100thing features中心化后cov谱，报告定义；
online累计matches与单checkpoint eval utilization分开；
每window matched数/GT数一起报告。
不额外扫描全部1024training windows，不启动新mechanism experiment；
diagnostic为NaN/无定义时解释，不因低值停止。
c/s非finite、loss非finite是数值错误，必须停止。

训练后对val32 step0/5000比较：
报告official任务指标绝对差；
“有效object masks”使用GT-free导出mask与GT IoU>=0.5的一对一匹配，
分别给class-agnostic和class-aware TP及GT数量，
不能以query数/PR代替。
“保持reconstruction”仅报告PSNR delta并固定判据：
context和novel PSNR相对本run step0均不下降超过0.5dB；
超过则标未保持，仍不自动重训。
正式metric为无定义/N/A时不能宣布成功。
不设置未经注册的AP/PQ通过门槛，不以结果决定自动进入第二阶段。

## 19. 第一阶段禁止事项与失败处理

明确禁止：
- optimizer-only FQ主线实验或额外paired arm；
- global learned100-query seed bank作为thing初始化；
- GRU、SM-RU、norm-matched gamma update；
- object→anchor feature feedback，beta schedule；
- geometry向object中心收缩、实例compactness loss；
- hard radius crop、ownership硬mask evidence；
- Gaussian-level reassignment；
- 修改SIU3R evaluator、本地旧evaluator或旧模型；
- 新loss family、额外contrastive/triplet/diversity/depth loss；
- 删除既有identity loss以规避“不新增contrastive”的要求；
- 自动调LR/weight/temperature/neighborhood/query count；
- 新backbone、预训练模型、VFM特征、pred depth充当GT；
- 依据训练效果自动尝试其它方案；
- 正式训练前push以外的顺序；
- 夹带用户dirty文件、顺手重构。

STOP仅限：
基线/asset不一致；
实现/contract/shape/梯度路径错误；
3090 OOM；
loss/grad/参数/geometry非finite；
必须的evaluator接口实际不能运行；
git push未成功；
缺失真实数据/依赖/必需资源。
可修的本规格内错误先修复并重跑受影响检查；无需再开架构讨论。
资源排队可等待；时间耗尽可严格resume；
正式训练指标差、PR低、winner集中、slot利用率低不STOP。

若official metric只因模型空预测而返回无定义：N/A并继续；
若基础设施eval失败：保留已写checkpoint，报告blocked节点和真实exception，
不得修改evaluator造出指标。修复未改变评测定义的运行问题后resume同一run。
训练结果差只报告失败，不自动改结构/调参/重训。

## 20. Codex最终回复格式

最终必须按下面顺序给出真实值，未完成项明确pending/blocked，禁止虚构：

1. 基线commit、实现commit SHA、当前执行branch/worktree。
2. push origin/main状态及remote验证SHA；是否先push后正式train。
3. 修改/新增文件精确列表；原dirty工作区保留情况。
4. architecture summary：scene initialization、独立evidence/ownership、
   residual object decoder、动态c/s、无feedback。
5. tensor shapes：h/mu/r、a、q/c/s、R、A、Gaussian、class/readout。
6. CPU contracts：命令、pass/fail数量、失败项。
7. 3090 smoke：GPU/node、loss/gradient/evaluator结果、
   peak allocated/reserved GiB、是否OOM。
8. formal job ID、节点、开始/结束/当前step、计划与实际5000步状态。
9. checkpoint与eval节点表及路径；每个checkpoint的git SHA。
10. train16/val8/val32：
    context/novel mIoU、PQ、mAP、AP50、PSNR；
    step0/200/500/1000/2000/3500/5000的完整表或结果文件。
    N/A附原因，不能写成0或用recall替代。
11. 是否产生有效object masks：class-agnostic与class-aware TP/GT，
    分类准确率、固定qualitative文件。
12. 是否保持reconstruction：context/novel PSNR delta及0.5dB判据。
13. limitations：128-scene monitor验证、GT pose输入、非完整SIU3R benchmark。
14. 明确声明是否存在任何未注册改动；若有必须逐条列出，不能隐藏。
15. 原始logs、reports、checkpoints、task metrics文件路径。

本提示词到此为止。按此实施，不提出备选、不自行修改研究目标、不追加诊断实验链。

---

## Codex execution record

The source prompt above was copied byte-for-byte from `docs/Object_Locus_V1_Codex_Prompt.md` before this appendix. Original specification SHA256: `1bf3dc7c0affceaff9f6fac3299004f33f1eca33e1efb5b2a1c0fd0d18a1d395`.

- CPU contracts: 13/13 PASS. Report: `/space/mawb/ssst/group_plus/object_locus_v1/cpu_contracts.json`; SHA256 `c9b9edf78c5d4566886a0ea031599e4897a752073e2d452bd4b6f0016cdb1c1a`. JUnit: `cpu_contracts.junit.xml`; SHA256 `9f206d50800441d510b13efbe2ac6c4566f7cc29846c967418f833a1bbecd106`.
- RTX 3090 smoke: PASS, final source-attested SLURM job `56986`, node `3dimage-11`, PyTorch `2.7.0+cu126`, CUDA `12.6`, FP32. Peak allocated/reserved memory: `5.0371 / 5.3945 GiB`. Full report: `/space/mawb/ssst/group_plus/object_locus_v1/smoke_object_locus_v1.json`; SHA256 `35e4db81c2af27e3ff747a0e9f42223564a8cd62758a4a4447f5d47e84c826cc`.
- Smoke validated the fixed training window, production forward/backward/optimizer path, shared decoder and geometry gradients, and one-window val32 local plus official evaluator/export interfaces. Official mAP/AP50 were returned as `-1` (undefined) for the single-pair smoke and remain recorded as such.
- Smoke source SHA256 attestation and per-module paths are in `smoke_object_locus_v1.json`; CPU and GPU were validated against the 12 task Python/shell source files, registry, options, and contract test. Original dirty worktree files were left untouched.
- Formal jobs `56979` and `56984` failed before any training step, as recorded above. Job `56987` passed the remote and run-root checks and wrote the run manifest, then exposed a driver `NameError` (`MANIFEST` was not imported) before step 0 evaluation or any optimizer step; there is no checkpoint, curve, or training-metric record. The driver now uses the already SHA-verified manifest returned by `locked_assets()`. It permits manifest provenance refresh only when there is no checkpoint or training/evaluation progress and the task/spec/data/pretrained locks are unchanged; progressed runs still require exact commit/source identity. The pre-manifest startup-log rule remains. This recovery is startup-only and changes no model/loss/evaluation math. Formal training has not started.
