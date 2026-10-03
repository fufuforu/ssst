# Object-Locus Panoptic V1：预训练理解、双向生成交互与八卡训练实施提示词

以下全文交给 Codex 执行。你是执行者，不是架构决策者。完成实现、必要验证、提交推送和固定训练；禁止根据中途指标改结构、改 loss、调参或追加实验。

## 0. 本次目标和固定决策

本次建立具有可靠初始化的联合任务基线：保留 LocusGS 重建路径，接入 MASt3R 图像编码器和配套 COCO panoptic 预训练 adapter / Mask2Former；由预训练 object queries 形成 object states，在重建 decoder 内与全部 anchors 双向交互，最终在同一组 Gaussians 上预测几何、外观和实例 membership。

这不是仅替换随机 object seeds，也不是把现成 2D masks 反投影为最终答案。预训练编码器、adapter、pixel decoder、query decoder和 mask embedding 必须整体接入。理解与重建端到端联合训练。

正式训练只运行一个新结构实验：Fresh-128 相同的128场景、1008窗口，64 epochs；使用 `3dimage-13` 的8张 RTX3090，每卡1窗口，global batch=8。总窗口曝光64,512次，正式 optimizer updates=8,064。禁止再启动单卡对照、小规模先行训练、额外消融或1201场景训练。

八卡与 Fresh-128 单卡之间的优化差异明确保留在报告中：同样曝光量不等于同样更新次数。本轮是新联合配方的任务验证，不是严格的架构单变量对照；不得把全部差异归因于结构。

所有科学决策已经在下文写死。执行者仅能机械确定现有 Fresh-128 资产的实际路径、checkpoint实际键名和集群既有 partition 名称；必须记录实际值，不得借此改变设计。

## 1. 起点、工作区、文件边界

### 1.1 固定代码起点

- 仓库：`https://github.com/fufuforu/ssst`。
- 起点：`design-mast3r-panoptic-access` 的 `150a443ff0b1ac434dd0e7410164ebc77cb4c66d`，包含已收敛的接入规格和现有 V3 / evalfix / Joint 接口。
- 建立独立 worktree：`/space/mawb/ssst_object_locus_panoptic_v1`；新分支：`object-locus-panoptic-v1-8gpu`。
- 不在 `/space/mawb/ssst` 原 dirty 工作区实现；不清理、不暂存、不覆盖其既有修改。不改正在运行的 Fresh-128、官方 LocusGS ScanNet 重建实验及其目录。
- 只从 reconstruction pretrained fresh transfer。禁止接续 V3、Expanded、Fresh-128、Joint 的模型或 optimizer；这些旧任务仅提供数据清单与比较结果。

### 1.2 白名单

新增以下文件：

1. `tokengs/models/object_locus_panoptic_v1.py`
2. `tokengs/models/object_locus_panoptic_v1_controller.py`
3. `tokengs/models/object_locus_panoptic_v1_pretrained.py`
4. `tokengs/models/object_locus_panoptic_v1_lift.py`
5. `scripts/object_locus_panoptic_v1_runtime.py`
6. `scripts/train_object_locus_panoptic_v1.py`
7. `scripts/eval_object_locus_panoptic_v1.py`
8. `scripts/export_object_locus_panoptic_v1_official.py`
9. `scripts/smoke_object_locus_panoptic_v1.py`
10. `scripts/submit_object_locus_panoptic_v1.sh`
11. `tests/test_object_locus_panoptic_v1_contracts.py`
12. `docs/object_locus_panoptic_v1_codex_spec.md`

只允许另外修改 `tokengs/models/__init__.py` 和 `tokengs/options.py`，登记新模型与新 options，旧默认值不变。

新模型是 `LocusGSRecon` 的新子类；用新的 stateful decoder 子类/包装器实现交错循环。禁止原地修改 `LocusGSRecon._decode`、`canonical_recon_models.py`、`instance_state_locusgs.py`、任何旧 V1/V2/V2.1/V3/Joint model/controller/loss/runtime、provider、renderer、官方 evaluator。不得顺手重构。

理解模块从 `/space/mawb/SIU3R` 引用，核验源码 commit `8ea80166be76854f938e90521f1a5b688b755c87`。需要独立版本时，仅建立该 commit 的独立源码 checkout，不修改共享 SIU3R 工作区。记录源码与依赖版本。理解 wrapper 的改变写在本次新文件里；保留第三方许可证。使用 `/space/mawb/anaconda3/envs/tokengs/bin/python`、现有 FP32 PyTorch/CUDA/gsplat 运行链。缺依赖只修启动/构建兼容，不更换模型实现，不降级 PyTorch，不改旧环境内已工作的 evaluator。

## 2. 三份初始化权重与严格加载

### 2.1 重建

文件：`/space/mawb/ssst/workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt`。

- step：47500。
- SHA256：`5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f`。
- 复用 `scripts/object_locus_v3_set_runtime.py::transfer_object_locus_reconstruction_weights` 的450个重建 state tensors 的逐键、逐形状严格 transfer。
- 保留重建 options、空间坐标、Gaussian参数化、radius decode与 renderer配置。原有 `radius_decode_frozen` 是固定解码尺度的配置，不是冻结 geometry 参数，不能据此关闭梯度。

### 2.2 MASt3R 图像编码器

文件：`/space/mawb/SIU3R/pretrained_weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`。

SHA256：`e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2`。

只加载 `patch_embed.*`、`enc_blocks.*`、`enc_norm*`，共292个 state tensors。编码器为 ViT-Large：24 blocks、1024维、16 heads、patch16、RoPE100。不要实例化无用的 MASt3R geometry decoder / DPT heads；它们的725个 state tensors 明确排除。

### 2.3 COCO panoptic

文件：`/space/mawb/SIU3R/pretrained_weights/panoptic_coco_pretrain_vitadapter_maskdecoder_epoch60.ckpt`。

SHA256：`3f7d5d1a065913bfc0686942d979ba8b28f0230fec9a8eea1d4214d2e603eb20`。

- 从 `state_dict` 加载 `model.adapter.*`：187个 state tensors。
- 从 `model.mask2former.*` 加载 pixel decoder、transformer query decoder、query feature/position embeddings、mask embedding：326个 state tensors；排除分类头和 criterion。
- `model.backbone.*` 不加载，图像编码器明确来自 MASt3R。
- `class_predictor.*` 的2个 tensors不加载，重新建立 ScanNet 19通道分类头。
- 不调用原 Mask2Former criterion，不启用其 auxiliary losses。

把源键到目标键的一一映射、源/目标 shape、加载状态写入 `weights_mapping.json`。已声明的加载子树内部使用严格加载；不能以 `strict=False` 静默接受不匹配。计数包括 buffers，不等于参数数量；若实物与上述计数不符，列出具体键并停止，禁止少加载一部分后继续。

严禁用 `siu3r_epoch100.ckpt` 初始化任何模块。权重缺失或 SHA 不符才是前置错误；不得拿其他权重替代。

global seed=42。新增随机模块在同一 `torch.random.fork_rng(devices=[])`、object seed=31415 内初始化；先严格加载预训练，再初始化新模块，禁止遍历全模型重新初始化。各 rank 从 rank0广播完全相同的初始参数与 buffers。训练期各 rank 的独立 RNG seed=`42+100003*rank`，checkpoint保存每个 rank 的 RNG。

## 3. 输入与预训练理解前向

### 3.1 两个分支看同一内容

复用 Fresh-128 的真实 provider、`split_data`、batch及两张 context图像。重建仍为256×256、patch8，不改 provider或训练裁剪。

理解输入由这两张已经裁剪、范围 `[0,1]` 的256×256图像，`bilinear, align_corners=False` 插值为512×512，再转为 `2*x-1`。这样保证内容和裁剪一致；明确记录这是256图像上采样到预训练输入尺寸，没有凭空获得原生512图像细节。禁止另取不同帧或不同裁剪。

理解 encoder 使用 MASt3R 原生 image-only patch encoder：patch embedding→24 blocks→enc_norm。**不插入 SIU3R 新增的 intrinsics token，不调用其 geometry decoder**；该 token 没有本次许可的预训练来源。相机只在重建与3D特征读取中使用。保留完整24层输出列表，adapter仍按 `[5,11,17,23]` 原索引取特征，不把列表压成4项。

每视图 encoder token `[B,1024,1024]`。adapter沿用 `CroCoViTAdapter` 的完整 SPM、4个 interactions、level embeddings、`add_vit_feature=True` 的直接相加路径，输入为同一理解图像与完整 encoder层列表。

adapter输出每视图4尺度，通道1024，空间为128²/64²/32²/16²；堆叠视图后 `[B,2,1024,H_l,W_l]`。接入官方 `VideoMask2FormerModel` 的 pixel decoder与 query decoder，构造方式及其余默认配置精确复用上述 SIU3R commit 的 `_set_mask2former`，但只使用 model部分，不创建 criterion / COCO class head。100个预训练 query feature/position embeddings完整保留。

在新wrapper内按原 `VideoMask2FormerModel.forward` 的顺序调用 `pixel_decoder` 一次，保留其 `mask_features` 与 `multi_scale_features`，再调用 `transformer_module(...,word_embeddings=None,output_hidden_states=True,output_attentions=False)` 一次。不要为获取F_m再完整前向一次，不用只返回2D mask的外层segmentation wrapper替代稠密feature接口。

输出：

- pixel mask features `F_m: [B,2,256,128,128]`；
- 100个预训练 object states `q_pre: [B,100,256]`，取实际用于原生最终 mask prediction 的、经过 pretrained decoder layernorm 的最后一组 `transformer_decoder_intermediate_states`；根据源 `[Q,B,D]` 转为 `[B,Q,D]`；
- mask embedding MLP采用 pretrained `decoder.mask_predictor.mask_embedder`，不重新初始化。

adapter的 SyncBatchNorm在新 wrapper 中等价转换为 BatchNorm2d，保留加载的 affine与running buffers。所有 BatchNorm始终用 `eval()` 的预训练 running statistics，**affine参数仍训练**；防止每卡小 batch 和 rank0单独eval触发同步/统计漂移。其余模块正常 train/eval，预训练模块原始 dropout配置保持原值。不得把整个理解分支置为 no-grad或冻结。

## 4. Object–anchor 交错前向，顺序不得更改

### 4.1 Tensor与初始化

全部浮点为FP32：

- anchors：`h [B,1024,1024]`、`mu [B,1024,3]`、`rho/radii [B,1024]`；
- anchor embedding `a [B,1024,256]`；
- object states `q [B,102,256]`：100 thing+2 stuff；
- thing参考 `c,s [B,100,3]`，stuff没有紧凑空间support；
- Gaussian `[B,65536,14]`，仍64 children/anchor；
- 最终 membership `[B,65536,102]`，独立 sigmoid masks；不再要求children继承同一parent mask。

L6 geometry heads运行后，调用既有 `scene_normalization(mu6)` 得到 detached `origin,ell`，其中ell下限0.05。此坐标系整次 forward固定。

anchor编码精确复用 V3 `encode_token` 的公式及维度：

`a=LN256(W_h(LN1024(h))+W_g([mu/ell, log(clamp(radii/ell,1e-4,1e4))]))`。

新建对应参数，Xavier初始化，LN eps=1e-5；不从 V3 object checkpoint transfer。

thing初始 `q=q_pre`。stuff初始为 `LN(mean_{view,pixel}(F_m)+e_stuff)`，`e_stuff [2,256] ~ Normal(0,0.02)`，固定wall/floor顺序。

初始化c/s只确定空间参考，不选100 anchors、不丢其余924：

`R0=softmax_anchor(cosine(q_pre,a)/0.1)`，分别对每个thing沿全部1024 anchors归一化；

`c0=sum_i R0_ji*mu_i`；`s0=sqrt(sum_i R0_ji*(mu_i-c0_j)^2+(0.05*ell)^2)`，各轴clamp `[0.05*ell,2*ell]`。

q/c/s不detach；只有origin/ell固定detach。初始化不读GT labels、GT instance或最终 ownership。

### 4.2 注册层与重建循环

注册层固定L6/L8/L10/L12。**不能先完整调用旧 `_decode` 再遍历缓存states回写**；回写必须进入尚未执行的后续decoder。

新 decoder循环保留 `instance_state_locusgs.py::forward_stateful` 内原重建顺序与功能：当前mu/radii形成ray bias和位置编码→GS cross-attention→GS self-attention→GS MLP→mu/rho residual heads→activated radii。

每个注册层，随后严格执行：

1. L6在同一次迭代中先初始化q/c/s；L8/10/12使用上层object state。
2. 编码全部当前anchors，object读取anchor evidence。
3. object读取图像pixel evidence。
4. object self-attention、FFN更新q。
5. 从anchor evidence与更新q动态更新c/s。
6. 独立计算anchor接收object消息的路由，回写当前tokens。
7. 保存该层回写后的tokens及原mu/rho/radii，进入下一层。

L6/L8/L10的新tokens作为L7/L9/L11 cross-attention的输入；无需额外pending geometry/self-attention bias。L12回写后才调用原activation_head生成最终Gaussians。只有L12回写直接改变最终Gaussian生成输入；早期层通过后续decoder传导，同时保持原多层重建监督的读取接口。

不直接写mu/rho，不向object center收缩geometry，不回写encoder image features，不添加compactness self-attention bias。

### 4.3 Object evidence decoder

4个注册层参数**不共享**。每层新 decoder固定dim256、8 heads、head dim32、FFN512、GELU、dropout0、post-norm、LN eps1e-5。

Anchor evidence：独立W_Q/K/V，标准多头点积 `/sqrt(32)`；thing加 `-0.5*sum_axis((mu_i-c_j)/s_j)^2`，stuff几何bias=0。**不使用旧 `_geometry_bias` 的 `clamp(-20,0)`**，避免远场同地板；FP32有限logits经过稳定softmax，沿全部1024 anchors归一化。无hard radius mask、无ownership mask、无support阈值裁剪。`R [B,8,102,1024]`，`Rbar=mean_heads(R)`，evidence=`W_O(concat_heads(sum_i R*V_i))`。

图像evidence：F_m bilinear缩到32×32，每视图1024个、合计2048个256维feature。对这些features做LayerNorm后，使用另一套8-head W_Q/K/V/O，标准 `/sqrt(32)` cross-attention，softmax沿2048 image locations。不读novel RGB或novel labels。

严格残差顺序：

`q1=LN(q+CA_anchor(q,a))`

`q2=LN(q1+CA_image(q1,F_image))`

`q3=LN(q2+SA(q2,q2,q2))`（thing/stuff一起）

`q_new=LN(q3+W2(GELU(W1(q3))))`。

新attention的输入投影Xavier、bias0，输出W_O与bias0；self-attention out_proj=0，FFN W2/bias=0，其他Xavier。初始化时保留预训练query的信息，训练后可适配。**不是将全部MLP参数置零**。

### 4.4 动态c/s

精确采用 V3 `update_geometry` 的更新机制，每个注册层都运行，使用Rbar与q_new：

`c_ev=sum_i Rbar_ji*mu_i`

`s_ev=clamp(sqrt(sum_i Rbar_ji*(mu_i-c_ev)^2+(0.05*ell)^2),0.05*ell,2*ell)`。

`dc=0.25*(c_ev-c)+0.05*ell*tanh(W_c(LN(q_new_thing)))`。

将dc的每个object向量范数限到0.25ell：`c_new=c+dc*min(1,0.25*ell/(norm(dc)+1e-6))`。

`v=log(s/ell)`；

`dv=0.25*clamp(log(s_ev/s),-log2,log2)+0.1*tanh(W_s(LN(q_new_thing)))`；

`s_new=ell*exp(clamp(v+dv,log0.05,log2))`。

W_c/W_s的weight和bias=0。c/s及统计不detach，不从最终Gaussian membership计算。stuff不运行这组更新。

### 4.5 Object→anchor 路由与注入

路由必须独立于anchor evidence R；R按anchors归一化，不能直接转置当消息权重。

对更新后q，计算 `m_q=pretrained_mask_embedder(q)`，包含thing/stuff共102个；`f_a=LN(W_a(a))`。两者用于路由的cosine相似度（L2 normalize eps1e-6）：

`l_ij=cos(f_a_i,m_q_j)/0.1 + geo_ij`，thing几何bias与4.3相同，stuff=0。

void路由logit固定0，不设新void分类loss。

`A_route=softmax_channels([l_thing100,l_stuff2,0_void])`，shape `[B,1024,103]`；**沿103接收通道归一化**。

`message_i=sum_{j=0..101} A_route_ij*LN(q_new_j)`，void贡献0。

每注册层有独立 `W_inject: Linear(256,1024,bias=False)`，weight全部0；属于新模块1e-4组。

`sigma_i=sqrt(mean_channel(h_i^2)+1e-6)`；

`d_raw=sigma_i*tanh(W_inject(message_i)/sigma_i)`；

将d_raw向量范数限到 `0.25*norm(h_i)`，分母eps1e-6，得到d_cap。

`h_new=h+beta(n)*d_cap`，`beta(n)=0.1*min(n/1000,1)`；n为已经完成的全局窗口曝光次数，step0 n=0。正式第u次更新前用n=8u计算注入与understanding权重，u从0开始。

geometry只由原head及后续decoder改变；不加几何注入支路。route、q、message及注入对允许的参数保留梯度。

## 5. 遮挡感知图像特征→Gaussian membership

### 5.1 固定几何的可微feature lifting

先得到最终Gaussians，然后从两张context稠密mask features读取证据。F_m插值到256×256，使用重建256分辨率的真实cam_view/intrinsics。

对固定几何G及零背景，既有 `GaussianRenderer.render_feature_channels` 定义线性feature合成器：

`S_G(X)_{v,p,d}=sum_i c_{i,v,p}*X_{i,d}`。

c含同一EWA足迹、opacity和沿深度排序的透射率。使用既有gsplat实现，不能用Gaussian投影中心采样，也不能用numpy audit脚本替代训练算子。

定义转置读取：

`L_G(F)_{i,d}=sum_{v,p} c_{i,v,p}*F_{v,p,d}`；`mass_i=L_G(ones)_i`。

实现新 `torch.autograd.Function`，避免依赖gsplat二阶梯度：

- forward：G/cameras/intrinsics detach；`torch.enable_grad()` 下创建零dummy feature X，调用原feature renderer，通过 `autograd.grad(S_G(X),X,grad_outputs=F.detach(),create_graph=False)` 得到L_G(F)；返回值由custom Function持有对F的梯度。
- backward：输入dY，返回 `S_G(dY)` 作为对F的梯度，G/camera/intrinsics返回None。
- 特征通道固定按32分块，256维共8块；mass单独1维计算一次；禁止构造 `[65536,V,H,W]` 或Gaussian×pixel×channel的稠密贡献表。
- 此处几何读取权重固定；其他child feature、Gaussian生成与最终mask rendering路径仍对geometry/重建参数反传。不能扩大detach范围。

`e_i=L_G(F)_i/clamp_min(mass_i,1e-6)`；`g_i=mass_i/(mass_i+1.0)`。mass=0时e=0且g=0。1.0的单位是当前256网格上的一个累计合成贡献单位，不是meters或radius。

### 5.2 Child fallback与融合

新建V3同形状child feature路径，复用其几何编码公式：parent embedding a256、每child14维geometry、64×16 child index embedding；MLP `286→256→256`，GELU，最后Linear weight/bias=0。归一化delta/logscale等clamp、eps精确沿用 V3 `gaussian_child_features`。f_parent=`LN(W_a(a))`；`f_child=f_parent+child_residual`，shape `[B,65536,256]`。

新 `W_res: Linear(256,256)` weight/bias=0。融合固定：

`f_i=g_i*e_i+(1-g_i)*f_child_i+0.1*tanh(W_res(f_child_i))`。

无支持Gaussian仍由自身child路径预测，禁止赋零mask或直接赋void。不能把2D mask概率加权平均当最终membership。f_i不再额外整体L2归一化，以保留预训练mask feature幅度。

### 5.3 Mask与分类

最终L12 `m_q=pretrained_mask_embedder(q_new)`；

`gaussian_mask_logits_iq=sum_d f_id*m_qd`；`membership=sigmoid(logits)`，shape `[B,65536,102]`。

**点积scale固定1.0，不除sqrt256**，沿用原生Mask2Former mask prediction幅度；这是对接入草案未定公式的最终裁定。102个mask独立，不对slot做softmax，route不是membership。

最终thing classifier=`Linear(256,19)(LN(q_final_thing))`，新头Xavier、bias0，18个thing类按现有ScanNet semantic id2..19映射到logit0..17，logit18为no-object。wall/floor是mask100/101，不加入thing分类。

为复用现有evaluator，完整提供 `thing_logits19`、`p_class=softmax(logits19)`、`conditional_class_prob=p[:18]/clamp_min(sum(p[:18]),1e-12)`、`objectness_prob=1-p_noobject`、`thing_class_logits=[-1e4_wall,-1e4_floor,logits19]` 的21通道兼容字段。不可用dummy零替代实际预测字段。

把同一Gaussians及membership送既有feature renderer，以alpha归一化生成 `[B,V,102,256,256]` 的 `region_mass`；复用V3 `alpha_normalize_membership`，void=`1-alpha`。按既有V3 semantic readout形成20类semantic scores。独立候选输出与packed panoptic输出仍分开，Gaussian RGB、mask与semantic采用同一组geometry。

target-all/novel预测只能用context图像生成q与f_i，再以requested cameras渲染；严禁读target RGB/features/labels参与预测。labels仅用于loss/evaluation。新前向显式区分 `read_context_decoder`（始终两张context，用于lifting）与 `render_decoder_input`（本次请求的context/target-all/novel cameras，用于RGB/membership渲染）。训练请求为已有两context加两novel，理解loss仍只取前两context；不能把读取相机和输出相机混用。

## 6. Loss、梯度与优化器

### 6.1 不新设计loss

直接调用未修改的 `tokengs/models/object_locus_v3_set_loss.py::object_locus_v3_set_losses`。最终L12唯一一次Hungarian；仅两张context的GT参与理解训练，沿用真实batch的 `semantic_label_all`、`instance_label_all`、valid域与IGNORE逻辑。

匹配cost固定 `2*(-p_class)+5*mask_BCE+5*mask_Dice`。训练理解loss固定：

`L_under=0.1*(2*L_class+5*L_thing_BCE+5*L_thing_Dice+5*L_stuff_BCE+5*L_stuff_Dice)`。

19类CE no-object权重0.1；matching里的概率clamp1e-6只用于cost；训练BCE为既有概率域实现，Dice平滑1。不得改为logit BCE或增加clip。GT构建直接复用 `build_context_instance_targets` / `final_hungarian`，不另造visibility/anchorGT监督。

重建loss精确调用原 `_layer_objective` 与 `_full_supervision(batch)`；保留原注册重建层及所有权重，不改pixel监督。没有L6/8/10 object auxiliary、2D auxiliary、anchor CE、identity、semantic CE、contrastive、triplet、entropy或compactness loss。

understanding warm-up按全局曝光n定义：

`w_under(n)=min(n/200,1)`，step0为0，200次曝光之后为1。

这是复用 V3 正式 train 脚本传入的 `min(step/200,1)`，转换到曝光轴；不要误用 runtime 中未被该正式脚本采用的200→1000延迟函数。总objective显示为 `L_rec+w_under*L_under`；GC是梯度规则，不偷偷乘到日志loss里。

### 6.2 三类参数组

| 参数族 | peak LR | WD |
|---|---:|---:|
| 全部原重建encoder、GS tokens/decoder、geometry、activation head | 1e-6 | 0 |
| MASt3R encoder、预训练adapter、pixel/query decoder、预训练query和mask embeddings | 1e-5 | 矩阵0.05；bias、LN/BN、embedding及其他1维参数0 |
| 新anchor编码、object decoder、stuff states、c/s heads、路由投影、注入、child/fusion、ScanNet class head | 1e-4 | 同上0.05/0规则 |

所有参数requires_grad=True；不冻结backbone、geometry、pretrained理解。BN running buffers保持固定不算冻结参数。

mask embedder在模型中只注册一个owner，object更新/route/final mask通过访问同一模块调用，禁止重复注册别名造成state keys和optimizer参数计数混乱。stuff_seed与child index embeddings显式no-weight-decay。

AdamW betas=(0.9,0.95)、eps=1e-8；FP32无AMP，无TF32；global grad norm clip=1.0。所有rank参数组名称/顺序一致，参数无重复/遗漏。严禁将注入放进reconstruction的1e-6组。

LR按当前更新将处理的曝光数 `t=8*(u+1)`：前200曝光线性 `t/200`，随后 `0.1+0.9*(1+cos(pi*(t-200)/(64512-200)))/2`。三组乘同一multiplier，最后为peak的10%；**不按8卡线性放大LR**。

### 6.3 八卡与GC的实际实现

用 `torchrun --standalone --nproc_per_node=8`、NCCL同步数据并行。为了保留“两种loss对重建参数具有不同梯度缩放”，**采用显式梯度同步，不套 `DistributedDataParallel` reducer**；不得把旧的两次backward/hooks原封不动放进DDP引起重复ready或提前reduce。

每rank同一forward获得两loss。按固定sorted参数顺序，使用 `autograd.grad(...,allow_unused=True)` 分别得到 `g_rec` 与 `g_under`；仅第一组保留图，第二组释放；w_under=0时第二组直接None。第二组对应 `w_under*L_under`。

原重建参数：`g_local=g_rec+0.01*g_under`；理解及全部新模块：`g_local=g_rec+g_under`。None项按0加；两项均None则记录unused。

先同步一份“梯度存在”的bitset（MAX），所有rank都无梯度的参数保持 `.grad=None`；其他rank缺的梯度补0。按sorted参数与固定25MiB bucket进行 `all_reduce(SUM)/8`，还原.grad，再全局clip1.0、AdamW一步。所有rank执行完全相同的collective顺序；不允许仅同步object参数。

这就是global batch8的同步数据并行；不声称用了PyTorch DDP wrapper。正式global更新不是八个独立optimizer步骤，不是每卡分别训练一个模型。每个epoch126次更新，8卡共同推进同一模型。

保持允许梯度：under→pretrained理解/新模块；under→重建以0.01缩放；rec→重建正常；rec→理解/新模块经object注入正常。只有feature lifting的读取几何权重detach，不允许其他 blanket detach。

## 7. 数据、八卡采样、固定训练量

### 7.1 精确复用Fresh-128资产

Fresh-128已执行job58078。根据其Slurm Command/WorkDir/StdOut及run_manifest定位真实manifest与plan；若Slurm历史不可读，仅在 `/space/mawb/ssst/group_plus` 与 `/space/mawb/ssst/workspace_group_plus` 搜索满足以下全部事实并有该run日志关联的唯一manifest：fresh reconstruction step47500、128scenes、1008windows、64epochs、64,512单卡updates。

这是现有资产定位，不是数据设计权。输出实际路径与SHA256，复制其1008窗口有序列表、固定split清单及所有frame identities到本run。不能用旧V3 Expanded continuation 32epoch plan替代Fresh-128，也不能从1201场景再抽样。

精确复用Fresh-128全部已注册评测splits，并额外确保原V3的 `train_all56`、`same_scene_holdout8`、`dev8`、`val32` 存在；已有同义字段只映射名称，不新增窗口。expanded train probe与16-window same-scene holdout如在源manifest中，原样保留。验证train/dev/val的scene交集并记录；名称不代表unseen，是否unseen由真实交集决定。训练scene/holdout frame泄漏检查一次即可。

源资产不唯一或不存在时，停止并报告缺哪个具体文件；禁止执行者临场构造一个“差不多”的128场景清单。

### 7.2 采样规则

epoch从0到63。`perm=np.random.default_rng(42+epoch).permutation(1008)`。

该epoch第k次更新（k=0..125），rank r读取 `windows[perm[8*k+r]]`。每rank本地batch1，禁止另加DistributedSampler的shuffle/padding，禁止drop_last或重复窗口。1008恰好被8整除。

每个window恰好64exposures；每rank每epoch126windows；global每epoch1008windows/126updates。累计曝光n=`8*completed_updates`。

正式训练一次完成64epochs、8064updates，不设中途任务指标gate，不因AP暂时为零停止。可因数值或真实实现错误停止，不能自动换配方续训。

| Epoch完成节点 | 新global updates | 已完成窗口曝光 |
|---:|---:|---:|
| 0 | 0 | 0 |
| 2 | 252 | 2016 |
| 4 | 504 | 4032 |
| 8 | 1008 | 8064 |
| 16 | 2016 | 16128 |
| 32 | 4032 | 32256 |
| 64 | 8064 | 64512 |

日志、checkpoint同时写epoch、update、exposure三种计数。不要把8064写成“只训练了Fresh-128的1/8数据”。

## 8. 显存与作业

固定正式节点 `3dimage-13`；Slurm单节点、`--ntasks=1 --gres=gpu:8 --cpus-per-task=32 --mem=128G --time=72:00:00`。partition使用该节点当前已有GPU partition实际名称；不擅自改到11号。8卡未空闲就排队，不占其他运行任务的GPU。

FP32、每卡batch1、context2、256/512分辨率及模型不得因OOM临场改变。预先实现non-reentrant gradient checkpointing：MASt3R encoder每block、adapter interactions、预训练query decoder每layer、原重建decoder每block；`preserve_rng_state=True`。自定义lifting不二次checkpoint。checkpoint函数内禁止日志append、state写入等重复执行副作用。BN running stats固定，避免重计算更新buffer。

不要求理论8倍提速。记录端到端训练吞吐（windows/s、updates/s）、每个rank显存、同步时间、eval时间；额外理解模型和gradient通信可能限制加速。

报告目录 `/space/mawb/ssst/group_plus/object_locus_panoptic_v1_8gpu`；run目录 `/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_v1_8gpu`。不覆盖旧run。

epoch0/2/4/8/16/32/64保存model checkpoint；完整optimizer+8rank RNG恢复点只滚动保留最新和上一份。先写临时文件，完成并校验后rename，再删本run旧非注册恢复点。依据实际参数字节估算2份full+7份model+报告空间，禁止又按每epoch保存完整optimizer耗尽磁盘。不得清理其他实验来偷偷满足预算，已有用户清理授权也不能覆盖活跃Fresh-128和本任务三份pretrained。

## 9. 一次必要验证，随后开始正式训练

### 9.1 CPU contracts

仅检查实际错误：tensor shape、4尺度/24层接口、严格加载及禁止权重、参数组覆盖、1024 evidence softmax轴、103route softmax轴、102 independent sigmoid、c/s有限与上下界、zero injection初始化、数据64exposure/rank分配、class/no-object/stuff映射、sampler无泄漏。

加一个小型线性模型检查GC梯度同步公式：8个sample分别求上述两loss梯度再平均，应与按样本平均objective、重建梯度0.01缩放的参考一致，FP32 `rtol=1e-5,atol=1e-7`；不拿整模型gsplat反传逐位等价作gate。

### 9.2 3090单卡真实smoke

使用固定plan epoch0 permutation的第一个真实window，临时模型，完整forward→两loss→GC梯度组合→clip→optimizer step；连续2次。检查F_m/q/c/s/route/Gaussian/membership/RGB/alpha/loss/关键梯度有限，无OOM；验证under对pretrained mask decoder有梯度、under对重建路径可回传、child与class路径可回传。零注入首步上游部分梯度为零是预期，不要求每个参数首步非零。

对lifting执行一次真实小通道adjoint关系检查 `<S_G(X),F>≈<X,L_G(F)>`，scale使用全内积绝对项归一化误差，门槛1e-4；再确认对F的反传非零且不要求geometry从此读取算子得到梯度。只跑一次，不追加visibility纯度审计链。

local evaluator和single-pair official export/evaluator接口必须通过，AP undefined可保留，不作gate。

### 9.3 八卡同步smoke与活动性

在13号节点8卡，临时fresh模型，从plan第0更新开始运行**40个global updates（320exposures）**。这是唯一短活动性检查，不是正式训练；结束后彻底丢弃模型、optimizer/RNG/plan位置。

检查各rank参数/optimizer step一致、plan曝光集合一致、梯度全量同步无遗漏、各rank峰值显存；更新40时4层W_inject、注入delta非零且finite，以及从reconstruction loss到至少一个pretrained理解参数存在finite非零梯度。若注入始终为零，只修明确断图/顺序/初始化实现错误，不自行增大beta或LR。通过后不追加50/100/200步审计。

注入关闭时的重建骨架在同权重同输入下与原重建路径作一次对照；使用原路径两次重复得到的数值噪声包络，并比较tokens/mu/rho/Gaussians/RGB。renderer非确定性不需要逐位等价；若只有原路径重复也出现相同render/loss抖动，不通过增大设计容差解决，记录该事实，不扩展到renderer审计。不能要求新结构理解输出等于随机V3。

必须保存每rankallocated/reserved显存、各关键参数组梯度norm、loss分项、输入windowidentity。只保留真正的contract；PR/cosine/effectiveQ/纯度/slot数/AP不是启动gate。

## 10. Git与正式启动顺序

严格：**实现→CPU contracts→单卡3090 smoke→八卡40-update smoke→commit→push并验证→正式训练**。

白名单提交；不提交权重、checkpoint、数据、结果图片。先把新分支推送到origin，再把实现安全集成到 `origin/main`，不得force push，不覆盖其他对话已推送的提交。只有普通git合并/注册项冲突允许解决；若影响本次模型科学定义则停止，不偷偷吸收另一版结构。

推送后核验训练SHA已存在远端；记录训练实际SHA和main SHA。正式job从已推送且clean的本任务代码运行。正式fresh模型重新从三份权重构建，RNG/optimizer/plan从头开始，smoke更新不计入正式8064。

正式每10次global updates记录一次loss分项、三组LR/grad norm、beta、注入活动性、每卡显存与吞吐，epoch边界额外记录一次。训练不逐步导出PNG；固定评估节点才导出。forward/loss阶段的本rank异常先捕获，所有rank同步成功/失败标志后才进入autograd和梯度collectives，防止一个rank提前抛错导致其余rank挂起。

仅实现/contract/smoke真实错误可修复；修复受影响检查一次。正式finite guard遇到nonfinite loss/grad/Gaussian时，各rank先同步失败标志，停止更新并保存本rank分项、tensor统计、window identity及最后有效恢复点，不用nan_to_num掩盖。禁止自动调参重训；基础设施中断只允许从同配方完整恢复点续跑，且记录中断及复原曝光位置。

## 11. 评估与报告

epoch0/2/4/8/16/32/64对全部固定splits完成local评估；epoch0/8/16/32/64完成official all/novel。不临时挑最好窗口。rank0 eval/no-grad，其他rank等待；设置collective timeout=4小时以覆盖注册eval，评估期间不更新参数。BN无同步collective。每次rank0 eval后恢复原train状态并同步继续训练。

固定报告：

- local semantic all/thing/stuff mIoU；panoptic mIoU/PQ；
- independent candidate mAP/AP50、CA/CW TP/FP/FN、precision/recall；
- official packed-panoptic mIoU/PQ/mAP/AP50，context=`all/context`、target-all=`all/target`、真正novel=`novel/target`；
- context、target-all、true novel PSNR；
- raw best-mask IoU、eligible candidate、Hungarian matched query字段分别保存；classification confusion、matched分类准确率、no-object、每类召回；
- 固定qualitative：至少train probe、same-scene holdout、dev8、val32，展示GT、候选mask、最终panoptic和RGB重建；
- 逐层注入norm、`norm(delta_token)/norm(token)`、注入参数grad norm、route分布、feature lifting support、fallback比例，仅记录；
- 预训练参数漂移、各组grad norm、吞吐、显存、每window实际exposure。

完整复用 `b00b94f` evalfix后的资格/面积/score预测域规则与GT-valid域IoU规则；不改official evaluator、导出格式、阈值或GT instance协议。不把candidate AP替代official packed-panoptic AP。

### 11.1 固定比较与结论

本次endpoint与Fresh-128 epoch64比较。旧endpoint暂未完成时，不等待它才启动；本任务跑完先交付自己的结果，对比列标PENDING，旧endpoint完成后只补表，不重训。本次与旧实验batch/LR组/预训练/耦合均不同，必须在比较表首行说明。

分别回答：训练池完整实例任务是否学会、同场景新窗口是否有效、未见过场景是否提高、重建是否保持。

将“本配方出现有用跨场景进展”的工程判据预注册为：val32 true-novel official packed AP50相对Fresh-128 epoch64增加至少0.01绝对值，context AP50不降低；same-scene holdout8 official context AP50下降不超过0.01；val32 context/true-novel PSNR相对同一重建pretrained起点各下降不超过0.5dB。**这些是endpoint结论标签，不是中途停止或启动下一实验的gate，不是统计显著性证明。**

此外用val32 scene为重采样单位，paired bootstrap2000次、seed42，95%百分位CI，重采样时重新汇总该scene全部窗口的official计数/AP所需预测。AP必须从重采样预测/GT重新计算，不能平均scene AP替代全局AP；同一resample同时用于两模型。只用既有official代码，不改代码。若缺Fresh-128逐scene原始预测，CI列标UNAVAILABLE并解释，不追加checkpoint重评作本轮前置。

若CI跨0，写“点估计提升、证据仍不充分”；若符合工程判据且AP50差值CI下界>0，写“本联合配方在该固定验证池有提升证据，不能归因到某一个模块，也不等于完整SIU3R条件已对齐”。若失败，报告哪项没提高，停止本轮，不自动尝试新的结构。

当前输入使用GT camera poses，必须写posed setting；不得宣称与unposed SIU3R完全等价。不得把训练成功写成跨场景泛化成功，也不承诺8卡获得8倍加速。

## 12. 交付

最终回复必须包含：

1. 基线、实现/训练commit SHA、push核验、修改文件及worktree状态；
2. 三份权重实际路径/SHA、450/292/187/326加载报告、新初始化参数清单；
3. 架构与各tensor shapes、4层调用顺序、梯度边界；
4. CPU、单卡和八卡smoke结果，每卡峰值显存、活动性是否接通；
5. job ID、节点8卡、每卡batch1/global8、8064updates/64512exposures、实际每窗口64次；
6. checkpoint/eval节点、所有split的mIoU/PQ/mAP/AP50/PSNR和未定义/缺失原因；
7. train/holdout/unseen三个层次的object mask与分类结论；
8. 与Fresh-128的完整比较、batch差异和CI适用范围；
9. 是否保持重建、是否产生有效实例、是否有未注册科学改动；
10. 小于28MiB结果ZIP及SHA256，包含本规格、provenance、manifest/plan、关键日志、逐GT表和真实图片，不含权重/checkpoint/数据。超限按图片分包，不删掉不好的case。

不得以“实现已完成，可以训练”结束任务；必要检查通过并推送后直接提交正式job并推进到固定endpoint。若遇明确阻塞，给出具体证据、当前更新量及未完成项，不假装任务完成。

## 13. 决策表

| 决策项 | 固定取值 | 依据及影响 | 取错的后果 |
|---|---|---|---|
| 代码起点 | 150a443ff0b1ac434dd0e7410164ebc77cb4c66d | 已收敛接入规格、可复用V3与evalfix | 从落后主目录实现，接口缺失 |
| 科学目标 | 同组GS的联合重建与理解 | 用户的token-based前馈任务 | 退化为独立2D分割外挂 |
| 重建init | step47500、450 tensors、指定SHA | 与Fresh-128同源 | 继承旧过拟合object或错误权重 |
| 理解init | MASt3R292 + COCO187/326 | 保留完整可用预训练链 | 只换encoder仍随机理解 |
| 类别 | 18thing+no-object；2stuff masks | ScanNet既有接口 | 分类、GT及eval错位 |
| 分辨率 | recon256；under512由同crop上采样 | 保持数据内容一致 | 额外改变数据/声称新增细节 |
| Encoder接口 | 24层列表、patch16、无新增camera token | 原生MASt3R特征与adapter索引 | K/V乱接或新增未训练token污染 |
| BN | running stats固定，affine训练 | 小batch与rank0 eval | stats漂移或同步挂起 |
| Object初始化 | 100pretrained q + 2stuff；c/s soft anchors | 语义query与scene geometry对接 | 随机seeds替代预训练或丢anchors |
| 注册层 | 6/8/10/12，原循环内 | 注入真实影响后续生成 | 仅改缓存不能影响重建 |
| Evidence / route | anchors轴1024 / channels轴103 | 读取与消息分配分离 | 转置R错误分配消息 |
| 几何bias | unclamped负平方；无hard crop | 避免远场地板伪均匀 | 错误局部约束 |
| c/s | 每层V3残差更新；.05..2ell | 动态支持、有界中心步长 | 固定或被ownership覆盖 |
| 写回 | only tokens；zero W_inject；beta peak.1 | 温和、真正的生成交互 | 假反馈或直接几何收缩 |
| lifting | 原feature renderer的转置；32通道块 | 遮挡感知与可反传 | 中心投影伪归属或巨型贡献矩阵OOM |
| 融合 | g=mass/(mass+1)；child fallback+.1 residual | 对可见/不可见GS均有预测路径 | 不可见GS误当void |
| 最终mask | pretrained mask MLP；raw dot scale1；sigmoid102 | 预训练幅度与集合预测 | 温度误改或slot强制竞争 |
| Loss | 原V3 final set losses不改 | 已在训练池证明任务可学 | 缝补新loss无法解释 |
| Trainability | 全部参数训练，GC重建.01 | 真正joint而非冻结下游 | 理解不能改变生成 |
| LR | recon1e-6/pretrain1e-5/new1e-4 | 保留重建、适配预训练、学习新模块 | 新模块学不动或pretrain毁坏 |
| 并行 | 13号8卡，显式平均全部梯度 | GC兼容的global8同步训练 | 各卡各训或DDP双backward错误 |
| 训练量 | 64epochs；64512曝光；8064更新 | 与Fresh-128曝光一致 | 误报8倍更新/少8倍数据 |
| Warm-up | LR/under200曝光；beta1000曝光 | V3正式理解warm-up在样本轴对齐 | 八卡warm-up不经意放大8倍 |
| Gate | 仅必要contracts/smoke、40更新活动性 | 及时正式训练 | endless audit或指标差就换配方 |
| 比较结论 | 配方比较；CI+固定工程标签 | batch和预训练均变 | 虚假架构因果/泛化结论 |
| 资产实际路径 | 执行者仅机械定位Fresh-128唯一manifest | 私有运行文件未随GitHub提供 | 临时抽样改变训练池 |
| 第三方state目标键 | 按实际module层级一一映射并记录 | namespace随封装机械变化 | silent partial load |
| Slurm partition | 读取13号当前GPU partition名称 | 集群配置是运行事实 | 排错节点或修改训练设计 |

以上最后三项只允许确定运行事实，不允许选择不同科学方案。其余不存在执行者可自由决定的设计项。

## 授权勘误（2026-10-03）

MASt3R model 1017 tensors，encoder严格加载292个，其余725个全部排除，包括 `dec_norm.weight [768]` / `dec_norm.bias [768]`。本勘误不构成科学配方变更。复用已核验SHA与Fresh-128资产，smoke全部丢弃，正式fresh训练8064updates。
