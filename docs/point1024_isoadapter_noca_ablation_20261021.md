# P1024iso No-CA：只移除学习式 Object→Anchor Cross-Attention

用户要求在70026版本上增加一个完整训练对照，任一12/13/14/17/18节点单机8卡。起点为21a85947388b41ccc22f09b52ff43adb0caf2573，原快照和作业不修改。

CLI：`dual_point1024_isoadapter_noca`；架构身份 `OMEGA_POINT1024_FMQUERY_ISOLATED_NO_ANCHOR_CA_GEOMETRIC_ROUTING_V1`。新模型/核：tokengs/models/point1024_isoadapter_noca.py。

## 唯一模块消融与几何定义

每个L6/8/10/12 Object layer：完全删除anchor_attention的Q/K/V/O四个Linear；不计算学习式Object→Anchor CA，不将anchor内容注入Query。保留原norm0：q1=LN(q)，随后FM image attention、self-attention、FFN和norm1..3保持原样。

显式定义 `R_geo[b,q,t]=softmax_t(-0.5*sum_d(((mu[b,t,d]-c[b,q,d])/s[b,q,d])**2))`。它只用几何距离和既有尺度，不用Q/K投影或anchor特征；不是把R留空，也不是继续使用随机/冻结的学习式R。两个stuff行保持原geometry_bias=0，对anchor均匀，仅用于日志兼容，不更新stuff几何。evidence_attention存为[B,1,Nthing+2,Nanchor]；原update_geometry的R.mean(1)现在就等于R_geo。

原DynamicObjectLayer.update_geometry不改：R_geo加权中心/方差、c的0.25插值和0.25ell步长上限、s的log尺度更新/0.05ell..2ell约束，以及W_c/W_s(qnew)残差均保留。默认DynamicObjectLayer仅提取_read_anchor方法，非NoCA路径仍调用原anchor_attention，数值计算不变。

## 保留的路径与消融边界

重建1024×1024 token、65536 GS、seed_anchors精确来源256特征、L6/L12重建监督、FM256超点Query初始化、256维理解decoder、GS child/anchor mask特征和4个隔离残差adapter都保留。NoCA仍通过child/GS mask特征读取重建信息，因此不能称为“理解完全不读取重建特征”；消融对象只是学习式Object→Anchor cross-attention。

现有Write=0，route/joint_delta不写重建，最终mask也不直接使用c/s。因此NoCA中的c/s仍数值演化，但不再通过CA影响后续Query/最终loss；W_c/W_s可能成为无梯度参数，不额外添加几何loss或mask gate来改变配方。L6/L8/L10 adapter失去原CA监督路径，预期无LU梯度；L12 adapter仍通过最终child/mask分支受LU训练。DDP保留find_unused_parameters=True，首步只要求L12 adapter非零；不能将早层adapter零梯度误报为失败。

## 初始化与预算

保持原真零步：SHA3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744，0updates、空Adam、原8rank RNG逐rank恢复；不接70026训练后权重。构造全部基础模块后才删除CA，保留其余模块初始化RNG顺序；再严格迁移625个保留张量，逐值与原A4零步一致。相对P1024iso新增删除32个CA张量/1,052,672参数；连同原36个重建初始化删除项共68个；16个adapter张量仍新增528640参数。总可训练参数192,544,624。迁移函数只接受明确枚举的68项删除，其他缺键仍报错。

训练完全沿70026：8337窗/1191场景；2context+4novel、256²；global8=8rank×1；4epoch/4172updates。LR1e-4、warmup100、cosine1e-5；LU .05×min(1,(step+1)/1043)。L6/L12 [1/3,2/3]；AdamW betas(.9,.95)、wd.05（1D/显式例外0）、clip1、Write0、同matching/NMS/损失不变；I/O缓存/有限重试沿21a。源zero硬链接；训练清单SHA8fe72589a2b8f1b52fa517cde39a0fce37d104adc2ee3a28585d1747292313f1。

## 已完成验证与执行边界

3项CPU测试通过：几何R simplex/特征不变性/几何响应，真实12层core与实际LR/LU逐参数双向隔离（1024真实种子检查，完整CPU渲染阶段仅降为6anchors），严格零步迁移拒绝规则。实际740MB源零步额外审计通过，625张量逐值一致，68删除/16新增，参数计数准确。另7项2+4数据与收据测试通过；Python编译、shell语法通过。CPU使用FakeOmega/SAM/renderer以及审计身份stub，不等于实际GPU前后向或8卡同步；未宣称NoCA GPU smoke通过。

独立源码：`/space/mawb/ssst_point1024_isoadapter_noca_code_20261021_v1`。
独立run：`/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_isoadapter_noca_full4_20261021_v1_8gpu`。

提交器使用3090,4090分区、generic gpu:8、nodes=1；枚举分区节点后排除候选池之外节点，从而让12/13/14/17/18任一节点满足8卡即可启动。不能用含5个节点的--nodelist（那会要求全部节点）。32CPU/192GB、24h时限；sbatch --test-only验证资源；提交后仅一次有界squeue。保留原epoch probes/每50步latest/每epoch checkpoint。仅提交这一个训练，不自动加评测，不触碰70026/70033/70042。实际job/状态见run/SUBMITTED.json与STATUS_AFTER_SUBMIT.txt；GPU首步/8rank hash待用户通知后检查。
