# 三维超点初始化结构对照：旧双分支与共享Query直接Mask

用户提出并行两组结构实验，并明确第二组选择**三维Gaussian空间：Query驱动非线性Mask头，直接输出各Gaussian的归属概率，再渲染到二维**。先各做256步短验证，不自动提交全量，不干预66432/66433，不恢复B旧实验，不重跑66353。

## 共用前端和短验证设置

冻结官方VGGT-Omega与SAM ViT-B；SAM区域按ESAM MV的面积覆盖方式分区，VGGT预测深度/相机对应的点携带区域标签，未覆盖点XYZRGB KMeans8。点特征经三维邻域编码、超点池化后初始化Query，完全没有GT初始化、二维Mask直接池化Query或随机fallback。保留每个有效区域、同区域压缩；上限512超点/4096融合点，每重建Query64个Gaussian。

两组使用同一16个固定训练场景、同一4train＋4heldout初末探针；与66414选择相同，数据manifest/heldout SHA不变。18节点各4×4090、16CPU/128G，并行共8卡。每rank一场景、梯度累积2次，全局批8；256更新/2048曝光，128遍16窗口，非全量128epoch。各自seed42新初始化、空Adam、无已训练源权重。AdamW LR1e-4、20更新warmup/短cosine、clip1、理解总权重0.05/20步ramp。正常联合反向，不套用此前旧V4的完全梯度隔离。

共同采用来源多对一完整Mask监督：合格超点对其GT实例都为正样本，损失按GT内部再跨GT平均；context两视图同类Mask NMS、复用选中的实例身份到目标视图。不是官方128评测；此次不因loss没有下降而自动判失败，不自动续训完整数据。

## A：原双分支交互/读出＋两组Token三维超点初始化

- 架构：`OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1`，CLI `--architecture dual_superpoint`。
- 相同GFM投影后的融合点，分别经过独立的3D邻域编码及超点池化，得到同一超点身份/几何对应的两组内容；重建种子投影到1024维，理解种子256维。两条内容编码和后续Query分开，几何种子由相同超点决定。
- 保留原GFM重建Adapter、原SemanticMemory（多层GFM＋局部RGB残差）、原1024维12层重建Decoder及ray bias/PE/几何更新、原256维理解层L6/L8/L10/L12：读Anchor＋图像证据＋自注意力。
- 理解→重建Write为0。两任务仍通过正常联合梯度影响可训练模块；Write=0不是梯度隔离。
- 保留旧读出：可见性相关的renderer-transpose提升二维语义证据，融合Anchor/Child特征及残差，与Object Query的mask descriptor点积得到Gaussian membership，随后渲染。
- 原实现固定1024 Anchor/100 Object的检查改为本新变体的可变S版本；不填充learned槽、不截断到100、不把区域不足时补成旧1024。旧生产类默认行为不变，只在新调用提供动态scene normalization；新动态Object层在100 Query时与旧层数值回归一致。
- 约193,799,281个可训练参数：shared528,640、reconstruction183,666,753、understanding9,603,888；693张量。
- run：`attempt03_legacy_dual_superpoint_short_node18_4gpu`。

A保留的是旧双分支的核心交互和读出，不是旧V4的严格纯初始化单变量复刻：Query/Gaussian数量、来源分配和推理去重也按当前超点方案改变；新训练采用当前0.05联合策略，而非旧隔离策略。须明确这些差异。

## B：共享Query＋直接三维实例Mask＋Gaussian语义

- 架构：`OMEGA_3D_SUPERPOINT_UNIFIED_DIRECT_MASK_V1`，CLI `--architecture unified_direct`。
- 保留当前256维12层共享Query及Gaussian重建头；删去原Child feature/Query mask descriptor相似度读出模块。
- Query生成一个小型非线性Mask函数的参数。函数在每个Gaussian上输入相对Query中心的XYZ/scene-scale、sin/cos位置编码、log-scale、opacity、RGB，执行两层MLP得到归属logit。输入维16、隐藏16；对全部Gaussian求值，不局限于本Query的Child，不使用固定Gaussian槽号或两塔特征相似度。
- 分块/非重入checkpoint控制内存。Gaussian或Query重排时对应输出保持重排等变；MLP内部普通线性层并不等于旧的Gaussian语义特征与Object Query embedding匹配。
- 另由每个Query直接读出64个Child的20类语义logits，Gaussian语义概率经渲染得到二维语义Mask。stuff的两个Gaussian membership直接取语义概率中的0/1类。
- 原thing分类/no-object与多对一完整Mask BCE/Dice保留，新增`0.1 × semantic CE`到理解loss，之后整体乘同一0.05及warmup；监督通过渲染像素完成，不直接给遮挡Gaussian贴二维GT标签。该额外监督是B的组成变化，不宣称与A目标完全相同。
- Panoptic不是另一个独立预测头：thing实例Mask结合对应Query类别、stuff语义概率，经原像素竞争/过滤及NMS组装成一致的实例ID和类别。单独的raw semantic prediction可能与融合后的panoptic语义不同，初末探针分别保存二者；没有聚类或实例embedding。
- 约14,683,877个可训练参数：shared13,912,881、reconstruction296,576、understanding474,420；296张量。
- run：`attempt04_unified_direct_masks_short_node18_4gpu`。

## 验证、比较和执行边界

28项CPU测试通过，包括A两个3D初始化器梯度/共用几何、可变Query包含超过100的分类、与旧100-query层数值一致、A完整context生成和梯度；B非线性Mask的Gaussian/Query重排等变与分块一致、几何/Query梯度、直接语义CE和Gaussian语义读出；包含既有初始化/Mask/NMS/诊断回归。

两项真实2rank CPU Gloo测试通过：A旧特征提升自定义反向和B直接Mask checkpoint，均使用不同Query数量、连续2次Adam更新后全可训练参数SHA跨rank相同。预检实际冻结文件SHA、固定数据及分组完整性通过，native新结构GPU前后向仍待作业执行，不把CPU检查或RUNNING称为GPU通过。

两组在同一硬件/数据/全局批/更新预算下做方案比较，但参数量显著不同；B还新增直接语义监督。不能把结果差异只归因于共享/分离Query或初始化。初末探针包含PSNR、理解各项、来源支持GT/正Query数量、最佳Mask IoU、raw语义像素准确率、NMS和最终实例，以及紧凑预测张量。首更新检查4rank全参数SHA/分组梯度及冻结模块无梯度；结束各自自动汇总。

共同run根：`/space/mawb/ssst/group_plus/unified_query_training_v1`，配对收据位于`structural_pair_20261017`。独立快照`/space/mawb/ssst_superpoint_structural_pair_code_20261017`，现有全量快照不修改。具体job/SHA以pair/submission.json为准；提交后只做一次状态核对，然后停止轮询，等用户通知完成后检查。

## 已提交并停止轮询

2026-10-17北京时间15:02:22：A **66487**、B **66488**，单次核对均RUNNING，18节点各4×4090、16CPU/128G。执行SHA`ead7f58`，独立快照干净；GPU首步、原生renderer-transpose/直接场前后向及四rank SHA尚未核实，不能将CPU/DDP检查或RUNNING称作GPU通过。pair/submission.json和SUBMISSION_STATUS_ONCE.json已记录。无额外全量/正式评测依赖，短测初末探针及汇总在各自同一作业内完成。已停止轮询，用户通知结束后检查，不重提、不影响66432/66433。

## 2026-10-17 用户通知后的失败审计与修复

66487/66488实际均FAILED（1:0），分别运行1m30s/1m31s。共同异常位于`unified_query_validation.run_probes`新增的raw语义准确率统计：前面直接将`build_batch(...)`传给step_loss，后面却引用没有绑定的`batch`，触发NameError。四rank均在initial探针中退出，未进入训练更新；无initial checkpoint/训练日志/首更新收据。因此不能解读为架构失败或训练效果差，也没有A/B质量结果。

修复为显式保存同一batch再传给step_loss，统计仍只取context前两视图，忽略20类ignore。新增完整run_probes回归覆盖8个初始探针、语义统计、输出保存、GT IoU、模型模式与Python/Torch RNG恢复；CPU允许空CUDA RNG设备列表。没有修改模型、数据、训练策略及评测协议。

原失败目录及快照全部保留。重试使用原目录名加`_retry01`，配对目录`structural_pair_20261017_retry01`，不可变代码目录`/space/mawb/ssst_superpoint_structural_pair_code_20261017_retry01`；修复后从零开始只补这两组未产生更新的短测。原提交脚本保留默认防重复行为，显式`--retry`选择上述独立目录。

同时有限核对全量66432：实际仍在运行，检查时2138/4172步（约51%），rank0日志所有浮点标量有限，最近100步平均3.689秒/步，训练估计剩约2.1小时（含少量探针/保存开销，非评测耗时）。66433依赖训练成功，尚未产生正式128评测。原代码快照和训练均不触动。

固定4 heldout探针初始→epoch1→epoch2：PSNR 11.625→16.755→17.367；理解loss 2.078→0.705→0.603；来源支持GT的最佳Mask IoU 0.1649→0.5469→0.6238。4 train对应PSNR10.935→17.231→17.694，IoU0.1941→0.5283→0.5891。IoU只对已有来源支持/匹配的GT汇总，并非全部GT覆盖率、mAP或PQ；这些探针不代表官方128场景质量。
