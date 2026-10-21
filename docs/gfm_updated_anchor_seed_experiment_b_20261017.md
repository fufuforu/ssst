# 并行结构实验B：更新后的Anchor初始化Object Query

> **用户已于2026-10-17取消本实验。** 66380/66381/66382单次核实均CANCELLED。原因是初始化目标澄清：需要三维点特征经超点池化形成Query，当前SAM支持的Anchor插值方案不符合；不是基于完整训练结果判定性能失败。日志/执行快照保留，最新完整checkpoint为200步，另硬链接保留checkpoint_stopped_user_cancelled_b.pt；取消收据见run/USER_CANCELLED_20261017.json。66353不动；不得自动恢复、重提B或将本记录当作新训练授权。下文提交/等待状态为历史。

用户于2026-10-17授权开始实验B；66353残差组保持原快照和进度。本实验是新attempt17，在18节点八张4090训练，之后自动官方128评测与比较；提交后仅单次核对状态，停止轮询。

## 目的与比较边界

66353的query残差来自SAM池化图像语义特征；B的残差来自**完成重建后的Anchor token**，SAM只提供空间支持（候选关联仍使用原语义特征）。两组均保留learned base、相同0.1初始门控、同一匹配/损失/数据/4epoch/梯度隔离。相对66353，B改变内容来源和前向顺序；这是一个完整结构方案对照，不能把结果全部归因于某一个算子。4090与对照3090不同、单种子，也限制严格因果解释。

## 实际前向

1. 冻结Omega产生GFM、预测相机、点云；原两Adapter、点云FPS的1024 Anchor初始化保持不变。
2. 重建decoder的12个block各执行一次，保存每层tokens/mu/rho/radii；本阶段不调用Object层、Write严格为0。保留原始重建计算图，Gaussian仍由原activation head在原重建终点上生成。返回给原多层重建loss的states也保留每层原始tokens/mu/rho/radii，不能只保留最终Gaussian的梯度而切断中间层监督；Object结果来自detach的重放，二者在返回状态字典中组装。重建参数继续由完整原重建loss优化。
3. 用现有`panoptic.encode_token`将最终Anchor token投影成256维证据；输入的tokens、mu、radii全部detach。复用L6的scene scale，与原L12 Anchor读出一致。投影自身属于理解参数，可以接收理解梯度；无需新增1024→256投影。
4. 原V4 SAM筛选/同视图去重/跨视图合并/80槽截断保留。每候选保留至多128个预测3D支持点；合并组汇集各视图支持。跨视图关联仍依据原几何重叠与图像语义池化相似度，不用GT或target视图。
5. 对每个支持点，在**更新后的Anchor位置**中找最多4个邻居，仅保留距离≤0.1×初始scene scale的邻居。邻居权重为`softmax(-0.5*(distance/(0.05*scene_scale))^2)`，插值得到该点的256维Anchor特征。没有近邻的点不参与池化；整个区域都无支持时，使用零残差，精确回到learned base，并记录，不借用远处Anchor。
6. 用原V4的局部形状MLP/点权重/形状投影池化这些Anchor特征，随后用原query content projection和几何编码生成seed。原几何池化参数仍有实际梯度；并非仅把模块留在注册表中但不使用。
7. 沿用66353的`LN(learned_base + sigmoid(gate)*RMS(stopgrad(base))*LN_without_affine(seed))`，gate初始0.1。fallback槽不变；c/s仍由原SAM区域点云规则产生，不在本轮同时改变局部几何读取偏置。
8. 理解decoder在保存且detach的L6/L8/L10/L12 Anchor状态上依次执行原四个Object层。不会再次运行重建block，也不将后来的query反馈到过去重建层。语义Adapter图像K/V、最终Child membership与Mask/类别头不变。

3D插值是由SAM区域预测点云构造的软汇聚，不是GT ownership；跨物体边界可能混合，且支持不足时残差不起作用。这些是本次实验要检查的风险，不能事先称为已解决实例对应关系。

## 训练与起点

- 运行：`/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt17_updated_anchor_sam_seed_isolated_node18_8gpu`。
- 独立执行快照：`/space/mawb/ssst_gfm_anchor_seed_code_20261017`；旧执行快照不改。
- 起点：共同零步checkpoint，SHA `b9d7bd597cc2bd0d2439a1eae3771760f2be55241a2db1c4ef3879b08a75a03e`，649个共享张量逐位复制，空Adam，原8rank RNG；不继承任何已训练模型、不重复适配。
- 相对原V4只新增gate标量1参数；相对66353参数键/形状完全相同。重建186881600、理解9128561参数。
- 完整1191场景/8337窗口，全局批8，4epoch4172更新，adapt/short0；LR重建/几何/理解=1e-4/1e-5/1e-4，100更新LR warmup+cosine，理解loss首1043更新ramp。
- 完全梯度隔离，重建仅重建loss、理解仅完整理解loss；原FP32 SUM/world reducer和全局clip1保留。Write0，不冻结重建，不加入新的ownership/matching损失。
- 新group必须通过原首GPU更新几何池化梯度检查及八rank全参数SHA检查；新增门控和Anchor投影梯度/参数变化证据。记录区域数、无支持区域数、支持点覆盖率及门控值。

## 验证与评测

提交前检查包含：CPU实体张量源预检，真实1024 Anchor接口下的两阶段编排、重建前向逐值一致、理解重放不重复重建、理解梯度不进入重建而重建梯度仍有效、Anchor内容/投影/池化/base/gate梯度、无支持区域回退及配置锁定；旧残差/初始化/几何池化回归检查。

CPU检查不替代GPU完整训练证据。训练成功后依赖官方128场景导出/评测，保持11项及文本两项空白；导出根据checkpoint的独立mode构造B模块，严格加载权重。比较新B、66353残差组、65831原隔离组及工具默认旧C。比较汇总等待本组评测与66354均结束，防止把尚未评测的66353误报为坏结果。

判断同时看PSNR与mAP/PQ，后续再结合SAM正匹配/noobject/最终贡献及无支持比例。只观察loss或门控变化不能证明结构有效；结果不佳时不自动叠加encoder、匹配或交互层修改。

作业号以新run的submission.json为准；提交后单次核对并停止轮询，等待用户通知。

验证记录：24项CPU单测通过（包含1024 Anchor真实接口、两阶段顺序、值一致与多层重建梯度保留）；真实零步源预检PASSED，649共享张量在CPU实体上逐位一致，空Adam/8rank RNG与数据SHA锁定，新增gate标量1，参数规模同66353。py_compile、bash -n、diff --check通过。GPU首步/同步尚未核实，不能以CPU结果替代。

## 已提交，停止轮询

训练66380，请求18节点8×4090，单次调度核对PENDING(Priority)，尚未分配资源；官方128评测66381 afterok:66380；比较66382 afterany:66381:66354，均Dependency。执行快照SHA678a47c6ae9b0829e95c07e49880527dc4496520，提交前快照干净。没有声明GPU前向/反向或八rank同步通过。66353及快照不动。按同类八卡运行估计开始后训练4–6小时、评测10–20分钟，排队另计；不继续轮询，等待用户通知。
