# 固定C的表征／读取／读出瓶颈诊断

用户要求在固定Q扩评运行期间执行瓶颈诊断，寻找改进方向。使用11号空闲8×3090独立提取特征，保持13号Q扩评64392/64393不变。
固定adapter理解梯度系数0.01、feedback0.05的完整源权重及C对象支撑分类读回补丁；原模型、几何、mask、双向结构、loss及monitor_v1全部不变。
源SHA ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0，C补丁SHA ad3100ceabbd4d9ab41f653cdb40ec1cb86dc882d71d9ad5ac14192947f4f0f9。

## 采样及特征

采用旧2400固定训练流的第一1024个不同窗口（完整8337训练池），以及既有128留出场景各一窗口。
训练与验证场景不重叠。原C始终eval/frozen，原参数与缓冲区版本逐窗口核验。
每个训练窗口的frame IDs、RGB哈希、分类LN特征、原分类logits及原生matcher的前景query集合，与旧Q真实八卡缓存逐项复现。
不重新训练C/Q或重跑正式训练；本轮只有固定模型前向与解析探针求解，优化器更新0。

比较19个输入位置，统一100个原query身份、256维探针输入：

- q_pre：MASt3R/Mask2Former产生、进入双向层前的query。
- q6/q8/q10/q12：四个交互层后的query，q12还没有C的分类专属读回。
- q_class：C分类LN后的实际输入。
- anchor_read12：最后一层真实平均anchor attention读取的256维anchor embedding。
- anchor_local：同一预测对象软Gaussian mask及可见mass加权的anchor embedding，64个child按原父anchor顺序聚合。
- object_local：C原mask_support_readback的对象可见Gaussian特征池化，使用其原软mask和可见性权重。
- visual_local：同一预测context mask乘alpha池化Fm，不使用GT区域。
- gfm_memory_local：原混合1024维memory；gfm4/11/17/23_local：四层原2048维GFM；dino_local：VGGT同一次原前向中已有的1024维DINO patch tokens。
- scene_local：全场景可见Gaussian平均，重复到100query，检查对象局部性。
- shuffled_qclass：逐窗口固定随机打乱query特征与目标对应，破坏query身份；不是同步排列K/V。
- null：零特征，检查仅类别／质量先验的效果。

1024/2048维输入先用同一预测mask池化，再逐向量LayerNorm、固定seed92831+输入宽度的随机投影到256维。
所有19种输入再统一LayerNorm并添加不受正则的常数bias。投影不学习，各probe学习系数数量相同。
固定投影可能丢信息，因此“探针差”不能证明原高维encoder没有信息。
局部特征使用最终预测支撑，具有定位和时间顺序优势；优于早期query只能定位候选改进，不能单独证明CA因果责任。

## 相同目标与闭式探针

每个窗口只运行一次冻结C的原生Hungarian matcher。所有位置使用同一query标签：匹配GT类别0..17、未匹配18。
分类探针为19输出的加权岭回归，no-object权重0.1，每窗口归一化总权重为1；不是修改原模型CE。
质量探针为1输出岭回归，目标是原context-generation硬mask对GT的最大类别无关IoU（同一valid像素域），仅非空mask参与，每窗口等权。
GT只用于目标及离线统计；预测Gaussian/像素支撑、encoder特征、attention均仅来自context生成，未读取目标图像。
这与旧Q监督“最终保留mask、同类别IoU”不同：本轮分开诊断类别信息和形状质量信息，不作为新的官方评分头。

固定岭系数0.001，不在验证集调参。FP32矩阵产品、FP64充分统计累计／求解；feature系数正则为0.001×窗口数，bias不正则。
在256/512/1024训练窗口保存解析系数，评估同一128留出场景，观察样本预算及训练／留出差距，不因短SGD未收敛误判。
没有优化器或原模型参数更新。主指标为匹配query19类准确率、条件18类准确率／macro、好mask误判no-object数、mask质量MAE/MSE及窗口内Spearman。
同时报告原C分类输出作为参照；所有结果是探针诊断，不是官方13列或新模型上限。

## 注意力与confidence

记录各交互层的anchor attention归一化熵、top10 mass、读取自身最终预测Gaussian支撑的质量及uniform参照，按好／差匹配mask与未匹配query分层。
这些是观察性统计，不能把attention分散直接判为错误或用依赖性证明结构改动会增益。

128验证context各帧，将VGGT confidence按有效GT深度像素排序为五个等数量分组；每帧全有效像素统一拟合一次depth尺度／偏移，分组不再分别拟合。
报告各组AbsRel、confidence均值及高−低误差；GT不进入特征。需区分confidence关联与因果门控、VGGT depth与GS rendered depth。
本轮没有confidence attention改动，也没有新的融合／DINO encoder训练。

## 执行与产物

GPU共享提取1024训练＋128验证窗口，GPU0按固定8窗口累计充分统计，各rank保存CPU缓存。
八卡160训练窗口缓存核验后保存STARTUP_VERIFIED.json；历史旧Q缓存不改。CPU分析依赖GPU完成，输出analysis/results.json、report.md及256/512/1024系数。
预计特征缓存约2.3GB，RGB/depth图像不重复落盘。证据root：`/space/mawb/ssst/group_plus/vggt_feature_reading_v1`。
CPU合同包括流式／整批解等价及梯度停止、预测mask池化保持两视图对应／空支撑为零、null先验与空质量监督；真实GPU缓存核验原C与旧训练缓存一致。

解读顺序：先确认探针优于null／shuffle且样本预算稳定；再看局部特征与query阶段差异；再结合类别、mask质量及注意力定位。
probe高而原输出低可支持头／监督方向；局部支撑高而query低可支持读取／对齐方向；DINO／某层GFM更高可支持特定融合候选。
所有候选最终需要同预算受控训练验证；本诊断不能单独解释剩余PSNR差距。

## 运行记录

CPU三项数值／对应合同通过，源及清单核验通过：1024窗口来自707训练场景、128留出场景，均无交集。
GPU缓存64394（11号八卡）成功完成1024训练及128验证窗口，原C参数／缓冲区冻结、全部1024窗口分类特征／logits／matcher与旧缓存复现通过。
执行快照361b2a0980efb36344aed24a72d10a6b5f02c264。原CPU分析64395因JSON将feature_views tuple编码为list而触发身份比较断言，未产生分析结果。
失败日志及ENGINEERING_FAILURE_CPU64395.json保留。修复仅统一身份表示，分别记录GPU缓存执行SHA与CPU分析执行SHA；纯探针数学文件SHA与原快照严格相同。
复用全部缓存及三个预算的系数，只重提CPU分析，不重复GPU前向，不训练模型。
