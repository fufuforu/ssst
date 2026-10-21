# SAM引导Query初始化：先核验候选覆盖，再检验可用性

状态：提出可执行诊断设计，已从完整C结果独立核对85个失败小物体清单；**未运行SAM，未修改模型或提交训练/推理作业**。检查过的现有环境没有`segment_anything`/`sam2`模块，在SIU3R权重、当前资产和常用缓存中未找到可用SAM权重；不将目录名含sam的其他项目视作SAM资产。

用户假设：SAM的实例先验能否补足目前Raw Mask失败的小物体；进一步比较SAM引导与原Query初始化的物体覆盖。

## 先澄清比较对象

当前`PretrainedUnderstanding.forward`由MASt3R图像encoder、ViT adapter四尺度、pixel decoder和Video Mask2Former query decoder输出联合两输入帧的`q_pre[B,100,256]`与`F_m[B,2,256,128,128]`。
L6/8/10/12的Query不仅读取Anchor，还依次读取图像特征、Query self-attention和FFN。85个案例是完整C**最终Raw Mask**的失败，不是已证明的初始化失败。

旧understanding-bottleneck文档的q_pre代理统计来自历史短训练模型，不能冒充当前完整C的初始化结果。当前q_pre若无同源缓存，应仅补采当前理解分支的初始化读出，不重复运行已完成的最终Gaussian诊断。

## 第一阶段：冻结、不训练的候选覆盖诊断

1. 使用相同128窗口的两张context RGB，GT仅用于预测后评价；不使用GT点/框、GT裁剪或target图像生成SAM候选。输入来自与C相同的256图像；SAM内部插值到其原生分辨率不等于加入原始高分辨率信息。
2. 三个独立参照：当前C的q_pre→原mask embedder×F_m的2D代理、当前C已保存的最终Raw Mask、SAM自动生成的class-agnostic候选。q_pre代理不是正式panoptic输出；SAM候选也不是已经过主模型交互的Query。
3. 首先做**每视图**相同候选预算的比较：原方法每视图100个、SAM最多100个；SAM仅用其预测质量/稳定性及预先固定的去重规则筛选。另报SAM全部候选的上界、候选数量和耗时，不能把任意多SAM候选当等预算收益。点网格/额外crop策略预先固定，额外crop收益单列，不能在这85个GT上调参选配置。
4. 报告IoU≥0.5/0.75的GT best-mask覆盖及一对一候选匹配覆盖。分全部492个GT、123个小物体、其中85个失败小物体；记录C-only、SAM-only、二者均有/均无以及候选并集覆盖。案例保持相同RGB/GT尺度。
5. **跨视图不能偷换口径。** 现有100个Query是共享两帧的身份；SAM单图候选没有这种身份。先报告视图级可分割性、85实例至少一可见帧/全部可见帧得到好候选的数量。GT事后选择两帧不同SAM候选只可称oracle上界，不能称已实现共享Query覆盖；实际两帧共享初始化需使用预测几何/外观做不依赖GT的合并，再按100个共享对象槽比较。不能将每视图100个SAM候选拼成200个共享Query后声称公平胜出。

清单：`/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1/attempts/attempt01/sam_proposal_probe_preparation_v1/small85_failure_cohort.json`。
GT选择条件：C、两帧总GT面积<1310.72像素（1%）、所有最终Raw Query的最好IoU<0.5；492总GT、123小物体、85失败已重核。清单只能用于评价分层，不能提供给SAM推理。

## 第二阶段：候选有效后，再做Query初始化结构对照

SAM mask token即使也是256维，也不保证处于当前Query语义/数值空间。零训练直接替换后表现变差，不能单独否定SAM实例先验。
更清晰的候选结构是：SAM mask在当前F_m上做可见区域池化，经可学习适配映射生成Query内容，必要时用预测点图池化几何位置初始化对象中心/尺度；SAM本身冻结。两帧候选合并只用预测信息。先限制只改变初始化，保持已有Anchor/图像交互、Query数量、loss、训练流及预算一致；小物体/重复候选/最终mAP/PQ分别评价。

这一步需要新的结构适配训练，不能把第一阶段的SAM候选召回直接当成最终模型收益。SAM不提供18类的类别判别，且可能产生物体部件、嵌套Mask与重复候选；它主要针对Raw覆盖假设，不自动解决no-object或类别错误。

判断逻辑：若SAM显著补足85个案例，支持“更强的实例候选先验值得引入”；若候选能补足但共享Query或训练后不能保留，问题转向候选编码、跨视图关联与后续交互。若SAM同样失败，只否定所测SAM/输入/自动提示策略的帮助，不能证明全部初始化方案无效。

后续若提交作业，沿用户要求：提交并确认真实启动后停止轮询，等用户通知完成再检查。
