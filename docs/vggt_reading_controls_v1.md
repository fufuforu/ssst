# C起点四组decoder读取对照

用户授权执行此前提出的基线／局部读取／局部视觉证据／可靠深度约束四组，不将Q作为干预变量。
固定当前最新C：完整adapter理解梯度0.01、feedback0.05权重ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0，
分类对象支撑读回补丁ad3100ceabbd4d9ab41f653cdb40ec1cb86dc882d71d9ad5ac14192947f4f0f9。无AA解冻、encoder增强、额外loss或GT生成输入。

## 锁定的共同预算

每组300正式更新／2400新增窗口曝光，来自已有完整8337训练池固定2400不同窗口流；非128场景重复拟合。
8×3090，microbatch1、accumulation1、global batch8。11/13节点相同型号，分两波，避免资源争用。
固定已有128个留出场景各1窗口，与训练场景无交集；不是完整312验证或4epoch长期训练。
采用原native重建＋V3完整set losses、原matcher、辅助层监督和梯度分组：GS理解梯度0.01、VGGT memory adapter理解梯度0.01。
feedback上限0.05、understanding weight1；源联合曝光35776，新曝光单独2400，不重启这两个成熟任务的warmup。
fresh native AdamW原分组／decay／beta.9/.95／eps1e-8／clip1；LR原峰值1e-5/1e-5/1e-4、200更新warmup、4172余弦截断300。
VGGT整个模型和MASt3R image encoder冻结/eval；其余按原requires_grad规则训练，C分类读回模块可训练，所有BatchNorm仍eval。
四组可训练参数集合、数目和起点完全相同，新增训练参数0。C读回原有detach保持，未偷偷联合重写分类梯度路径。

## 三个独立干预

baseline：原计算，新增观测hooks不改变输出。

object_local_read：L6保持原读取；L8/10/12用上一交互层预测anchor soft masks（detached，无最终mask／GT预知）引导object-query→anchor CA。
mask逐query按最大值归一化，额外bias=λ log(0.1+0.9 mask/max)，λ峰值0.5。
空／常量支撑及两stuff query不加bias，全局读取始终保留至少0.1软底。anchor内容、所有Q/K/V/O原投影和feedback方程不变。
这里只改读取权重的计算，可能经query及feedback间接影响后续anchor，不能声称整个模型最终anchor永远相同。

anchor_visual：每个交互层将原μ投影到两个预测context相机，在现有Fm上bilinear采样；仅正Z、图像范围内的视图参与平均，无有效视图为0。
补充证据LayerNorm／unit norm；residual=λ×0.25×detach(norm(a))×normalized_evidence，λ峰值0.1。
因此峰值附加证据norm不超过anchor embedding norm的2.5%。不加encoder、MLP或额外参数。
保留原CA读取方程；改变K/V内容后实际attention权重也可能随之改变，不宣称实现冻结R。
该几何采样本身有梯度，但相机/K来自context冻结教师且detach；不会把GT相机或targetRGB用于读取。

reliable_depth：作用于GS anchor→VGGT patch CA，而非object-query→anchor CA。
保留原点到无限Pluecker射线的垂直距离bias，增加沿射线深度偏差项。
已有归一化场景predicted_points和confidence按14×14平均池化，与原patch ray的view/row/column序完全一致；不加载point_head。
confidence每视图q20/q80映射并clamp到[0,1]，只是可靠性权重，不声明为概率；退化同conf映射0。
附加bias=λ×reliability×clamp(-0.5[((μ-point)·unit_ray)/max(radius,原bandwidth_floor)]²,-20,0)，λ峰值0.1。
低confidence附加约束弱，不削弱原射线约束；无新标定算法、GT深度、相机优化或loss。

三干预起点λ0，前20个新更新按min(completed_updates/20,1)开启，之后固定。
这些强度预先选为保守短对照，不在留出集拟合；若无增益，结论仅适用于本实现／强度，不足以否定整个改进方向。

## 验证及执行

CPU合同验证局部支撑全局回退／无GT梯度、两视图投影对应和边界、沿射线深度可靠性／平移不变／梯度、真实V/O attention零强度等价。
先单卡及八卡各组真实两步joint backward，验证共同C输出复现、loss/梯度/参数finite、encoder/BN不变、下游更新及rank同步。
smoke第1步零干预、第2步完整强度，验证新增路径实际参与backward；smoke optimizer/RNG不带入正式训练。
每组单卡／八卡各导出真实完整强度留出窗口，并用固定未修改SIU3R官方evaluator评分。
记录真实CA V每head norm、完整O输出norm、逐head attention entropy，避免把mean(R)@原a代理误作完整CA输出。

四组先各跑正式20更新／160曝光并保存optimizer/RNG/采样的滚动checkpoint。
CPU统一核验四组共同起点、参数容量、八rank更新同步和finite后，自动从各自20更新checkpoint精确续跑至300；不是重新训练20步或fresh optimizer。
这样可在四组都真实启动后停止人工轮询，不必等第一波完整300步结束才能验证第二波。
11号baseline与anchor_visual，13号object_local_read与reliable_depth；各节点顺序续跑，最多两个八卡训练同时占卡。
checkpoint只保存实际可训练下游参数、非encoder buffers、原AdamW/RNG/计数，冻结源权重按严格身份复用。
每组仅保留一个atomic滚动latest（20/100/300），约2.6GiB；日志／收据／失败历史保留，避免四组各复制6GB源checkpoint。
四份＋一个atomic临时文件约13.03GiB，启动前磁盘至少12GiB且正式导出前核验余量；磁盘问题不通过改模型或精度处理。

## 评测与Q

训练300完成后自动八卡导出／官方评分，每组同128窗口，GT/input/frame IDs跨组一致核验。
RGB/depth由原未修改官方Evaluator重新评分，mAP/PQ/mIoU通过官方状态全局合并，不平均rank AP；真实CPU/GPU合并合同逐组核验。
只输出官方13列，两个mIoUt空白。附带相同GT集合的分类/no-object/好mask诊断。
本轮不带旧Q评分，以免输入分布变化误判结构；最终若要比较带Q的AP，应各组等预算重新拟合Q，作为后续单独授权任务。
本轮评测不证明4epoch完整训练增益；用短对照判方向，再决定长期训练。

证据root：`/space/mawb/ssst/group_plus/vggt_reading_controls_v1`。
严格固定快照，提交推送后计算节点只运行snapshot；monitor_v1保持质量告警，历史失败不覆盖。
