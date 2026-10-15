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
每组仅保留一个atomic滚动latest（20/100/300），普通格式约2.6GiB；日志／收据／失败历史保留，避免四组各复制6GB源checkpoint。
续跑快照以node-local /tmp暂存普通torch archive，再无损ZIP_DEFLATED level1写共享临时文件并atomic替换；完整参数／optimizer／RNG／计数不删、不改精度。
压缩文件不使用mmap；参数、AdamW状态、RNG及采样计数roundtrip逐项相等，官方评测环境的torch reader也验证通过。
锁保护两个节点的共享checkpoint写入，仅一个共享临时文件；工程存储修复不改变训练循环、初始化、loss或优化器。

## 评测与Q

训练300完成后自动八卡导出／官方评分，每组同128窗口，GT/input/frame IDs跨组一致核验。
RGB/depth由原未修改官方Evaluator重新评分，mAP/PQ/mIoU通过官方状态全局合并，不平均rank AP；真实CPU/GPU合并合同逐组核验。
只输出官方13列，两个mIoUt空白。附带相同GT集合的分类/no-object/好mask诊断。
本轮不带旧Q评分，以免输入分布变化误判结构；最终若要比较带Q的AP，应各组等预算重新拟合Q，作为后续单独授权任务。
本轮评测不证明4epoch完整训练增益；用短对照判方向，再决定长期训练。

证据root：`/space/mawb/ssst/group_plus/vggt_reading_controls_v1`。
严格固定快照，提交推送后计算节点只运行snapshot；monitor_v1保持质量告警，历史失败不覆盖。

## 2026-10-14执行记录

实际训练流2400不同窗口覆盖1075训练场景，固定128留出场景交集0。
attempt01取消于正式更新0：静态复核发现观测hooks会访问原模型复现期间暂时移除的reading_control，修复后独立重提。
attempt02 GPU预检64574遇到模型module早于native runtime兼容bootstrap导入，缺fused_ssim_cuda；正式更新0，日志及ENGINEERING_FAILURE.json保留。
attempt03 GPU预检64616发现逐位读出断言过严：Gaussian、Fm、q_pre完全相同；logits差9.54e-7、membership差2.95e-6；正式更新0，原报告保留。
修复加原模型连续两次生成的独立波动见证；query最终状态、anchor最终embedding、Gaussian、Fm及q_pre仍要求差0，class logits/membership各自要求原生重复与干预差都≤1e-5且finite。
attempt04基线见证：原模型membership重复差4.77e-6、零干预差3.46e-6；两者logits差9.54e-7，上游全部严格相同。
单卡基线／局部读取／局部视觉证据的真实两步联合更新及原官方真实窗口评分已通过；其余预检仍进行中，不是正式训练完成。

预检／20步启动执行SHA410bd30b74a613919776ce4531a60fd2338844c6；预检64667，启动64668/64669/64670/64671，统一启动gate64672。
只替换尚未开始的续跑及评测为7468268f239b562feaf793b9d52a3b126f4cd860，任务64716/64717/64718/64719、汇总64720。
原续跑64673–64677在PENDING时取消，没有重复训练。18项数学文件/受保护函数/正式update loop AST身份均与实际预检快照一致。
serializer_only_preflight_reuse.json明确记录两个执行SHA、无损roundtrip及数学合同。新快照只排除Git内历史group_plus图像/分析副本，全部实际科学代码及绝对路径资产/数据保持。
当前证据：`/space/mawb/ssst/group_plus/vggt_reading_controls_v1/attempts/attempt04`。

用户再次明确授权清理旧实验文件。已删除26个文件：旧feedback factorial三组15个checkpoint、当前完整源已被epoch4替代的epoch1–3三个checkpoint、完成诊断的8份features.pt。
释放78.21GiB，实测可用91.34GiB；源epoch4、legacy epoch6、C/Q补丁、VGGT资产、当前任务与小型指标/分析记录fingerprint不变。
清单/逐文件删除日志/回执：`/space/mawb/ssst/group_plus/storage_cleanup_20261014_reading_controls`。
旧feedback短对照不再保留可重新运行的模型权重；小型官方指标及结论仍可比较，未假称保留所有历史大文件。

进度：`python3 scripts/show_vggt_reading_controls_progress_v1.py`，显示正式更新、窗口曝光、作业状态和不含排队/评测的训练ETA。


## 2026-10-15 排队与评分恢复

复核实际状态：64667单卡/八卡预检全部完成；四组64668–64671均完成正式20更新/160窗口曝光，64672统一启动核验通过。
object_local_read的64717已完成300更新/2400曝光和128窗口八rank导出，但官方评分环境因入口顶层导入模型而缺少lpips失败；训练和导出并未失败。
该失败记录保留；reliable_depth的64719因上述评分失败被依赖取消，未进行20更新以后的训练。

官方评分、reduce和summary入口现不导入TokenGS模型；仅export延迟导入模型环境。
在固定官方Python环境验证纯评分入口及官方worker导入通过，18项训练数学合同仍与实际GPU预检快照410bd30一致。
修复执行快照649ef09aef473128849539baf7453654033ce281；共享存储无损checkpoint参数、optimizer、RNG和计数不变。

将尚未启动的64716/64718（PENDING、RunTime=0）取消替换，原记录保留。
新作业64851只复用object_local_read既有128窗口进行官方评分，绝不重新训练或导出。
64852 baseline、64853 anchor_visual、64854 reliable_depth均等待64851通过后，分别从自身20更新checkpoint精确续跑至300并评测。
三组之间不再存在无关训练依赖；申请同型3090八卡，允许11/13，排除12，避免固定等待单一节点。
64855等待四组成功后汇总原定13列结果。64851已实际在3dimage-13启动。
恢复收据：attempt04/schedule_scoring_recovery_20261015.json；评分导入修复/数学合同证明：attempt04/scoring_import_recovery_preflight_reuse.json。

恢复现场核验：64851完成128窗口/8rank官方评分，各rank官方CPU/GPU状态合并合同通过，evaluation/COMPLETE.json为READING_CONTROL_EVALUATED。
64852已实际在3dimage-13八卡启动；64853等待资源、64854等待优先级，均无跨组训练依赖。四组此前均已完成20正式更新/160曝光及统一真实启动核验，按约停止持续轮询，让续跑与自动评测继续。
