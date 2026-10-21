# C固定特征的质量排序／对象存在性对照

2026-10-14用户授权按三组方案执行，八卡若更快则用八卡。共同起点是adapter0.01/feedback0.05完整适配2＋冻结联合4源权重及C对象专属读回300步补丁（SHAad3100ceabbd4d9ab41f653cdb40ec1cb86dc882d71d9ad5ac14192947f4f0f9）。R固定C，Q仅新增质量头，O仅新增对象存在性头。原VGGT/MASt3R/adapter/decoder/query/GS/mask/分类及C读回全部冻结并eval，原双向6/8/10/12及feedback继续。

两个新头同输入：C分类LN后、原class_head前的100个256维query特征；同容量256→128→1 GELU、相同seed123456、末层零初始化。全部输入detach。原模型与原loss定义不改；新增诊断监督只更新新增头。

Q训练目标为C最终保留的context实例mask对同类别GT的最大IoU，无同类重叠为0；只在context非空最终实例上监督。SmoothL1(beta1)。推理分数s_Q=s_C×sigmoid(quality_head)，仅改pred.json评分，候选、类别、像素赢家和所有PNG不变。零初始化统一乘0.5，排序和AP保持，不宣称JSON字节相同。

O使用原C固定native Hungarian匹配的二元正/负目标，负例权重0.1，窗口内加权BCE。存在性logit为原19类logsumexp(前18)−最后noobject logit，加新头残差；候选概率相当于前18类logit统一加残差，条件类别比例不变。新的19类概率只供候选筛选，原argmax/noobject及0.05门限沿用。像素竞争与AP分数仍用原C类别概率/公式，0.5mask、0.05alpha、半面积过滤及void规则不改。O候选改变会改变最终PNG，需要评估全部理解项；原冻结分类头的noobject argmax并未改，分别统计原／有效门控拒绝好mask的数量，不能把两者混称。

两组各300更新/globalbatch8/2400不同训练窗口，完整8337训练池的原第一2400固定stream；不在128验证场景训练。相同原AdamW object_memory peak1e−4、betas0.9/0.95、decay0.05、clip1、200更新warmup、4172cosine截断300。头优化器fresh；质量和存在性是两个独立优化器，互不反传。窗口损失等权平均，Q空监督窗口精确贡献0。

为节省重复冻结C前向，两组共享**流式缓存**：每轮跨4/8worker产生同一全局8窗口特征与目标，GPU0聚合8窗口并分别更新两个小头，再进行下一轮。每窗口仅缓存一次，正式各自仍300更新/2400曝光；8卡每卡1窗口，4卡每卡2窗口。缓存保存feature/logits/target和源窗口/input哈希，便于后续只训练小头。小头GPU0优化只是实现全局batch8的同一窗口平均，预检对真实8rank平均梯度/两更新参数误差验证≤2e−6。

预检在13号8张3090/384G运行，热身后反序4/8/8/4比较同一8训练窗口的真实冻结C前向/缓存时间和数据身份。8卡中位速度提高至少5%则正式选8，否则4。分别测单GPU和8rank小头速度；昂贵冻结C由多卡分担，小头仅GPU0更新，不为它单独申请八卡任务。所有benchmark/smoke不计正式更新。数据固定窗口seed42+source_index以确保缓存重放及4/8等价；geometry_quality_policy=monitor_v1与原校准/训练数据数学保持。

CPU合同：新头零初始化/条件类别保持、整批与逐窗口平均一致、输入梯度停止/空监督、Q只变分数且PNG保持、O门控同时用于像素及实例列表。八卡真实预检含8个原同128验证窗口C重现、16训练窗口缓存、单/八卡真实两更新及五条件官方13列评分（R/Q0/O0/Q2/O2）；四/八缓存目标一致，原C参数/缓冲区保持，GPU/CPU官方合并合同通过后才提交正式。

正式流式缓存/两头训练共用一组所选GPU，300步后同进程复用已加载C进行128验证：每个窗口一次C生成，分别导出R/Q/O，不重复加载大型checkpoint。其后自动原生评分/全局状态合并与13列汇总。R与Q逐帧核对原C PNG；复用原C置信度作为排序基准，防止1ULP分数差扰动ties。所有RGB/depth/GT核对一致后复用既有重建评分。Q/R的未变IoU/PQ可复用原C原生值，先保存并验证新4/8分片原生值≤1e−7；AP重新原生评分，未修改SIU3R evaluator。

只进行128筛选，不自动拼Q/O或长训/扩312；完成后按mAP与门控/误检/PQ/mIoU取舍决定下一步。GT质量不进入推理评分。正式20更新/160曝光会自动保存startup_validation.json和可恢复头/optimizer/缓存checkpoint，核验后停止主动轮询，让300步及评测继续。

证据root：`/space/mawb/ssst/group_plus/vggt_readout_probe_v1`。提交收据、不可变快照与历史失败均保留。产物：training/cache_rankXX.pt、head_checkpoint_update0020/0100/0300.pt、evaluation各R/Q/O metrics、analysis/siu3r_comparison.csv/.md及results.json。

只读进度：

```bash
python3 /space/mawb/ssst_vggt_overfit_diagnostics_v1/scripts/show_vggt_readout_probe_progress_v1.py
```

首个真实预检64381在开始缓存前失败：原训练manifest窗口没有source_index字段，验证窗口有。
补齐训练窗口的原list位置，实际8337窗口和2400固定stream CPU检查通过，窗口数据未改变。
attempt01失败日志与ENGINEERING_FAILURE.json保留，正式更新/曝光均0。
修复后用新不可变快照/attempt02继续预检，不修改旧运行快照，不重复正式训练。

attempt02预检64388产生8个C参考窗口后，3个FP32末位分数差（最大1.7881393432617188e−7）
触发借用的1e−7检查。实际8窗口PNG、ID、类别、类别内分数排序全部一致，原始失败收据保留，正式仍0更新。
本诊断使用独立FP32分数复现检查≤1e−6，继续严格核验PNG/ID/类别/排序；R/Q仍使用原C缓存分数，
因此这些末位差不进入实际基准或质量组分数，旧312评测函数不改。新attempt03再预检。

## attempt03正式提交与实测资源

执行快照：`00e69458acea40f2fd9e74c0458d8cb692163b86`。预检64389完成（COMPLETED/0，16分26秒），
`preflight/PREFLIGHT_VERIFIED.json`确认五条件原生评分、零初始化复现、冻结参数及缓冲区、
四／八卡同训练窗口与目标、单／八卡两次真实小头更新等价均通过。

同一全局8窗口缓存实测四卡两次4.0232／2.9807秒，八卡两次2.0716／1.7982秒，
中位比为1.8099倍。因此使用13号八张3090、32CPU、384G内存，共享一次冻结C缓存流。
质量／存在性头GPU0平均更新耗时2.287／2.447毫秒；八卡归约为3.342／3.452毫秒，
故小头保持GPU0更新两个独立优化器，全局batch8及各300更新／2400曝光不变。
1.81倍是这次缓存阶段的测量，不能直接当作包含启动、导出、官方评测的整项任务加速比。

正式作业64390已在13号八卡启动，流式缓存与两个小头训练后自动导出并官方评分R/Q/O的128场景。
CPU汇总64391依赖64390成功结束；无需用户另行提交评测。
收据：`attempts/attempt03/submission.json`，预检及正式训练计数分开，历史失败保留。

正式启动检查已通过：`startup_validation.json`记录两组各20次真实优化器更新、160个新增窗口曝光，
相同全局batch8、固定sampler、原C参数与缓冲区冻结、两个新头末层非零。
20步checkpoint保存两个头、优化器状态及八个缓存分片。停止轮询前最近一次进度为各36/300步、
288窗口曝光，剩余训练估计12.7分钟（不含导出和官方评测，会随运行变化）。
停止主动轮询，让64390及依赖汇总64391继续；结果尚未完成。
独立读取20步checkpoint验证SHA、每个优化器状态step=20、有限且非零的头权重、
160不同窗口与八分片缓存一致，保存`startup_independent_audit.json`及`POLLING_STOPPED.json`。
