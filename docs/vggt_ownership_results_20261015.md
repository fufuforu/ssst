# Object–Anchor–Gaussian 固定权重诊断：2026-10-15

## 判断与下一步

**最明确定位到的下游瓶颈是类别/no-object筛选；原始Mask覆盖，尤其小物体，仍是更大的未解决来源。重建的继续下降体现联合优化的取舍。当前证据不能把Object–Anchor错误写回或Anchor–Gaussian不一致确认为共同主因。**

具体区分四个问题：

| 候选瓶颈 | 本轮判断 | 直接依据 |
|---|---|---|
| Object–Anchor Read/Write | 有空间错位和跨GT写入，但因果责任未确定 | 真对应优于打乱；控制物体面积后，“不一致越大、Mask越差”的关系消失或反向；高不一致也有正确实例 |
| Anchor–Gaussian实例归属 | Anchor/geometry分支本身很弱；不能归因为Child破坏了良好Anchor | C的Anchor继承渲染仅100/492个GT有好候选，geometry child为104，融合视觉后的Raw为358 |
| 最终分类与Mask组装 | **已有好Mask的主要额外损失发生在no-object资格筛选**；像素竞争次之 | C：358→294→289→288；Baseline：354→293→287→287。类别错误另使最终几何好候选288→同类好候选257 |
| 联合训练优化 | **确认重建退化，未见最终Mask整体塌缩** | PSNR −0.3164 dB，所有标注区域RGB误差上升；最终best-query IoU差−0.0006，区间跨0，mAP反而上升 |

134/492个C实例在Raw阶段就没有任何IoU≥0.5候选，不能靠最后分类补救。这134个中85个是面积<1%的小物体。冻结观测不能继续把这部分唯一分配给“特征不够丰富”或“读取能力不足”；两者仍需干预才能区分。

**目前不支持直接实施“统一Ownership强约束Read/Write/Child”的大改，也不支持直接增加到全部交互层。** Read保留全局上下文有实证合理性；Write定位性在L8/10较强，L12又减弱，未呈现层数越深越一致的趋势。以视觉修正的soft ownership组织非对称交互可保留为假设，但不能宣称它已被证明能解决当前分类损失或PSNR下降。

**只推荐一个优先结构方向：竞争感知的Gaussian对象分类读回。** 在当前C的每Query独立、可见Gaussian特征加权平均读回之上，让分类残差显式看到本Query的融合视觉支撑，以及与它争夺同一Gaussian/Mask支撑的其他Query，区分“好对象候选、重复候选、错误类别”。沿用原19类含no-object头、原loss和官方组装，先只改这一个分类读回模块；零初始化复现C，冻结重建/Mask/feedback路径，与等容量的独立Query读回比较。验收看GT级筛选损失、同类最终Mask覆盖及官方mAP/PQ共同改善，不能只数保留Query或只看AP。该方向不承诺恢复134个Raw失败或修复PSNR。

这与此前C的独立均值读回、旧O的固定单Query特征二元存在性残差不同。旧O的300步实验没有收益，本轮不建议原样重复。也不再次把已有效但收益有限的Q排序头当作Mask/存在性问题的解答。**这里只提出设计，没有改模型或提交新训练。**

## 完成与可比性

- C：完整adapter0.01/feedback0.05模型，加原object-support分类补丁；Baseline：同一C继续联合300更新/2400曝光的既有终点。
- 三个源权重的实际SHA、两模型128输入/帧ID/GT文件hash、同一冻结encoder来源、成熟feedback时钟和官方源码commit均已核对，见证据root的`comparability.json`。未加载Q。
- 每模型128个固定留出场景，各一窗口；相同492个context GT实例，含两个无thing GT窗口。GT仅用于预测完成后的匹配、诊断与病例选择，不进入生成。
- 原作业64911完成C128和Baseline92后，Baseline一张target预测出现1像素边界差异，严格复现校验停止；历史失败保留。恢复作业64963 **COMPLETED 0:0，27:50**，复用220窗口，仅补36个缺失窗口。
- 最终256窗口的segmentation/GT PNG与各自原评测逐位相同，候选ID/类别相同，候选分数容差1e-6，RGB容差≤1量化格；本次恢复36窗口均第一次观测通过，没有重试挑选结果。问题窗口的无观测重复与有观测比较：Gaussian/query/anchor差0，membership差≤3.9e-6，context panoptic差0。
- 6份分片完成收据、256个JSON/NPZ覆盖与配对核验通过，**新optimizer更新0，模型参数/buffers均未变**。CPU数学/GT打乱测试5项通过；主分析、补充审计和病例图均完成。
- GPU执行快照为`6d716520a65f6c7e997e1b22088d26de32486a19`和仅恢复工程改动的`9cbc2b0b0e98c58d5762f282cb95e8442b9141b9`。CPU分析补充了面积/物体数控制与病例复核，不重新执行模型。

证据root：`/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1/attempts/attempt01`。

## 正式13列：同128场景，保持原协议

理解指标为0–1；“新视图”沿官方target集合，含2输入+4额外帧，两个mIoUt空白。mAP/PQ复用已严格复现输出的官方全局合并状态，未平均分窗口AP/PQ。

| 模型 | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 输入mIoUs↑ | 输入mAP↑ | 输入PQ↑ | 输入mIoUt↑ | 新视图mIoUs↑ | 新视图mAP↑ | 新视图PQ↑ | 新视图mIoUt↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | .08730 | .2132 | 22.5966 | .7808 | .3680 | .56024 | .30799 | .64552 | | .55265 | .27376 | .60979 | |
| Baseline-300 | .08868 | .2156 | 22.2801 | .7747 | .3722 | .55690 | .31260 | .62795 | | .55071 | .27865 | .60311 | |

Baseline相对C：输入/新视图mAP **+0.461/+0.489个百分点**，PQ **−1.756/−0.668个百分点**。存在排序/检测与panoptic质量的取舍，不能说理解全面退化或全面提升。

## 实例质量在哪一步下降

以下按同一GT在全部100个Query中的**最好几何IoU≥0.5**计数，允许忽略类别、多个GT独立选Query。它是阶段覆盖诊断，不是AP/PQ，也不是可部署的GT oracle输出。

| 阶段 | C /492 | Baseline /492 |
|---|---:|---:|
| 最终Anchor membership原样继承到Child并渲染 | 100 | 84 |
| geometry child特征membership渲染 | 104 | 92 |
| 原模型融合视觉后的Raw Mask | 358 | 354 |
| 原类别/no-object资格筛选后 | 294 | 293 |
| 原像素竞争后 | 289 | 287 |
| 原面积过滤后最终Mask | 288 | 287 |
| 最终Mask再要求预测类别正确 | 257 | 253 |

Anchor/geometry的额外渲染固定最终Gaussian几何与相机，仅替换诊断feature channel。**原视觉融合是在补救弱几何实例支撑**；不能强迫融合后Child退回遵从弱Anchor，也不能据此把现有视觉特征判为整体无用。

C在Raw之后净少70个好几何实例，其中64个在资格筛选损失；像素竞争实际丢10、救回5，面积过滤再丢1。Baseline的对应净损失67个，其中61个发生在资格筛选。资格阶段尚未检查类别是否正确，所以类别错误还会造成独立语义损失。

保留原生Hungarian一对一Query时，C的333个好Raw中：260个类别正确、49个no-object、24个类别错；Baseline：255/49/27，共331个。C的49个好Mask/no-object平均Raw IoU **.731**，最终本Query IoU为0；不是全由原始Mask太差解释。不能把49个Query全部当49个GT彻底丢失，GT级best-query计数已另行处理接替者。

小物体定义为两context帧GT面积占比<1%：123个中，C的好Raw候选仅38个，最终19个；Baseline为33→18。<0.1%的33个极小实例，C仅2个有好Raw，最终0个，Baseline为1→0。小物体主要在Raw之前已经失败。

C有313个GT对应≥2个好Raw候选，Baseline有309个。因此一些no-object抑制和像素竞争是正常去重，不能把全部Query损失当错误；同时也说明只给每个Query独立调阈值不足以决定保留哪个候选。

## Read / Write / Child及负对照

Read取真实8头CA；Write同时取真实路由及通过投影、tanh、norm cap后的feedback贡献。在观测公共gate固定时，各Query贡献向量和重建实际feedback，全部窗口最大误差 **8.85e-9**。实际feedback/token范数比约**1.25%**，接近现有`.05 × .25`限制，Write没有数值爆炸的证据。

Child按真实1024×64 parent-major顺序、context可见质量汇聚回Anchor；不是几何最近邻。下表是492个原生匹配Query的空间JS（0–1，低表示空间分布更相似）和visibility加权soft Dice。Read需要上下文，JS高不等于错误。

| C层 | Read–Child JS /Query打乱 | Write–Child JS /Query打乱 | 实际贡献–Child JS /Query打乱 | Anchor–Child Dice /Query打乱 |
|---|---|---|---|---|
| L6 | .724/.779 | .689/.763 | .688/.760 | .319/.075 |
| L8 | .725/.786 | .614/.778 | .617/.778 | .308/.117 |
| L10 | .730/.787 | .625/.778 | .629/.779 | .335/.119 |
| L12 | .718/.791 | .681/.774 | .678/.771 | .302/.058 |

每窗口16次Query无固定点打乱，另有独立Anchor空间排列控制；真对应的增益不是随机对应制造。C L12 Write JS相对Query打乱降低 **.0924 [95%场景bootstrap .0731,.1127]**；Anchor/Child Dice提高 **.2444 [.2140,.2751]**。Baseline走势相同，L12 Write JS为.685/打乱.771，Dice为.226/打乱.048。

GT写入统计用预测完成后的context GT renderer-adjoint质量。仅在GT物体可见写入中，C L12约72.0%贡献范数投向其他实例；限制parent纯度≥.8仍为72.0%。但总可见写入中约44.2%落在GT物体、其他实例写入约31.8%，**不能把72%称为全部feedback的错误率或损伤率**。其余可见质量包含stuff/无thing标注区域；范数归因也不是有符号语义损伤。

在物体数≥2、写入有效且diagnostic owner与matcher一致的474个C实例中，真实自身GT偏好相对GT标签打乱的增益：L6 **2.18个百分点**、L8 **12.02**、L10 **10.14**、L12 **3.20 [2.43,4.08]**。Baseline为2.35/12.43/10.48/3.46。中层比末层更专属，但四层都强于对应打乱；不存在“全程随机写回”的结论。

## 与错误、AP/PQ的关系：需要面积控制

C L12好Mask正确类/好Mask no-object/好Mask错类/差Mask的Anchor–Child Dice分别为.397/.236/.435/.148。**错类组甚至比正确类组更一致**；其Raw IoU均值.791、最终几何IoU .793，主要问题是类别。差Mask组的GT写入选择性确实较差，但小实例天然占很少可见Anchor，强烈混淆绝对foreign比例与空间分布距离。

| C L12与Raw IoU的Spearman | 未控制 | 只控制GT面积 | 控制面积、实例数、贡献量 |
|---|---:|---:|---:|
| 跨GT写入比例 | −.582 | +.040 | +.053 |
| Write–Child JS | −.362 | +.418 | +.416 |
| Anchor–Child Dice | +.529 | +.026 | +.024 |

控制后的Dice/Raw相关区间 **[−.071,+.119]**；Write JS/Raw为 **[+.330,+.485]**，方向反而与“越不一致越差”的解释相反。仅控制面积就发生这一变化，不能归咎于多控制了贡献量。Baseline同样出现反向/近0。

这不是“跨GT写入有益”的因果证据：读写指标与输出共同由模型/物体尺度产生，部分相关也不能消除所有混淆。它说明**现有相关性不足以把一致性作为优化目标**。最终IoU控制后Dice关联在C仍有小正值.140 [.035,.236]、Baseline .088 [−.017,.201]，也没有统一的强关系。

每窗口官方性能关联同样没有清晰统一方向。C的Write JS与输入/新视图AP相关为−.109/−.129，与PQ为+.062/+.054；跨GT比例与AP约+.032/+.040、与PQ约+.379/+.372。不能依据一张粗相关图认定foreign Write导致mAP/PQ损失。分窗口指标只用于关联，未替代正式全局指标。

## C→Baseline联合训练漂移

按相同GT实例对齐，不假设相同Query号或Anchor编号跨模型有相同语义。括号为场景成簇bootstrap的95%区间。

- 原生匹配Raw IoU均值变化 **−.00229 [−.00785,+.00371]**，最终IoU **+.00637 [−.00657,+.02031]**。
- 独立best-query Raw变化 **−.00070 [−.00507,+.00368]**，最终 **−.00060 [−.00886,+.00725]**；未见实例Mask整体退化。
- L12 Anchor–Child Dice **−.07658 [−.09389,−.05999]**，变化明确；但Write–Child JS仅 **+.00356 [−.00088,+.00826]**。Dice变差没有伴随最终Mask整体同幅变差，视觉分支可能在补偿。
- 原生匹配分类正确341→337；18个正确变错误、14个错误恢复。no-object 124→127，16个恢复、19个新增；Query切换时需结合best-query结果解释。
- 跨GT写入比例差均值−.00133；Ownership变化与最终IoU变化的相关很弱：Write JS变化ρ=−.018，Anchor/Child Dice变化ρ=.012。不能把联合漂移的输出变化归于单个一致性指标。

PSNR正式差为 **−.3164 dB [95%场景配对区间 −.4157,−.2137]**，直接使用原官方逐窗口render_scores计算。按GT分互斥区域、在相同六帧上统计：

| 区域 | 占全部像素 | RGB MSE相对增加 | 占净SSE增加 | 场景配对区域PSNR变化及95%区间 |
|---|---:|---:|---:|---|
| 物体内部 | 32.94% | +7.96% | 35.30% | −.328 [−.464,−.199] dB |
| 2像素语义/实例边界 | 7.00% | +7.83% | 11.50% | −.360 [−.477,−.248] dB |
| 背景stuff内部 | 36.26% | +7.63% | 27.37% | −.342 [−.512,−.178] dB |
| 其他/无有效标注 | 23.80% | +6.30% | 25.83% | −.094 [−.238,+.063] dB |

区域PSNR按每窗口有效帧先平均、再场景平均，物体内部/背景各127个有效场景；SSE使用全部128场景，区域像素互斥且完整覆盖。区域PSNR固定data_range=1，官方torchmetrics按GT推断range；绝对区域PSNR不冒充正式分数，相同GT配对差消除共同range归一化。

物体、边界、背景的SSE增加区间均大于0。**退化不是只集中于物体边界或跨实例区域，是更广泛的重建优化取舍。** 两encoder源固定，这里所说漂移是下游表示/解码路径与联合目标的端点变化，不是证明encoder自身退化。只有一条训练轨迹，不能据此断言哪个梯度或交互模块造成了全部下降。

## 典型案例（原始两context联合IoU）

经人工复核的完整图集：[reviewed_gallery_v2/gallery.md](/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1/attempts/attempt01/analysis/reviewed_gallery_v2/gallery.md)。每图按GT对齐两模型；绿色TP、红色FP、蓝色FN；图内标一帧IoU，各ownership列独立色阶。

| 场景 / GT | 量化现象 | 说明 |
|---|---|---|
| scene0050_00_context1441_1460 /2 | C Q5 Raw .945，no-object，最终本Query 0、所有Query最好仅.0066；Dice .809、Write JS .261 | 所有权已较一致，仍有高质量候选被no-object丢弃，且未获其他Query有效接替 |
| scene0144_01_context1068_1086 /9 | C Raw .966、最终几何.967；GT chair→sofa；Dice .891，跨GT写入1.14% | 强一致性不能保证类别正确 |
| scene0231_02_context2510_2543 /3 | C Q57 Raw .883→像素竞争.0002→最终0；GT仍由其他Query达到.632。Baseline Q57最终.847 | 重复Query争夺可让原生匹配Query消失，不能直接称GT彻底漏检 |
| scene0549_00_context376_392 /12 | Anchor继承/geometry child IoU均0；融合Raw .966、最终.970；Write JS .944、跨GT写入99.12% | 视觉融合补救弱Anchor；高不一致可与优质正确实例共存，是关键反例 |
| scene0277_01_context36_108 /5 | 小door仅566像素，C Raw .184、跨GT写入99.97%；Baseline Raw .250 | 存在差Mask/非专属Write共现；与上例对照后不能认定Write是差Mask原因 |
| scene0553_00_context607_629 /6 | C→Baseline Q59 Raw .906→.903，final-best .877→0；Anchor/Child Dice .776→.777，Baseline变no-object | 下游筛选可突变，而Raw和Ownership几乎不变；C本身类别也错误，不能称其原为正确panoptic实例 |
| scene0658_00_context6_68 /1 | C door分类正确、Raw .819/最终.820；Read JS .989、Write JS .962 | 全局Read与局部输出差异很大，仍能正确预测，不能强制Read局部化 |

## 可复查产物与边界

- 主结果：[analysis/results.json](/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1/attempts/attempt01/analysis/results.json)。
- 覆盖、负对照区间、分组、漂移和关联复核：[completion_audit_and_sensitivity.json](/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1/attempts/attempt01/analysis/completion_audit_and_sensitivity.json)。
- 面积/实例数/贡献量逐项控制：同目录`partial_rank_covariate_sensitivity.json`。
- 每Query各层ownership、各GT阶段IoU、跨模型漂移：同目录`query_ownership.csv`、`GT_aligned_instances.csv`、`GT_aligned_drift.csv`。
- 诊断概览：同目录`diagnostic_overview.png/.pdf`；不把阶段GT覆盖计数称为正式AP。
- 归因是观测gate下的精确加法分解，未做删除Query的反事实；早层所有权对齐最终parent可见GT，不能当成早层位置真值；归档ownership数组FP16仅用于可视化/GT打乱敏感性，主统计FP32，重建误差容差复核通过。
- 负对照是离线破坏对应关系，不是修改模型前向的干预；相关/端点漂移不证明因果。探索性区间未做多重比较校正，128场景结果不等于完整312场景结论。
- 本轮没有新训练、模型结构改动或评测协议改动；未提交后续GPU作业。
