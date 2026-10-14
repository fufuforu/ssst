# 完整权重理解失配复核：2026-10-14

**两种失配在完整训练后仍存在。** 原生匹配query的mask与对象存在性仍不一致；固定mask/类别的GT评分诊断仍显示较大排序差距。此结果支持针对query对象存在性、mask支撑和质量评分的一致性做后续对照，不支持仅继续延长同配方训练作为主要改进方案；尚未确定哪一模块是因果根因。

上次A/B/C均adapter理解梯度系数0.01，均只联合训练500步。A原视觉训练/feedback0.1；B视觉冻结/feedback0.1；C原视觉训练/feedback0.05。此前44/321和mAP0.236063→0.322882均来自C。因此本次与此前引用诊断的研究参数相同，主要区别是联合训练时长500→4172步，并非本次才改feedback。

## 执行与同样本核验

完整checkpoint为适配2epoch边界后联合4epoch、adapter0.01/feedback0.05的最终权重，SHA ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0。文件逐字节SHA校验、原生official checkpoint loader、recipe/config/适配资产/累计6258更新与50064曝光、新增联合4172与33376时钟通过。

GPU诊断64161在13号四张3090完成，COMPLETED0:0，5分55秒，08:45:14至08:51:09。执行快照SHA2fe23430e6dc76562c075ad5b7f5e493d16acc2a。四rank各一窗口预检成功后，同快照执行此前seed4242固定随机128场景的128窗口；总计132次冻结前向，零优化器调用。未改模型、head/loss、feedback、训练冻结范围或几何政策；推理使用eval/不求梯度，不是新增冻结训练对照。

128窗口名称、manifest indices、GT类别和实例ID与之前C逐条相同，共492个context、507个target GT实例-窗口条目。每个窗口输入RGB provider哈希和两视图分割PNG逐帧重现这次完整官方导出；GT只在context-only生成之后用于诊断。四rank参数版本未变，全部记录及汇总完整，用户队列为空。

排序检查使用保存的原生COCO RLE状态，不重复渲染/GPU评分。完整312场景1860窗口的输入/新视图baseline mAP、AP50、AP75独立重算与官方值匹配至1e−12。另筛出与此前C相同248场景1478窗口，原生AP groundtruth状态逐条相同，不平均分片或场景AP。

## 1. 好mask匹配query仍被判no-object

匹配用原生context Hungarian的类别/BCE/Dice联合cost；hard mask IoU为原官方渲染上的离线诊断，不替代原生soft matching cost。两组共492个GT，均完全分配。

| 权重 | 好mask匹配query（IoU≥0.5） | 其中no-object | no-object/好mask | 所有匹配query的no-object | GT类别argmax正确 |
|---|---:|---:|---:|---:|---:|
| C500，adapter0.01/feedback0.05 | 321 | 44 | 13.71% | 110/492（22.36%） | 354/492（71.95%） |
| 完整4172，同参数 | 334 | 50 | 14.97% | 126/492（25.61%） | 338/492（68.70%） |

mask质量变好时两组的“好mask”分母也变化，不能只用14.97%与13.71%的差值断言显著恶化。进一步对齐两次均有好mask的同一313个GT条目，no-object为40→41；12个由no-object变object，13个由object变no-object。此配对证据说明该失配未随训练延长而消失，不证明全部GT发生系统性恶化。

334个好mask匹配query中，另外25个预测了错误对象类别（未判空）；此前为21个。原生Hungarian分配并非IoU-only，也不是“任一未匹配query偶然有好mask”就应被保留。不能直接取消no-object筛选或认为分类头是唯一错误位置。

target类别无关原始mask覆盖：338/507→348/507（66.67%→68.64%）；最终保留mask覆盖：275/507→279/507（54.24%→55.03%）。同类别最终覆盖248/507→249/507，改善很小。长期训练能小幅增加可用mask，但对象存在性/类别与最终输出仍是候选瓶颈；这些诊断召回不是官方mAP/PQ。

## 2. 置信度排序差距仍存在

固定所有预测mask与label，仅在离线评分中用其对同类别GT的最佳IoU替换分数；原生COCO AP实现不变。

| cohort/权重 | 输入原生mAP | 输入GT分数诊断 | 新视图原生mAP | 新视图GT分数诊断 |
|---|---:|---:|---:|---:|
| 完整312场景，4172步 | 0.261360 | 0.352805 | 0.246245 | 0.333883 |
| 相同248场景，4172步 | 0.250760 | 0.339494 | 0.237035 | 0.320847 |
| 相同248场景，旧C500 | 见旧报告 | 见旧报告 | 0.236063 | 0.322882 |

完整312场景的排序诊断差距为输入9.14、新视图8.76个百分点；相同248场景新视图为8.38个百分点，旧C500为8.68。两次同cohort实际新视图mAP仅提高0.0973个百分点。GT评分诊断不是模型成绩、可部署方案或保证可达收益，不能将0.333883填进官方13列。

当前分数是类别概率乘获胜像素平均membership，没有直接评估mask与对象质量的一致性。该观察说明继续训练后分数质量仍有可测量问题，但不能排除数据、优化、mask投影或query表征的共同作用。

## 对下一步的含义

保留adapter0.01和双向结构；本次没有证明feedback0.05整体最优，也没有重新调β。
优先小预算验证query读取自身mask/几何支撑后，再使用原分类头，能否改善对象存在性/类别与mask一致性，同时观察native mAP和排序诊断差距是否缩小。应使用同起点、同样本、同预算的原结构对照，不直接启动又一次完整训练。
理解loss分项梯度来源是独立稳定性任务，未混入此次只读复核。尚未实现或提交上述结构训练对照，待用户决定。

## 工程与产物

旧A/B/C CLI及真实128场景汇总的results JSON回归完全一致；旧C的492/321/44/110/354计数独立重现。新Python/Bash语法检查通过，真实四卡预检与正式128前向均通过。

首次CPU状态长度检查误将原生segmentation AP未使用的bbox空列表当成错误，失败在评分结果产出前。已按原生helper保留空状态，只重排/筛选逐窗口状态；零模型前向/更新，不改metric/GT/mask/label。失败收据保留在engineering_failures，修复后CPU诊断成功。运行中GPU不可变快照未修改。

- [完整复核JSON](/space/mawb/ssst/group_plus/vggt_full_b005_bottleneck_recheck_v1/review_final_20261014/results.json)
- [完整/同248排序诊断](/space/mawb/ssst/group_plus/vggt_full_b005_bottleneck_recheck_v1/cpu_review_20261014/ranking_oracle_diagnostic.json)
- [128场景原始query阶段汇总](/space/mawb/ssst/group_plus/vggt_full_b005_bottleneck_recheck_v1/attempts/2fe23430e6dc76562c075ad5b7f5e493d16acc2a/formal/analysis/results.json)
- [提交和固定执行SHA收据](/space/mawb/ssst/group_plus/vggt_full_b005_bottleneck_recheck_v1/submissions/2fe23430e6dc76562c075ad5b7f5e493d16acc2a.json)
- [完整训练的原官方13列](vggt_feedback_b005_full_results_20261014.md)

CPU比较复现入口：scripts/analyze_vggt_full_b005_bottleneck_recheck_v1.py；提供新/旧raw目录、新/旧ranking JSON和新输出目录。源码只新增只读诊断/分析，不修改训练runtime。
