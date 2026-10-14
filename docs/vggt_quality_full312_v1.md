# 固定Q的312场景复评

用户授权只固定Q扩大场景复评，并先解释confidence分层及是否需要定位特征读取瓶颈。此轮不训练Q、不调分、不提交confidence或新结构试验。

Q固定在300更新／2400窗口的头checkpoint，SHA `61dd9a8fbd8915c1c8779e61c6e0a3997b2b3d15a2a9639e297b1c0946ffcee5`。
原C为adapter理解梯度系数0.01／feedback0.05完整权重及对象专属mask支撑分类读回300步补丁。C、Q整体冻结eval，无优化器。
VGGT/MASt3R、原双向6/8/10/12、loss和geometry_quality_policy=monitor_v1保持。核验新执行源码的数值文件与已通过的Q训练／评测不可变快照相同。

官方312场景、1860窗口、11160张图像，pair文件SHA `59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b`。
复用C已完成的全部RGB、depth、GT、panoptic PNG及重建评分，原128窗口Q JSON及八分片原生评分状态复用；仅为其余1732窗口计算固定Q评分。
每个窗口只用两张context生成C分类LN后的100×256特征，评分为原canonical C分数乘sigmoid(Q特征)，不引入目标图像或GT到头的输入。
数据provider完整batch仅用于核对输入哈希、frame IDs；额外目标图像不参与后续Q新特征推理。

使用13号8×3090、32CPU、384G。八rank各重现一个旧Q窗口及一个新窗口：完整原校准／六图渲染、RGB/depth/GT与C复现、PNG／ID／类别及分数排序、原Q评分复现，且验证校准／渲染没有改变context生成的分类特征。
16个真实witness全部通过后写STARTUP_VERIFIED.json，再续跑新1732窗口的context特征推理。图像通过只读源文件软链接复用，每个新评分JSON独立拥有，旧证据不改；不重复写入数GB图像。
128复用窗口不做重复正式推理／评分，witness是单独的完整性检查。新窗口评分仍使用原生SIU3R evaluator，逐rank独立CPU/GPU状态合并合同。

CPU汇总依赖GPU导出及评分成功结束。汇总检查所有1860名称／source_index唯一且完整、312场景、128复用状态、原生语义指标与未变PNG一致。
保存新原生合并结果，再复用原C未变mIoU/PQ；Q mAP重新原生全局合并，整张官方13列，mIoUt两列空。
另用原生保存状态重算原128窗口mAP，须与旧Q精确一致；检查原128筛选场景以外184场景（1097窗口）AP，逐类比较C/Q。184场景为AP诊断，不另创建三套13列表格。

证据root：`/space/mawb/ssst/group_plus/vggt_quality_full312_v1`。已有C完整评测与Q训练／128评测保持原路径。
提交收据、不可变快照、witness、每rank导出／评分进度及analysis/siu3r_comparison.csv/.md/results.json保留。
确认八rank真实startup后停止主动轮询，让扩评及汇总继续，不等待全部完成。

```bash
python3 /space/mawb/ssst_vggt_overfit_diagnostics_v1/scripts/show_vggt_quality_full312_progress_v1.py
```

## 后续诊断的边界（本轮未执行）

confidence分层：固定模型，按context预测confidence排序／分位分组，离线比较深度误差、跨视图一致性及mask错误。confidence是非概率分数；GT仅作离线参照，不改attention，不通过验证集选择推理门限。
应控制场景／可见性／物体类别／深度和边界区域差异，避免把不同区域的天然难度误称为confidence因果作用。即使相关，也要用实际门控干预验证。

先定位信息来源／读取／输出三个位置，而非预设特征不足。Q在固定特征上提高AP，证明至少部分输出评分可改；C对象支撑读回的局部收益、旧U1/U2对anchor读取的作用与U3无增益，尚未证明encoder特征缺失或整个decoder上限。
可先用当前C固定特征做同容量分类／mask质量探针，比较query、对象支撑特征和不同encoder层；对象池化优先用预测支撑，GT区域仅作单独定位诊断，不能把它的定位优势当作特征强弱证据。
若已有局部特征有信息而query没有，才针对读取／对齐；若query已有信息而输出差，先改头与监督；若多个读取位置都欠缺，再验证DINO／多层GFM新融合。
扰动读取分支只说明当前模型依赖它，探针可解码只说明信息存在，都不能单独证明修改该模块会提高正式指标；最后仍需同预算受控训练。
depth confidence偏置属于几何信息如何使用，不是新增外观特征，不能将三种提案视为同一瓶颈的等价修复。

## 本轮提交

CPU评分JSON／query ID对应与只读图像复用合同2项通过，官方评分环境CLI导入通过；
源checkpoint SHA／数值文件不变、1860窗口／312场景、128复用／1732新增及184独立场景1097窗口清单核验通过。
执行快照`4cebe1ab76cc6a027e4697e397405496fc4c7e87`已提交推送。
固定Q八卡评测64392在13号运行；CPU自动汇总64393依赖64392成功结束。
提交时Git旧7890代理失效，使用环境提供的17899代理进行仅本次命令覆盖，未改全局配置；未产生重复评测或训练。

真实八卡启动通过：STARTUP_VERIFIED.json记录16窗口（8个旧Q、8个新增C），旧Q评分、C RGB/depth/GT／mask复现，
context生成特征与完整渲染流程一致，全部参数冻结、优化器更新0。独立检查各rank witness的输入hash／frame IDs／source_index与原C记录一致。
停止前八rank新增导出分别17/16/16/18/21/20/22/23窗口（每rank216或217新增，另16复用），
最慢rank剩余导出估计约11分钟，不含后续原生评分及CPU汇总。
独立证据startup_independent_audit.json与POLLING_STOPPED.json保留；停止主动轮询，让64392/64393继续。

## 已完成

64392成功结束（COMPLETED/0，22分17秒），64393成功结束（2分2秒）。独立完成核验保存于analysis/completion_independent_audit.json。
全部1860窗口／312场景、八rank原生合同、128旧Q状态复用／AP重现通过；原重建、PNG及语义指标保持。
输入mAP 0.2625081241→0.2668554485（+0.434732个百分点），新视图0.2471912354→0.2514087856（+0.421755个百分点）。
独立184场景1097窗口AP提高0.470799／0.502560个百分点，支持新增场景上的评分增益；单seed，不宣称所有类别获益。
全312输入18类中11类AP上升、新视图10类上升。13列完整结果已自动输出至analysis/siu3r_comparison.csv/.md。
此结果提供输出排序可改的依据，仍不能单独定位encoder／anchor／query的表征或读取瓶颈。
