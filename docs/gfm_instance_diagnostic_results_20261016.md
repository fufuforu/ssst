# 当前GFM双分支的实例瓶颈诊断结果

66047已COMPLETED 0:0，耗时7分40秒。两模型各128场景、492个相同context GT实例，4rank各32场景；全部参数版本不变、参数更新0。两模型均为已完成4172步的V4：梯度隔离65831、整体理解权重0.05的65838。没有重训或改官方协议。

## 先给结论

当前理解端最明确的问题是：**SAM初始化槽位几乎不转化为最终实例；已生成的好Mask又被no-object及输出组装大量淘汰。小可见区域实例还有Raw Mask覆盖不足。** 尚无证据把首要瓶颈归为全局GFM特征不够或Anchor–Gaussian归属整体失效，也不能据此立即开启Write或强迫Read完全局部化。

## 1. 完成及复现边界

- 原66042失败于逐位相等检查，旧失败保留。本轮8份首场观测器收据均通过有固定上限的原生重复性验证：Gaussians、分类概率逐位一致；无观测器两次Membership差异约1.0e-6量级，观测器差异同量级；首场最终packed分割均无差异。此时才有证据支持前次小浮点差异来自原生重复波动。
- 两模型RGB身份、GT IDs/classes/pixels逐场景一致；真实分组重放q_pre/中心/尺度均通过，故slot来源不是凭query固定ID猜测。
- **0.05组与原官方768帧分割复现：767帧完全一致，另一帧只差1像素。** 原官方final context Recall@0.5=260/492，与诊断相同，可作为主要定量证据。
- **隔离组存在额外跨运行/跨硬件复现差异**：平均packed像素差异1.6078%，最坏一帧49.997%；不能把它解释成观测器误差或简单浮点容差已经解决。确认checkpoint_epoch04与官方checkpoint_latest是同inode同文件；当前18节点4090，原官方13节点3090，精确原因尚未定位。CPU原官方PNG复核：原final Recall@0.5=262/492，诊断259/492，7个GT跨越0.5阈值，best-IoU均值0.49064 vs0.48819。隔离组仅辅助参考，不用其细小跨组差异证明收益；旧官方指标不改写。

## 2. 同一批492个GT：损失主要发生在no-object筛选和组装

下表为忽略类别的GT-oracle best-mask Recall@0.5，**不是mAP或PQ**。同一阶段可有多个query覆盖一个GT，不是一对一检测召回。

| 阶段 | 隔离组（辅助） | 0.05组（主要证据） |
|---|---:|---:|
| Raw Gaussian渲染Mask |368/492=74.80%|385/492=78.25%|
| no-object/类别得分资格筛选后 |297/492=60.37%|297/492=60.37%|
| 像素竞争后 |274/492=55.69%|273/492=55.49%|
| 最终面积保留规则后 |259/492=52.64%|260/492=52.85%|
| 最终再要求类别正确 |223/492=45.33%|235/492=47.76%|

0.05组Raw→资格筛选丢88个，无恢复；资格→竞争丢40个、恢复16个，净丢24；竞争→最终丢13个。Raw385→Final260净少125，约32.5%的Raw可覆盖实例未能转化为同等最终覆盖。逐GT追踪是126个Raw-good变Final-bad，另有1个反向恢复，不能把净差当严格互斥损失分解。

这里的资格条件是19类softmax的argmax不是no-object且最大thing概率≥0.05；若argmax为thing则最大概率至少1/19>0.05，所以本轮资格淘汰实际由no-object判定决定。类别预测错误还会使最终同类有效覆盖从260降到235。

**不能把“best raw query被判no-object”的254个GT直接叫254个漏检。** 未匹配的重复query应预测no-object，另一个query可能仍正确保留；本报告使用实际资格前后每GT最佳Mask变化来确认真实88个损失。

## 3. SAM初始化槽位没有转化为输出实例

0.05组128场景共12800个query位置，5623个实际由SAM初始化、7177个为learned fallback：

- SAM槽仅10/5623通过资格筛选；fallback槽742/7177通过。
- native Hungarian给40个GT分配了SAM槽，但其中39个被预测no-object，只有1个类别正确；452个GT匹配fallback槽，其中331个类别正确。
- 原官方context最终420个预测实例中，**419个来自fallback初始化槽，只有1个来自SAM槽**（包括无GT场景的预测）。
- Raw有385个GT可覆盖，最佳Raw query中83个来自SAM槽、302个fallback；资格后仅2个GT的最佳有效query为SAM，最终260个可覆盖GT的最佳query全为fallback。
- 隔离组方向类似：5621个SAM槽中19个eligible；原官方420个预测实例中2个SAM、418个fallback。其精细数值受上述复现差异限制。

这些是“初始化时属于哪类槽位”的统计，**不代表fallback最终只用learned信息**：它们也读图像和Anchor，并可经query self-attention接收SAM槽信息。可以确认的是SAM槽位本身几乎没有成为最终实例，不能据此断言SAM对网络完全零作用。当前V3初始化隔离对照66039正好可检验这种现象是否与V4几何池化有关，本轮未查看或修改该训练。

## 4. 小可见区域实例在前后两端都吃亏

“小”固定定义为两个context合计有效GT像素<1024，共112个，并非物理尺寸或官方COCO AP-small。其余380个。

0.05组：Raw小实例36/112=32.14%，其余349/380=91.84%；Final小实例13/112=11.61%，其余247/380=65.00%。107个Raw失败中76个（71.0%）来自小可见区域实例。所以既有候选/Mask形成不足，也有已形成Mask被后处理淘汰。

Raw SAM逐可见视图oracle Recall@0.5=38.62%；真实合并/截断后的SAM显式支持在跨视图GT上覆盖30.89%。两者分母不同，不能直接相减当筛选损失；learned fallback没有显式SAM支持。2D初始query投影到最后层Recall约63.01%→67.48%，这是独立2D探针，非官方训练输出，不可与最终GS读出直接逐阶段相减作因果解释。

## 5. Ownership存在结构，尚不能判作主要错误来源

0.05组L12：Anchor–Child空间JS=0.534，打乱query对应后0.761；Read–Child=0.714，打乱后0.843；route–Child=0.614，打乱后0.773，均比打乱更一致。L12 Anchor–Child JS与Raw IoU Pearson约-0.362，Read约-0.287；是中等相关，不是方向冲突或错误写回的因果证明。大小/可见性等因素尚未控制。

反事实读出：只继承Anchor Membership渲染的Recall25.00%，几何Child单支24.80%，最终融合语义图像提升特征后的Raw78.25%。前两条是非官方完整融合读出的消融，不应称“Anchor到Child严重劣化”；完整融合在这里帮助了Mask形成。

Write=0。记录的route是潜在路由分布，不是实际写回贡献，不能诊断为跨实例写坏。Read允许全局上下文，不能强迫其与局部实例Membership完全相同。

## 6. 典型图例

每图两行为两个context，五列依次为GT、最佳SAM显式支持、最佳Raw query Mask、同query资格筛选后、同query最终Mask；红色覆盖。初始SAM支持与Raw query按各自最佳匹配选择，不保证二者是同一query来源；后三列保持同query。

- scene0633_01，GT5，可见92999像素：Raw IoU0.976，最佳query80判no-object；筛选后所有候选best IoU只剩0.0042，最终0.0042。图：`/space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1/attempt02/full/analysis/case_good_mask_rejected_no_object.png`。
- scene0342_00，GT18，614像素的遮挡小可见实例：Raw IoU0.850，Final IoU0。图：`/space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1/attempt02/full/analysis/case_small_visible_instance_rejected.png`。

自动top-loss可能优先选中1像素GT，汇报时上述案例比极小标注碎片更有代表性。

## 7. 建议

不优先重做Encoder，也不直接加更多交互层或开启Write。等现有V3对照判定初始化差异后，优先验证**Mask支持的分类读回**：让Object Query分类/no-object同时读取其预测Mask支持区域的语义证据，使实例选择与实际Mask对齐，并检查SAM槽的GT匹配、no-object率和小实例覆盖是否改善；初次保持保护重建的梯度策略。

这支持“用共同实例支持连接Mask与分类”的方向；尚不足以证明应让Read/Write/Child完全统一或采用强制双向写回。当前没有提交新训练或补推理。

## 产物

`/space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1/attempt02/full/`：原始summary.json/report.md、instance_stage_audit.json、golden_output_audit.json、每rank128场景记录与图。新增CPU审计脚本`analyze_gfm_instance_results.py`和`audit_gfm_diagnostic_golden_outputs.py`可复核，未运行新模型推理。
