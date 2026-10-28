# Baseline8冻结质量头A8/C8：13服务器并行实验

## 目标与授权

用户授权在空闲13服务器执行建议的Baseline8＋Token-only/Token+GS对照，优先看mIoU和mAP。独立于12服务器71621–71625的Baseline4类别感知质量头链，不修改或迁移该链，不合并两项改动。

本轮只训练A8/C8两个小头，不重训Baseline、不增加GS-only/打乱GS/类别感知/软语义损失或联合训练组。最终输出五个部署案例：Native、A_Rank、C_Rank、A_Cal、C_Cal。Rank仅替换实例JSON分数；Cal同时替换分数并用固定gamma=.25调整像素竞争，保持候选、类别argmax、stuff、面积过滤不变。

## 来源与公平比较

原Baseline8完成源：semantic_resolution_v1/v4init_isolated_extend8_recovery_20261023_v2_jsonfix/Baseline/eval/source_checkpoint.pt。
SHA bd1afe7e27ca01533ff5bfef6cd218a99c4f2294de08399a40cd2acf6eb98bc5；进度epoch8/update0/full8344，原632张量、604 Adam状态。冻结主模型、参数更新0，源原件及完成收据不改。不是semantic_quality_extend8模型，不带新质量头。

使用原GS实验partial_1406渲染库SHA82403af8b36fb4b64774afc7462ea6611896fcfe7a02785098bce9ff50d8d751。该库已在13完成过Baseline16完整前向诊断；新任务仍将核验加载路径/SHA并记录实际运行环境，不用不兼容的v4库冒充原库。

复用原数据清单和场景划分，不复用Baseline4的Token、GS、Query或IoU标签。原8337训练窗口＋1860正式窗口由Baseline8重新生成，Token、GS32、候选ID、Native mask和Context IoU来自同一次generate。每窗输入RGB SHA及6项前向签名绑定；正式六视角的float32输出精确缓存，A8/C8导出共用该state，不重新generate。

理解与GS特征函数、网络定义、标签helper、panoptic装配、模型代码与原成功实验相同；只有源模型改为Baseline8、两头范围和8卡执行配置变化。CPU AST核对GS32/网络函数未变，原模型及标签文件Git差异为空。

旧Baseline4路径只读用于相同输入/GT清单的配对核验及历史结果比较，不用于绑定新Query。训练缓存以global cohort index区分合法相同Context/不同Novel的训练窗口；仅正式1860个导出name要求唯一。

## 两头训练与选择

场景1072训练/119开发/312正式，开发场景对小头拟合留出，主干已见；不能称整模型独立验证。GS标准化仅从本轮Baseline8训练分区拟合mean/std，clamp std>=1e-4、标准化后clip[-10,10]。

A8=Token256→LN→Linear256→GELU→Linear1→Sigmoid（66561参数）；C8=LN(Token256)拼接GS32→Linear256→GELU→Linear1→Sigmoid（74753参数）。全部输入detach；两头各seed31415、AdamW lr.001/WD.0001、batch1024、clip1、最多100轮、patience12，按开发集MSE最低选检查点。

保持成功Baseline4 GS实验的class-agnostic最终Context二值mask最大IoU目标与MSE，不加类别感知或像素项。正式GT不用于选模型；正式预测先冻结，再做质量回归/一致性事后统计。不扫描gamma或按正式指标挑epoch。

## 指标和归因边界

正式1860窗/312scene，target=all6含Context，与总表一致。Native实测五重建指标；其余四分支RGB/depth共用，分割mIoU/PQ/AP/AP50/AP75/AR全部使用未改动SIU3R评分器8ea80166be76854f938e90521f1a5b688b755c87。

主要比较C8−A8（同前向GS补充作用），其次看C8_Cal相对历史C4_Cal的最终成绩，并保留Native8与旧Baseline8的跨运行差异。不能把换检查点的总收益全记为GS收益，也不能将没有新增打乱GS8组的对照解读为已完全排除容量效应。Baseline4已有打乱GS证据是另一项实验结果。

## 提交与交接

独立代码：/space/mawb/ssst_frozen_baseline8_gs_quality_code_20261024_v1。
独立run：/space/mawb/ssst/group_plus/semantic_resolution_v1/frozen_baseline8_gs_quality_20261024_v1。
13服务器afterok串联：8卡新缓存→CPU两头拟合→8卡五案例导出→5卡五案例并行评分及汇总。评分各GPU独立处理完整1860窗口，不平均分片AP；模型/缓存都不与12任务共享写入。
粗估不含排队约2–4小时；原GS四卡10197窗缓存191分钟，本轮八卡，仍受GS特征CPU计算和大文件I/O限制。GPU加载/同步/首个窗口完成尚未由CPU预检证明。
提交后只查一次接收状态，保存作业号、资源和依赖后停止轮询，不等待首步、指标或完成。用户通知后再检查cache/FROZEN_rank00..07、cache/export_complete、两头FIT_COMPLETE、五输出配对、RESULTS及COMPLETE；不因旧PENDING重复提交。
