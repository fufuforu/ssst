# 实例Mask两个主要问题：汇报材料

直接使用：[三页中文PDF](/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1/attempts/attempt01/presentation_v1/Mask瓶颈诊断_汇报图.pdf)。

原图为16:9、高分辨率PNG，可以分别插入PPT：

1. [两个问题的量化概览](/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1/attempts/attempt01/presentation_v1/01_两个主要问题_量化概览.png)
2. [Raw Mask形成不足：远处小物体与遮挡实例](/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1/attempts/attempt01/presentation_v1/02_RawMask形成不足_真实案例.png)
3. [好Mask被no-object删除／类别错误](/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1/attempts/attempt01/presentation_v1/03_好Mask筛选与分类失败_真实案例.png)

可按下面的文字讲解：

> 我们在完整C模型的128个固定留出场景上做冻结推理诊断。问题分为两个阶段：第一，部分实例在分类前就没有形成好Mask，492个实例中有134个没有任何IoU达到0.5的Raw候选，小物体占85个。第二，已有好Mask也不一定被正确识别、保留：有合格候选的GT从358个降到最终288个，主要损失发生在no-object资格筛选；还存在Mask准确但类别错误。右侧实际案例分别展示候选形成不足、好候选被删除、几何正确但分类错误。这些现象明确定位了失败环节，但还不能把根因唯一归于特征或Object–Anchor交互。

用户补充的训练流程口径应保持明确：**常规是先适配2 epoch，再联合训练4 epoch；本轮Baseline额外300步是对照。** 额外续训PSNR变化可以不放在主报告中；若放附录，应写“额外续训对照的端点变化”，不能写“常规2+4训练导致联合优化退化”。此前把它与前两个问题并列称为主要问题过重，此处修正。前两个问题均能在完整C上直接看到，不依赖额外300步结果。

图中统计和案例口径：

- 均使用C已有正式预测、GT与诊断归档，没有重新推理或训练。该C以完整2+4底座加已有分类读回补丁构成。
- 概览“有好候选”按每个GT、所有100个Query中最好几何IoU≥0.5计数，忽略类别；不是正式AP/PQ，也不是部署时用GT挑Query。358→294→288是各阶段覆盖，净减少70个，不是假定固定70个实例都只单向丢失。
- 第2页在所有Raw Query中事后取对该GT最好的候选，最终Mask也独立取最好，排除“只是选错Query”的解释。蓝色为漏掉的GT，绿色为TP，红色为FP，黄色为GT轮廓；青色方框标放大区域。图上IoU计算覆盖两输入帧，图片只展示其中一帧。
- 第3页追踪原生匹配Query。no-object案例同时检查其他Query，最终最好IoU也仅.007；分类错误案例绿色只说明几何正确，不代表语义正确。
- 两个Raw案例：scene0500_01/GT36，远处椅子，Raw最好IoU .426；scene0342_00/GT12，遮挡椅子，Raw最好IoU .215。图片为说明性病例，不能据两个病例声称已证明尺度/遮挡是唯一原因。
- 来源与逐例数值：[manifest.json](/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1/attempts/attempt01/presentation_v1/manifest.json)。

生成器：`scripts/present_vggt_mask_bottlenecks_v1.py`。用现有native Python的numpy/Pillow/matplotlib重绘，支持新的独立输出目录，不覆盖已完成实验。
