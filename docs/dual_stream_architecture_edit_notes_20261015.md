# 几何–语义双流：可编辑架构图改版

用户最新明确：**不是直接复用QuerySplat**。两组状态分别承担几何驱动的Gaussian重建、实例/语义驱动的对象理解；原来“直接切换官方QuerySplat底座”的设计已被此方向更新。

原文件保持不变：`/space/mawb/ssst/docs/QuerySplat_可编辑架构图.pptx`。

新文件：[几何语义双流_可编辑架构图_v1.pptx](/space/mawb/ssst/docs/几何语义双流_可编辑架构图_v1.pptx)。
PDF：[几何语义双流_可编辑架构图_v1.pdf](/space/mawb/ssst/docs/几何语义双流_可编辑架构图_v1.pdf)。

单页预览：[双流总图](/space/mawb/ssst/docs/几何语义双流_预览-1.png)、[差异对照](/space/mawb/ssst/docs/几何语义双流_预览-2.png)、[原图参考](/space/mawb/ssst/docs/几何语义双流_预览-3.png)。

三页内容：双流总图、方法差异对照、原QuerySplat可编辑图参考。文字、模块、token图标、箭头均为原生PowerPoint形状，输入图片沿用原文件；不是把整页PNG铺进PPT。参考页原图整体等比例缩放，未篡改其标签作为新方法。新设计没有使用原QuerySplat重建效果图来冒充本模型结果。

## 原图应改的五处

1. Geometry Query改为点云初始化的Reconstruction Tokens。VGGT点云采样、位置/内容编码进入初始化；重建token读取TokenGS/LocusGS图像编码器的RGB＋预测Plücker特征。相机、点云、Gaussian使用一致坐标系。
2. 上层重建输出包含位置/尺度/旋转、颜色、opacity全部参数。原图“几何输出一半、外观输出另一半、再合成同一个Gaussian”的职责分工不再保留。
3. 原Appearance Query分支换成Object/Understanding Tokens，实例候选跨视图关联与Mask语义特征池化形成初始化，同时预留可学习槽位。SAM/其他候选模型尚未锁定，图中虚线标注待定。
4. 下层K/V来自独立的语义视觉Memory和当前重建token。语义Memory暂按VGGT视觉/多层特征＋新适配器表达，来源仍需验证；理解token间Self-Attention不能替代图像证据。图中两组token数量不要求相同，不做父token与实例一一配对。
5. 删除原“Geometry Token相加初始化Appearance Query”的单向条件化连线。新图明确Read重建token提供K/V，细粒度视觉支撑参与逐Child描述子，实例membership/class独立输出；Write是归属约束和门控的**可选下一阶段**路径，不声称已验证收益，也不使用最终Mask倒灌过去层。

## 相似性应如何表述

两者都使用Query读取Memory、Self-Attention/FFN更新状态，外观相似是正常的，属于共用的Transformer解码范式。不能通过改名或额外画几个模块就宣称方法原创。

当前可讨论的实质差异是：重建与对象理解的任务分工、几何/实例初始化、跨视图对象身份、逐Gaussian实例归属，以及归属约束的非对称交互。仅替换点云/SAM初始化仍可能只是增量；贡献要由受控实验支持。建议先比较相同重建底座、候选预算与监督下的可学习Query/SAM引导初始化，再独立开关Write；不把所有变化叠加后归因于某一个机制。

状态仍是架构设计，未实现、未训练、未获得新指标；SAM对旧C那85个失败小物体的回补率也尚未实测。

总图表示交替阶段：当前阶段的Gaussian/归属读出用于下一个阶段的可选写回；图外侧head可在阶段间调用，不借用最终预测回填过去层。具体交互层数与实现仍需锁定。

生成入口：`scripts/build_dual_stream_architecture_pptx_v1.py`；使用标准库OOXML，不依赖python-pptx。源文件SHA和原生形状数量记录在输出同名JSON中。PDF采用LibreOffice真实渲染核对，不能只凭XML文本声称排版正确。
