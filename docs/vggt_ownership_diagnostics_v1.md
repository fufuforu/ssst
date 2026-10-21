# 固定权重 Object–Anchor–Gaussian 瓶颈诊断

> **已完成**：64963 COMPLETED 0:0，两模型各128窗口、相同492个GT实例、0训练更新。结果见 [`vggt_ownership_results_20261015.md`](vggt_ownership_results_20261015.md)。原64911的一像素复现失败保留，恢复复用220窗口、仅补36窗口；不要重新执行准备器或launcher。用户要求提交作业后停止轮询，等其通知完成再核验。

用户授权：完整C与C继续联合300更新的baseline，在同一128场景各一窗口上固定权重推理，定位ownership/实例质量/联合训练漂移；无参数更新、无模型结构修改、无官方评测协议变化。

## 模型和数据先核实

C由完整adapter0.01/feedback0.05底座（SHA ecd5cefabf9fb1b637d2c27e4bc65e137e88ed70dd9f9f650e944674f1e24ac0）加原C分类对象支撑补丁（SHA ad3100ceabbd4d9ab41f653cdb40ec1cb86dc882d71d9ad5ac14192947f4f0f9）构成。
baseline300为reading_controls attempt04 baseline的终点patch（SHA ca787097fe93d45776446ec511b96c6139a3449de83e9152d7c1cd654fba7ff2）。
准备器实际核验三个权重文件SHA、C/基线起点关联、128输入hash/frame IDs、RGB/depth/segmentation GT文件hash、训练数学身份和官方commit/src。
两模型生成时钟35776/38176均已成熟，feedback峰值都0.05；没有重新启动warmup，本轮均无Q。
GT只在完整原生预测结束后用于诊断、关联和病例选择。生成仍只用2个context，官方target标定/6帧渲染保持原样。

每模型每窗口均与其已有官方输出复现：预测与GT segmentation PNG逐位相同、候选ID/类别相同、分数误差≤1e-6，RGB量化差≤1。
原生CUDA读回有历史实测微小atomic波动，因此rank0首窗口增加两次无观测原生前向和一次有观测前向见证：Gaussian必须差0、membership/logits差≤1e-5。
每窗口检查参数版本、buffers、requires_grad、grad，始终eval/frozen；不构造优化器或训练探针。
推理使用空闲3090，worker数只影响独立分片，不是训练global batch。先申请3张允许11/13、排除12，复用已有bootstrap/离线资产。

## Ownership定义及负对照

- Read：L6/8/10/12真实8头object→anchor attention，保留headwise熵；空间比较用heads平均，但不把其与真实V/O输出混淆。
- Write：真实anchor→102 query＋null路由，及经过W_inject投影、tanh、norm cap和0.05幅度后的实际feedback。
- 实际贡献分解：`delta[a,d]=gate[a,d] sum_q route[a,q] projected_query[q,d]`，在观测到的公共非线性gate下，逐query向量和严格重建实际delta；同时记录贡献norm和与总delta的有符号投影以区分抵消。
  这不是删除query的因果反事实：gate固定在观测值，不能将unsigned norm称为实际造成损伤的大小。
- Anchor：每层独立sigmoid anchor membership；Child Gaussian：最终65536个子Gaussian membership，按真实1024 parent-major×64 children关系映回parent。
  主结果使用context renderer adjoint visibility mass加权，uniform child pooling作为控制；不用最近邻伪造parent关系。
- Read/Write/实际贡献/Child统一比较query在1024 parent anchors上的空间分布（JS、support enrichment），Anchor/Child比较visibility加权soft Dice。
  sigmoid membership不是互斥概率，不将其直接冒充Write softmax；query间stuff/null统计另记。Read可以读全局上下文，不以Read=Write=Child为正确标准。
- 每窗口16次query对应关系无固定点打乱，保留各自空间分布；另做独立anchor排列以保留query身份/总量但破坏空间对应。
  关注真实对应相比负对照的增益及与错误的关系，不把非零差异本身当缺陷。
- GT错误写回：预测完成后将context GT通过同一renderer的adjoint映到最终child，再汇聚到parent，以GT可见质量测量query贡献流向自身/其他GT的比例；另限于纯度≥0.8的parent复核。
  reference是最终parent/child可见GT支撑，早层anchor可能仍在移动，不是早层点位拥有真值的因果证明。未匹配但raw mask很好且对应同GT的重复query，不自动视为跨实例错误。

## 质量损失定位

复用`raw_stages`和原`assemble_panoptic`：raw sigmoid mask→类别/no-object资格→像素竞争→原面积比例过滤后的panoptic。
保存每个query对全部context GT的IoU矩阵、资格、类别、mask面积及原生最终Hungarian映射；同GT独立best-query作为补充，不能把query被过滤直接等同于GT漏检。
同一最终geometry/官方相机下额外渲染继承的anchor membership和geometry child feature membership，再与原融合后child membership渲染比较；这些是只读feature-channel诊断渲染，不替换正式输出。
分组：好mask且分类正确、好mask但no-object、好mask错误类别、差mask、GT小物体（context面积<1%，另列<0.1%）、同GT多个raw好候选。

## 漂移及性能关联

跨模型按`scene/window + GT instance ID/class`对齐，分别报告原生matcher query和类别无关best-query，不假定相同query号或相同anchor编号跨模型代表同一物体。
官方13列、官方全局AP/PQ复用已核实相同输出的既有结果；补充每窗口官方AP/PQ只用于ownership与性能的关联，不用其平均冒充全局AP/PQ。
RGB以同GT划分互斥object interior / 2px boundary / background stuff / other，记录SSE和像素数及逐帧masked PSNR；区域诊断不改变官方全图PSNR。
区域PSNR固定data_range=1，官方torchmetrics按GT推断range，绝对值不等同；相同GT的C/Baseline配对差消除共同range项。补充脚本已用原官方render_scores逐窗口核验配对差≤1e-5 dB。
另按GT实例记录RGB误差，以区分mask/类别漂移与物体重建变化。单条训练轨迹的端点对比和相关性不能独立证明某个交互造成表示退化。

输出保留小型JSON/CSV、可审计压缩ownership映射和raw masks；最终报告聚焦量化证据、4–6个典型案例、主要瓶颈判断及最值得检验的一个结构方向。
ownership可视化数组可无损于预测地以FP16归档，正式统计在FP32计算；模型/权重/训练数学及官方输出精度不变。

入口：`prepare_vggt_ownership_v1.py`（可比性核验/固定快照提交）、`diagnose_vggt_ownership_v1.py`（无更新推理）、`vggt_ownership_math_v1.py`（纯观测数学）。
证据root：`/space/mawb/ssst/group_plus/vggt_ownership_diagnostics_v1/attempts/attempt01`；任何失败保留并只修复受影响阶段。
