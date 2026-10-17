# Unified Superpoint Query V1：实现与边界

用户明确要求按“三维超点初始化一组Unified Query，共享解码，同时读出重建和理解”实现。本轮为独立新模型与训练/导出入口，未提交作业、未迁移旧模型、不恢复已取消的B、不改变66353快照。

## Pipeline与已经做出的选择

```
两张context RGB
  -> 冻结官方VGGT-Omega：相机、深度/点云、4层GFM
  -> 共享GFM投影：2048 -> 256
  -> 按源像素将特征提升到预测三维点
  -> 三维体素融合跨视图观测
  -> 2层局部三维邻域编码（XYZ相对位置＋邻居特征差）
  -> 三维超点划分
  -> 超点内几何感知池化 -> 投影＋中心/尺度编码
  -> 一组Unified Queries（没有learned query/fallback表）
  -> 12层共享Decoder：读GFM memory＋Query self-attention
       -> 重建Adapter／几何更新／Gaussian头，每Query生成64个Child
       -> 理解Adapter／18类＋no-object头／跨所有Gaussian的实例Mask头
  -> RGB/深度渲染；实例匹配、像素竞争及Panoptic输出
```

**初始化池化的是已经经过三维融合和邻域编码的点特征。** 不是之前的“用几何预测权重池化二维语义图”，也不是B的“SAM区域插值旧Anchor”。没有额外重建token表或Object Query decoder；同一条Query序列产生两种任务输出。stuff是从共享Query汇聚产生的两个输出描述子，不是另一组可学习query。

三维超点目前选用**几何体素过分割基线**，不是ESAM的数据集预处理器/稀疏卷积骨干的完整复现。用户此前未指定超点生成算法，这里明确选择一个不依赖SAM、可确定执行的实现，并预留外部超点ID入口。它可能在实例边界混合，不能预先称为高质量超点。

## 三维特征与超点

- 从每视图48×48格内选有限预测点中confidence最高的源像素（原图256）。这是覆盖采样，不是使用全部131072个像素点。原始GFM投影后按实际源像素双线性采样，带有源视图/像素索引。
- 在共同预测坐标系内按0.01×scene scale起始体素融合观测，位置与特征取均值；超过4096个融合点时增大体素，直到预算内，不通过全局top-k丢弃已选观测。保留source→fused-point映射。
- 三维点编码器：12近邻，两层局部MLP，输入中心特征、邻居特征差和归一化XYZ差，max聚合＋残差归一化。所有离散几何/邻居关系detach，特征及MLP保持梯度。
- 超点起始体素边长0.08×scene scale；超过512个时自适应放大，得到完整覆盖、互不重叠的三维局部区域。Query数为实际非空超点数，**不是固定100**，没有80＋20规则或随机补槽。
- 超点中心/RMS半径由区域点计算，半径下限0.02×scene scale。局部XYZ编码得到形状向量和`2*sigmoid`点权重，池化`mean(weight*3D_point_feature)+max(local_shape)`，再做内容投影及位置/尺度编码。
- `model.encode_context(rgb)`给出`omega, cloud, memory`；可在`cloud['points']`上运行外部3D过分割，然后传入`model.core(cloud,memory,superpoint_ids)`或`model.generate(rgb,superpoint_ids=...)`。外部ID必须对应**融合后的点**，是非负整数，不得来自GT实例标签；超过预算会报错，不静默截断。

这是ESAM启发的“点特征→超点池化→query”，不是使用ESAM权重，也没有将几何体素划分冒称为SAM/ESAM分割输出。

## 两个任务头及监督

- 默认共享宽256、8头、12层，读取两视图32×32的GFM memory。几何头逐层更新中心/半径，零初始化更新；重建读出沿用现有LocusGS Gaussian activation。
- 每超点Query生成64个Gaussian。默认最多512 Query/32768 Gaussian，低于旧1024×64的上限；这不是同容量单变量对照，不得直接将性能变化唯一归因共享query。
- Gaussian父Query身份只表示**由谁生成**。实例Mask为每个Gaussian对每个Query独立预测的sigmoid关系，允许一个Query的Mask覆盖多个超点生成的Gaussian。没有把所有Child硬归到父Query实例。
- 类别为每Query18类＋no-object；stuff两个Mask描述子从共享Query汇聚生成。重建包括背景超点，不因理解头判no-object而删除其Gaussian。
- 默认匹配开启来源约束：预测完成后才读取context GT，根据源像素的GT实例统计每超点的所属关系，纯度≥0.5才允许相应query/GT匹配；在允许关系内执行一对一最小成本匹配。没有足够支持的GT不强行匹配，记录`unsupported_GT_count`。可显式关闭来源约束做全局匹配对照。
- 这是本项目的来源约束Hungarian实现，**并非ESAM SparseMatcher的逐行复刻**。训练GT不进入超点划分、三维特征或query初始化，也不用于推理。
- 分类/no-object权重、Mask BCE/Dice、stuff项沿用原系数形式，但查询数和允许匹配集合改变，是新的模型/训练定义；官方推理阈值、竞争及area-filter规则保持。
- 两任务损失都可更新共享投影、三维编码、超点池化和共享decoder，CPU梯度测试分别验证了两条路径。任务头分组表示模块职责，不表示渲染几何梯度已隔离：理解Mask依赖Gaussian位置/opacity时也可向重建头反传。

## 代码入口

- `tokengs/models/unified_superpoints.py`：配置、像素特征提升、三维融合、邻域编码、超点池化。
- `tokengs/models/unified_query_core.py`：唯一Query序列、共享decoder与两任务读出。
- `tokengs/models/unified_query_v1.py`：FrozenOmega、context-only生成、相机标定、渲染与step_loss。
- `tokengs/models/unified_query_loss.py`：动态数量实例匹配、来源约束及损失。
- 注册名`siu3r_unified_query_v1`，选项`train_siu3r_unified_query_v1`。
- `scripts/train_unified_query_v1.py`：独立训练入口，支持单卡或torchrun的1/2/4/8 rank、每rank单场景、固定全局批8累积，原manifest/heldout SHA、同确定性epoch顺序、AdamW、LR warmup/cosine、clip1、断点及各rank RNG。
- `scripts/export_unified_query_checkpoint.py`：从本模型checkpoint固定权重导出，复用原128场景RGB身份/GT PNG逐位检查和官方格式。`--smoke`只导出1场景，不能当128结果。
- 现有`export_object_locus_v3_set_official.py`仅将100/102泛化为N/N+2，保留所有原阈值与竞争规则，增加N<1000的packed ID约束。新旧100-query固定示例及旧诊断回归通过。

核心使用：

```python
from tokengs.options import config_defaults
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.unified_superpoints import UnifiedQueryConfig

model = LocusGSUnifiedQueryV1(
    config_defaults['train_siu3r_unified_query_v1'],
    config=UnifiedQueryConfig(),
).cuda()
prediction = model.generate(context_rgb)  # [1,2,3,256,256]，仅context
output, metrics = model.step_loss(batch, understanding_weight=chosen_weight)
```

训练入口必须显式提供`--understanding-weight`，没有替用户决定最终联合训练权重，更不会默默套用旧双流隔离策略。`--understanding-warmup`默认1043，`--epochs`默认4；`--max-updates`是工程运行边界，后续同配置`--resume`可提高边界，不回放已完成更新。新模型拒绝加载旧V4/B的参数或Adam。旧训练脚本不接受这个架构，应使用独立入口。

## 验证与当前状态

12项CPU测试通过：完整点覆盖与融合梯度、三维邻域影响、外部超点ID、池化/3D编码梯度、单一query双头、两任务共享层梯度、来源约束与无支持/空GT、可变N及旧100-query组装、冻结骨干接口、生成/目标视图渲染、模型与非冻结checkpoint严格往返、step_loss两任务编排；同时包含旧实例诊断回归。测试使用合成Omega和可微假渲染器，真实Gaussian activation与共享网络实际执行；不声称验证了原生CUDA rasterizer、真实Omega推理或多卡训练。

另用提交前原始100-query组装函数对5个固定随机/结构混合输入核对，5/5输出逐位一致。py_compile与diff --check通过。尚无GPU冒烟、正式训练或完整评测结果；本轮没有提交任何作业。新的结构参数与预算需要在真实无更新前向/反向短验证后再确定训练配方。旧B保持取消，66353保持原样，不自动恢复或迁移它们。
