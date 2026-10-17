# Unified Superpoint Query V1：实现与边界

用户明确要求按“三维超点初始化一组Unified Query，共享解码，同时读出重建和理解”实现。本轮为独立新模型与训练/导出入口，未提交作业、未迁移旧模型、不恢复已取消的B、不改变66353快照。

## Pipeline与已经做出的选择

```
两张context RGB
  -> 冻结官方VGGT-Omega：相机、深度/点云、4层GFM
  -> 共享GFM投影：2048 -> 256
  -> SAM自动Mask按面积降序覆盖为互斥分区，小区域覆盖大区域
  -> 按源像素将分区ID及特征提升到预测三维点
  -> 同分区内三维体素压缩（跨分区绝不融合）
  -> 2层局部三维邻域编码（XYZ相对位置＋邻居特征差）
  -> 按携带的超点ID分组
  -> 超点内几何感知池化 -> 投影＋中心/尺度编码
  -> 一组Unified Queries（没有learned query/fallback表）
  -> 12层共享Decoder：读GFM memory＋Query self-attention
       -> 重建Adapter／几何更新／Gaussian头，每Query生成64个Child
       -> 理解Adapter／18类＋no-object头／跨所有Gaussian的实例Mask头
  -> RGB/深度渲染；实例匹配、像素竞争及Panoptic输出
```

**初始化池化的是已经经过三维融合和邻域编码的点特征。** 不是之前的“用几何预测权重池化二维语义图”，也不是B的“SAM区域插值旧Anchor”。没有额外重建token表或Object Query decoder；同一条Query序列产生两种任务输出。stuff是从共享Query汇聚产生的两个输出描述子，不是另一组可学习query。

**最新用户指令已将默认生成器改为`esam_sam`。** 参考官方ESAM提交`188fc6de44f7577fecec2d69b75c7adbd9992251`的[多视图预处理脚本](https://github.com/xuxw98/ESAM/blob/188fc6de44f7577fecec2d69b75c7adbd9992251/data/scannet-mv/load_scannet_mv_data.py)：SAM区域按面积降序写入像素标签，小区域覆盖大区域；点携带标签，未覆盖点使用XYZRGB KMeans补充分组。旧几何体素方案仅保留为显式`superpoint_source="geometric_voxel"`消融，不再默认使用。

这是ESAM的超点生成流程在预测几何上的适配，**不是整个ESAM网络的复现**：原版实测深度/相机改为VGGT预测几何；原版ViT-H改用现有已校验ViT-B；原版20k点随机采样改为网格覆盖＋每区域至少一点＋同区域体素压缩；三维骨干仍是本项目邻域MLP，而非ESAM稀疏卷积Res16UNet。没有使用GT几何、GT实例或GT相机生成query。

## 三维特征与超点

- `FrozenESAMSAM`使用AMG默认参数（points_per_side32、crop_n_layers0），保留全部输出Mask，不使用旧SAM接口按置信度截断128的逻辑。重叠Mask转互斥像素分区；仅保留有有效预测几何的区域。
- 从每视图48×48格内选confidence最高的有效源像素，并为每个尚未选到的非空SAM区域额外保留一个有效像素，防止小区域被网格采样消除。要求深度正且有限、点/置信度有限；不照搬ESAM实测米制深度>0.1阈值，因为VGGT当前处于median-depth=.25的预测尺度。
- 未覆盖的**已选点**按ESAM MV规则补分组：少于8个时单组，否则XYZRGB六维KMeans8（RGB为0..255），n_init10；为复现固定random_state42。XYZ沿用VGGT归一化坐标，不宣称与ESAM米制几何的聚类距离完全相同。
- 不同视图的区域ID分别编号，不把跨视图区域擅自认作同一实例。共享预测坐标中的3D邻域编码可以交换跨视图信息，后续实例Mask可以覆盖多个区域的Gaussian。
- 仅在同分区内按0.01×scene scale起始体素压缩观测，超过4096融合点时增大体素；标签为体素键的一部分，绝不混合不同区域。保留source→point→superpoint映射。区域数超过512或点预算不能保留全部区域时显式报错，不静默截断或改回几何体素。
- 三维点编码器：12近邻，两层局部MLP，输入中心特征、邻居特征差和归一化XYZ差，max聚合＋残差归一化。所有离散几何/邻居关系detach，特征及MLP保持梯度。
- Query数为实际非空超点数，**不是固定100**，没有80＋20规则或随机补槽。
- 超点中心/RMS半径由区域点计算，半径下限0.02×scene scale。局部XYZ编码得到形状向量和`2*sigmoid`点权重，池化`mean(weight*3D_point_feature)+max(local_shape)`，再做内容投影及位置/尺度编码。
- `model.encode_context(rgb)`给出`omega, cloud, memory`；可在`cloud['points']`上运行外部3D过分割，然后传入`model.core(cloud,memory,superpoint_ids)`或`model.generate(rgb,superpoint_ids=...)`。外部ID必须对应**融合后的点**，是非负整数，不得来自GT实例标签；超过预算会报错，不静默截断。

这是SAM分区驱动的“3D点特征→超点池化→query”。SAM不直接池化二维GFM生成query；SAM只提供离散分组。

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

原版本12项CPU测试通过；本次新增ESAM路径检查，结果见下方最新记录。原覆盖：完整点覆盖与融合梯度、三维邻域影响、外部超点ID、池化/3D编码梯度、单一query双头、两任务共享层梯度、来源约束与无支持/空GT、可变N及旧100-query组装、冻结骨干接口、生成/目标视图渲染、模型与非冻结checkpoint严格往返、step_loss两任务编排；同时包含旧实例诊断回归。测试使用合成Omega和可微假渲染器，真实Gaussian activation与共享网络实际执行；不声称验证了原生CUDA rasterizer、真实Omega推理或多卡训练。

另用提交前原始100-query组装函数对5个固定随机/结构混合输入核对，5/5输出逐位一致。py_compile与diff --check通过。尚无GPU冒烟、正式训练或完整评测结果；本轮没有提交任何作业。新的结构参数与预算需要在真实无更新前向/反向短验证后再确定训练配方。旧B保持取消，66353保持原样，不自动恢复或迁移它们。


## ESAM分区适配记录（2026-10-17）

冻结Omega/SAM独立从官方资产加载，checkpoint不重复保存这两组冻结参数；训练metadata记录config、Omega及SAM身份，官方导出校验二者。旧体素版本配置需要显式指定`geometric_voxel`；新旧科学配置不得当作同一实验续训。

本轮未提交训练、未查询/改动已有作业。实际SAM自动分割、CUDA原生Gaussian渲染、真实数据上的区域数/预算分布尚待GPU短验证；CPU测试使用合成Omega/SAM/renderer，不能替代这些检查。没有承诺新初始化已经改善mAP或小物体召回。

验证：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest tests.test_unified_query_v1 tests.test_gfm_instance_diagnostics`共16项通过（5.506s测试执行，不含启动）；新增小覆盖大、网格遗漏小区域补采样、跨视图ID分离、同位置异区域不融合、区域预算拒绝、未覆盖KMeans确定性/标签dtype、无效深度排除、SAM冻结和checkpoint排除检查。`py_compile`、`git diff --check`通过。scikit-learn固定1.5.2并在当前tokengs环境实际安装。
