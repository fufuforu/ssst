# Omega 点云 + SAM 对象双流 v1 实现

新增独立模型 `siu3r_object_locus_dual_stream_v1`，配置 `train_siu3r_object_locus_dual_stream_v1`。原模型、权重和历史评测结果保留。此版本不是直接复用 QuerySplat 的 Geometry/Appearance 两组 token：重建 token 输出全部 Gaussian 参数，另一组负责对象理解。

## 实际入口

```python
from pathlib import Path
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(Path.cwd())  # 本集群 renderer 扩展装载，必须先运行
from tokengs.options import config_defaults
from tokengs.models import model_registry
opt = config_defaults['train_siu3r_object_locus_dual_stream_v1']
model = model_registry[opt.model_type](opt).cuda()
output = model.generate(context_rgb)  # [B,2,3,256,256]，范围[0,1]，不接收 GT
```

`generate` 输出沿用现有 gaussians / reconstruction / region_mass / thing_logits19 / gaussian_membership / states 字段，另记录初始 anchor/query 几何及 SAM 跨视图合并数量。`step_loss` 复用既有重建、多层实例损失，先独立标定全窗口相机；本轮不运行优化器。

## VGGT 点云如何初始化 anchor

1. 冻结官方 Omega，一次 512×512 aggregator 前向同时给出 `[4,11,17,23]` 层特征、预测相机、深度、置信度，避免重复 backbone 计算。严格匹配官方 checkpoint 三个子树并核对 SHA256。
2. 根据预测 K 和 depth 反投影。采用本项目 pixel edge-origin / 像素中心 `x+.5,y+.5` 约定。所有相机、点和深度统一至第一张 context 相机坐标；场景中位深度归一到 .25，保持原 LocusGS 数值尺度。K 从 512 缩放至 256，不能沿用旧标准 VGGT 的 518 比例。
3. 每个视图独立保留置信度最高的约80%有效点，在48×48网格的每格选最高置信度点，再对合并候选做确定性 3D FPS，获得1024个 anchor。不是全图 top-k，也不使用 GT 点云。有效候选不足时重复已有有效位置；全无效直接报错，不制造伪点。
4. `mu` 为选中点坐标；支持半径取第六近邻距离，夹至场景 RMS 尺度的 `.01~.3`。由 `rho=inverse_softplus(radius-epsilon)` 接入原 decoder，关闭继承配置中会抹掉这些半径的 `locusgs_freeze_decode_radius`。
5. 内容向量为原可学习 GS token 加 `MLP([归一化XYZ,归一化半径,该点RGB])`，维度7→1024→1024。位置、半径、内容均随场景变化。后续位置/半径通过原 decoder 学习细化，几何采样本身不反传给冻结 Omega。
6. 重建 K/V 来自恢复的原 TokenGS/LocusGS RGB＋预测 Plücker 图像编码器，而不是旧模型的 VGGTMemoryAdapter。该编码器结构复用，但此版本不自动迁移旧重建 checkpoint。

## SAM 如何初始化 object query

- 实际使用冻结官方 **SAM ViT-B**，自动网格提示，`points_per_side=32,crop_n_layers=1`，不使用 GT box/point。根目录常用资产位置未找到可用 SAM 权重后下载官方文件；这不是已经验证对85个失败小物体最优的 SAM 版本。
- 每个 context 生成候选，按预测质量×稳定性排序，最多保留128个/视图。将 Mask 映射到128×128语义网格；同视图 IoU>.8 去重。
- 语义 Memory 为 Omega 四层独立投影并加权融合，加一个随机初始化的高分辨率 RGB 卷积残差，再经卷积细化到256通道。不加载 MASt3R、CroCoViTAdapter、COCO Mask2Former。冻结视觉特征不等于已有完整理解能力；新的 adapter/head 仍需训练。
- 对每个 Mask，在当前语义 Memory 中池化内容；从对应预测点云估计3D中心、尺度和最多128点支持集。不同视图候选仅在双向最近邻支持重合>.5且池化特征余弦>.7时合并；支持距离阈值为场景尺度的.05。单视图候选允许保留，不因没有第二视图匹配而丢弃。
- 最多80个候选 query：`LN(mask_pool(F_sem)+MLP(归一化中心/尺度))`；剩余槽位使用可学习 query，至少预留20个，总数仍100。另有2个 stuff query，沿用原控制器。中心/尺度直接来自候选支持，不再先把旧 q_pre 与 anchor 做相似度匹配来初始化。
- SAM 的 latent token 不直接拷贝为 query；SAM 只提供区域和自动候选，query 的特征基底来自本模型语义 Memory。
- 关联阈值、候选排序、80/20比例是 v1 可检验假设；位于相近表面的实例仍可能错合并，不声称已解决跨视图身份。

## 哪些仍和原模型相同

L6/L8/L10/L12 的 Anchor Read→Image Read→Query Self-Attention/FFN、Gaussian 参数 head、逐 Child 几何/索引描述子、遮挡感知视觉提升、Mask 渲染和实例分类继续复用。Read 仍可获取全局上下文。Write 默认严格为0；可选 `write_strength∈[0,1]` 只缩放原有带 null 路由的有界残差，**尚非新设计的统一 Ownership Write**，不把它包装成新贡献。

本轮的实质改动是几何/区域初始化、跨视图候选身份、语义 Memory 来源、重建/理解分工及恢复 RGB+ray 编码器。这些是否有益必须用相同预算的 learned-query / SAM-query 对照检验，不能凭代码或示意图断言。

## 相机标定与验证边界

Omega 全窗口相机通过共享 context 点云做置信度加权、五轮 Huber Sim(3) 对齐；输出 context 相机保持 generate 结果，生成的 Gaussian 不被标定过程修改。新网格256协议名 `omega_shared_context_depth_sim3_256_v1`，未伪称旧标准 VGGT 的518网格协议；没有改写旧官方评测入口或阈值。该新标定需真实数据验证后才能用于新模型正式报告。

CPU 6项检查已通过：重投影、点采样/无效输入、空 SAM 回退、跨视图合并/重复去除及梯度、远距离不合并、已知 Sim(3) 恢复。语法编译和 diff 检查通过。CPU合成 Mask 测试不构成真实 SAM 效果证据。

真实权重 GPU 冒烟脚本：`scripts/smoke_dual_stream_v1.py`；检查完整前向、输出有限性/尺寸、冻结 backbone、零 Write、独立标定不修改生成结果。产物目录 `/space/mawb/ssst/group_plus/dual_stream_v1_smoke/attempt01`。在 `COMPLETE.json` 出现前只能称代码实现和CPU检查完成，不能称GPU链路验证成功。无优化器、参数更新0；尚未测85个小物体回补率或新模型质量。

资产：Omega `/datasets2/mengxl/vggt_omega/vggt_omega_1b_512.pt`，SHA `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`；SAM `/space/mawb/ssst_dual_stream_assets/sam_vit_b_01ec64.pth`，SHA `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`。

Omega Python 源码复用本地 QuerySplat 第三方目录 `/space/mawb/querysplat/third_party/vggt_omega`，父仓 commit `3465a1d2c789d8ebe71f4f9c0bae3f2c2fd726ae`；新代码没有引入其 scripts 包，避免包名冲突。SAM 官方源码下载 `https://codeload.github.com/facebookresearch/segment-anything/zip/refs/heads/main`，zip SHA `c0641b7d42fa696c63aac06c8c0f24169da13ec18eef1f129770a99ee90f77cd`，已用原生 tokengs Python 安装。官方权重来源 `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`。外部权重未加入 git。
