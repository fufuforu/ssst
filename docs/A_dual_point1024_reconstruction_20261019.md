# A 双分支：恢复点云采样1024个重建token

用户指定基线为 A full4：`attempt05_legacy_dual_superpoint_full_node13_8gpu`，
PSNR18.5003、AbsRel0.18980、RMSE0.39511、Target PQ62.7599%。
这是 A 双分支，不是原 Unified、evidence 或超点细化容量组。

新增独立注册键 `dual_point1024`，架构名
`OMEGA_POINT1024_RECON_SUPERPOINT_UNDERSTANDING_DUAL_V1`。
入口为 `scripts/train_unified_query_v1.py --architecture dual_point1024`；
现有统一导出入口从 MODELS 自动识别新架构，不改官方评分或后处理。

重建初始化恢复 `LocusGSGFMDualStreamV2.generate`（V3/V4继承）的原路径：

- 仅使用两张context的冻结Omega点云、置信度、RGB和GFM，不读取GT。
- 原 `seed_anchors`：分视图图像网格覆盖、置信度筛选、确定性3D FPS，1024锚点；
  使用原第六近邻半径及退化点保护，点不足时沿用原实现的重复补齐。
- token = 学习的1024×1024基础token + 几何/RGB的7→1024内容MLP
  + 精确采样来源对应的GFM特征LN/2048→1024投影。
- 每锚点64个GS，总数65536；不沿用原A重建侧超点编码器及256→1024投影。

理解分支保持 A 的融合点云、SAM超点身份、独立邻域编码和动态query数量。
沿用 A 的1024/256双分支宽度、12层重建、L6/L8/L10/L12交互、Write=0、
联合梯度、几何更新及 `superpoint_scene_normalization`、Gaussian特征/理解query相似度Mask读出。
匹配、损失、渲染、NMS及正式指标不改。

1024是重建token数，不是理解query数，也不是token维度变更。
新输出以 `gaussian_parent_anchor` 记录GS到重建锚点的父关系，
不提供错误的一对一 `gaussian_parent_query`；Mask仍由理解分支计算。
输出额外保留初始锚点位置、半径及原像素来源供审计。

原 A core 仅提取初始化和父索引两个可重载方法，原A默认运算及参数键不变。
新架构删除被替换的重建初始化参数，新增基础token/内容MLP/GFM投影；
不是与原A逐张量完全一致的模型，不能直接严格加载A已训练checkpoint。
本次仅实现结构；未移植已训练权重，未生成新训练快照或提交训练。
已有67940、已取消67931及旧实验资产均未修改。

验证：20项CPU检查通过（26.766秒，不含环境导入耗时），包含新增3项及原结构、
点云初始化和GFM组件17项回归。新增检查覆盖与原点云初始化公式逐值相同、
采样像素来源及新模块梯度、共享模块键/形状与参数分组，以及真实12层解码器
在1024重建锚点/2理解query下前向输出65536个GS和4通道membership。
Python编译与 `git diff --check` 通过。

验证命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /space/mawb/anaconda3/envs/tokengs/bin/python -m unittest \
  tests.test_superpoint_dual_point_reconstruction tests.test_superpoint_structural_variants \
  tests.test_dual_stream_initialization tests.test_gfm_dual_stream_v2 -v
```

CPU使用合成Omega/SAM与替代renderer，不代表真实冻结编码、GPU渲染、完整反传或
多卡同步通过。新增模块有梯度；真实GPU的65536 GS显存、吞吐及训练稳定性待验证。
