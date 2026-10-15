# 双流正常学习基线：短验证自动衔接全量首轮

> 后续覆盖：用户确认GFM双Adapter v2并要求18节点4卡。新结构、50步资产转换及新验证起点见 [gfm_dual_stream_v2_implementation_20261015.md](gfm_dual_stream_v2_implementation_20261015.md)。下文2卡/v1描述保留为历史首轮配置。

用户已同意：短验证通过后进入完整数据训练，先学出正常结果并尽快接近旧VGGT+MASt3R指标，再研究新结构优化。此授权覆盖本次短验证及条件满足后的全量训练；不重跑旧模型、不自动解冻Omega或启用Write。

## 固定方案

- 新模型 `LocusGSDualStreamV1`，冻结官方Omega和SAM；训练其余约2.30亿新初始化参数。Write=0，结构和现有实例/重建loss不改。
- 当前资源选择2张48GB A6000、单卡microbatch1、累积4，有效batch8；FP32训练分支，冻结Omega aggregator沿已验证BF16推理。
- AdamW betas(.9,.95)、eps1e-8、weight decay .05（bias/norm和原有no_weight_decay参数除外），梯度范数clip1。重建、语义adapter/query/实例head峰值LR1e-4；位置/半径refinement头LR1e-5。短验证warmup20更新，全量warmup100更新，余下cosine到峰值的.1。理解loss权重1，均在提交前固定。
- 短验证：从训练manifest按固定种子选16个互不相同场景，每窗口至少2个实例，优先覆盖类别；没有根据模型预测选数据。128更新检查，未通过可延长到256更新，总曝光1024或2048。
- 固定monitor：8个短验证训练场景＋8个不重叠的原固定heldout128场景。前后使用同一RGB/GT/相机标定流程，保存分项loss、匹配Mask IoU、类别正确率、context/额外视图PSNR及2组可视化。monitor均值是小队列诊断，不冒充官方mAP/PQ。
- 通过后从当前权重和AdamW状态继续，进入原完整训练manifest的1191场景/8337窗口，首轮4epoch。每epoch覆盖所有窗口，仅为batch8尾部补齐7个已记录窗口，1043更新/epoch，共4172更新、33376曝光；短验证时钟另记，不挤占全量epoch。不承诺随机初始化新分支仅4epoch即追平成熟旧C。

## 转全量条件（不能见总loss下降就通过）

`scripts/dual_stream_training_policy.py::acceptance` 同时要求：完整loss反向有限、重建/语义/query/panoptic代表参数确有更新；固定训练PSNR提升≥1dB、重建loss降低≥10%、理解loss降低≥5%；匹配Mask soft IoU提升≥.01、binary IoU均值≥.1；匹配类别正确率≥max(.2,初始+.05)，且正确分类并有非空Mask的匹配比例≥.1；独立验证PSNR相对初始不下降超过1dB且数值有限。

这是“确实学到重建和实例”的早期门槛，不是成熟模型质量验收。128更新不通过会按预设继续至256；仍不通过写GATE_FAILED并停止，绝不降低门槛自动开启全量。数值/反向/数据契约错误立即保存rank级FAILED与日志，不跳窗口或吞错。短验证128/256没有达到此门槛时需要检查具体失败原因，不等于直接认定架构不行。

## 保存、恢复与最终比较

每50更新保存可训练权重、AdamW、各rank RNG、phase/epoch/采样cursor与窗口曝光计数，另留初始、短验证通过、各epoch端点；冻结权重外置，checkpoint记录其SHA身份，不重复保存Omega/SAM。旧已完成作业数据不重跑。恢复必须显式`--resume`，校验代码、数据、冻结资产、配置和world size；从保存游标继续。

每250个全量更新和每个epoch做固定小monitor，保存曲线JSONL和示例PNG；全量结束时用原固定128场景、相同帧ID/RGB hash、原panoptic组装及阈值导出。所有GT PNG还需与旧C导出逐文件hash相同。随后使用固定SIU3R commit `8ea80166be76854f938e90521f1a5b688b755c87`及原官方环境评分，输出原13列（两处文本mIoU留空）及旧C差值表。target仍为2context＋4额外帧，不换为单独4帧评测。

深度导出沿旧PNG序列化除数.15以保持量化口径；这只是文件编码比例，不把Omega相机/点云乘.15或声称得到米。官方评测仍自行逐图拟合scale+shift；几何仍采用context中位深度.25坐标。

旧C同队列参考：PSNR22.5966dB，输入mAP .30799/PQ .64552，target mAP .27376/PQ .60979。它使用已适配且训练成熟的旧分支和分类readback，当前新模型不保证首轮即达到；最终比较同时看训练进展和输出是否正常。

## 缓存与验证

冻结SAM候选按实际RGB、资产SHA、SAM配置缓存，保存bit-packed masks；候选跨视图身份、可训练语义特征和query每次重算。冻结Omega仅缓存最近8组context的精确CPU输出；目标标定按所有实际RGB＋context预测点图缓存。缓存不使用GT，不改变前向数学；不把全量Omega特征持久化成数百GB资产。

4项新增CPU测试通过：完整epoch覆盖和不同rank数下有效batch顺序一致、仅loss下降无法过关、正常学习可过关/验证塌缩被拦截、冻结缓存逐值一致且输入变化失效。全量manifest与128评测场景无交集，真实CPU加载短训练首窗口与固定验证首窗口通过，验证RGB hash等于旧结果。上一轮真实前向作业65163已通过；本轮真实反向尚待训练作业执行验证。

入口：`scripts/train_dual_stream_v1.py`、`scripts/dual_stream_training_policy.py`、`scripts/dual_stream_frozen_cache.py`、`scripts/dual_stream_official_export.py`、`scripts/score_dual_stream_v1.py`。只提交一个包含短验证和条件化全量的Slurm作业，使用不可变代码快照。提交后确认启动即停止轮询，等用户通知再检查。
