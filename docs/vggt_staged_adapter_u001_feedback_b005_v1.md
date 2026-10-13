# Adapter0.01 + feedback0.05：完整数据适配2/联合4

用户2026-10-13授权先启动该版本，再执行联合梯度尖峰诊断和已有A/B/C扩大验证集复评。
从已验证的适配epoch2边界重新执行冻结联合4 epoch，不续训旧联合epoch6。
边界SHA a50af40e1cf639514c2eb1d7b3127a79ecbc096d6d3a071a70d9a9fd2587750c；
适配VGGT资产SHA c1f6a8fe3865da3ed737d7dcb86cadbd5967dccc6c94b1483ad6c6b3f0767ace。

唯一研究改变：原点积/capped feedback乘0.5，上限0.05，原1000新增曝光warmup保留，训练与评测一致。
adapter/GS理解梯度系数0.01；VGGT全部冻结；MASt3R/ViT adapter/pixel decoder保持原联合训练范围。
不叠加视觉冻结、AA解冻、decoder重初始化、任务头/loss或几何改变。
继承四rank RNG、原采样epoch2至5、fresh联合AdamW及warmup，global batch8/micro1/accum2。
4172新增调用/33376新增曝光；加上复用适配阶段累计6258/50064。
主训练node13四张3090，保持上一轮硬件、数据与训练数学；geometry_quality_policy=monitor_v1。

复用原runtime，新增staged feedback参数与严格recipe身份；使用已验证的ScaledFeedback，不创建新参数。
CPU测试验证默认旧配置兼容、边界/采样/四rank平均/RNG、保存feedback与评测恢复一致；
单卡与四卡各真实两调用/阶段smoke，包含6923，smoke时钟不进入正式训练。
提交前push固定SHA；计算节点使用不可变快照；主训练20正式更新/160新增曝光验证后停止轮询。
依赖官方评测四卡，在11/13/14/17/18等待，1860窗口/312场景，13列，两处mIoUt为空。

训练root：`/space/mawb/ssst/workspace_group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1`。
证据root：`/space/mawb/ssst/group_plus/object_locus_vggt_recon_adapt_freeze_adapter_u001_feedback_b005_v1`。
进度与剩余时间：

```bash
/usr/bin/python3 /space/mawb/ssst_vggt_overfit_diagnostics_v1/scripts/show_object_locus_vggt_staged_progress.py --adapter-understanding-scale 0.01 --feedback-peak 0.05
```

后续任务分别使用已有A/B/C的保存权重，不修改主训练快照，不混入主训练。
扩大复评将包含未参与选beta的248场景；联合路径诊断保留原GC/统一clip1和AdamW，
先记录全局分模块R/U、裁剪比例和真实参数更新，不改变优化策略。

## 已验证启动

执行SHA990a5d4fbd497496c5a73748469254540781e095；单卡64039 COMPLETED0:0。
四卡作业64041在13号运行，独立四卡smoke通过后进入正式冻结联合训练。
20正式调用/160新增曝光保存状态通过独立CPU审计，累计2106调用/16848曝光包含复用适配。
实际beta0.0076，理解weight0.76，LR multiplier0.1，Adam参数时钟19/20，
四rank采样计数完全匹配原epoch0/1及epoch2前20步；VGGT probe变化全为0。
四卡峰值GiB19.4181/19.4268/19.4289/19.4248。
依赖正式官方评测64042等待afterok:64041，四卡，不要求八卡。
启动20步的阶段剩余估计约10.67小时，初期估计可能变化；不要当成保证。
证据startup_validation.json、POLLING_STOPPED.json；主训练停止轮询，继续运行。
