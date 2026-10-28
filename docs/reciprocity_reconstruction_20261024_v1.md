# 互惠工程修复与重建差距诊断

本次入口为 `scripts/prepare_reciprocity_reconstruction.py` → `scripts/submit_reciprocity_reconstruction.py`。独立工作树基于 Membership v2 SHA4069215，原模型、数据、损失、官方评分器代码均不修改。未启用子代理。

## Membership G1/G2

两侧槽位索引验证整数类型后统一为CPU int64再严格比较，保留原门禁及MATH additive-bias SDPA guard。复用v2全部8337窗教师，只读符号链接；准备阶段核验每个文件SHA、源身份和形状及首更新8个RGB输入。固定教师当时成功加载v4 renderer；新链固定同一二进制并记录path/SHA，固定在其已验证兼容的3dimage-14使用四卡。

前置门禁在真实4rank DDP下对G1/G2分别执行两microbatch/一步Adam，核对48个几何张量梯度、真实参数改变、冻结模型及Adam不变、rank间权重及几何Adam完全一致。门禁无正式checkpoint，各正式组重新恢复Baseline4源及完整Adam/RNG。G1/G2保留1043更新、global8、LR1e-6、G2 lambda=.01/100步引入。afterok后正式导出、全量评分，另报Context/真Novel五项重建及7×7实例边界误差。

## GSQual 新同前向四组

A重新拟合Token-only；B GS-only；C Token+GS32；D Token+跨场景置换GS32。固定Baseline4 SHA694d2991…，1072/119/312场景划分；8337训练窗和1860正式窗。原二值Context class-agnostic IoU目标四组一致；dev MSE选模型，正式GT只事后评价。C/D初态和参数量相同。置换在train/dev/official各自内部保持完整行置换，禁止同场景donor；不使用GT，属于cohort负对照而非部署方案。

所有token、GS、标签、Query ID来自同一generate。正式窗在该次generated state上渲染全部视图，要求Context最终整数mask与原Context输出严格相同；落盘精确float32候选通道、alpha、类分数、原raw semantic argmax、RGB/depth及hash。只省去必定不可参赛的通道，读取时补零，CPU回归验证最终panoptic完全相同。四组正式部署读取这些输出，**不重新generate或依赖跨前向相等假设**。冻结参数版本/grad核对、全窗唯一完整门禁不放宽。记录首窗旧/新候选ID漂移，但旧缓存不作为标签或对应来源。

新Native结果先完整正式评分；A/B/C/D分别输出直接IoU排序及固定gamma=.25像素校准+直接排序结果。相同前向重建只评分一次，其他复用。原NMS、类别、no-object、区域门禁和stuff规则不变；校准仅缩放thing竞争分数。历史A单列漂移，不混入新A因果对照。保留Context一致性与Token-only残差统计，相关性不解释为因果。

## Baseline16 重建差距

固定SHA12608501…、progress16/16688。按head训练1072场景及head dev119场景各选16窗，同一scene只选一窗；分别覆盖8个小/大Context帧间隔，以pairIoU打破平局，预先保存完整输入哈希及provider相机baseline。保持原训练窗2Context+2Novel协议。两小集场景均已被骨干见过，internal16仅为头开发划分，非模型独立测试；不用正式1860选loss。

R0原生；Rcam以双Context相机朝向确定旋转、baseline长度确定尺度、中心均值确定平移，将GT相机/内参对齐至固定GS坐标后渲染。baseline小于1e-5、朝向残差≥15°或中心残差≥.15 baseline则拒绝，保存原因；只与相同通过窗的R0比较，不能伪称可靠Oracle。该方法依赖Context朝向和尺度近似，收益不能唯一归因相机。

Rrgb固定几何/scale/rotation/opacity及原生相机，只优化65536个Gaussian RGB，Context MSE，100步Adam LR=.01、RGB[0,1]；Novel GT仅评分，前11维逐位不变且颜色必须更新。报告两小集的Context/Novel PSNR/SSIM/LPIPS/官方逐图scale+shift深度、GT实例边界、固定原生alpha低/高覆盖带误差。Rcam/Rrgb是GT辅助或每窗拟合诊断，非可部署模型。原规范重建损失含MSE、SSIM、可见性，不在本轮删除任何项。

本批不预先提交LPIPS训练：必须先读取这组诊断证据，支持外观/感知不足时再固定原损失vs原+LPIPS两同预算短pilot。此条件在GPU任务结束后才能判断；不能在提交后等待结果。

## 执行与审计

CPU检查覆盖旧Membership/SDPA与修复、四头/置换、mask稀疏落盘等价、相机已知Sim3恢复及退化拒绝、RGB梯度隔离，共17项。真实GPU/DDP门禁待作业执行，CPU通过不代表GPU验证。每阶段trap写失败收据；afterok失败自动取消依赖。训练和评测时间限额是资源上限，不是耗时预测。

提交器先对全部12阶段sbatch --test-only，再登记独占提交尝试、逐个持久化作业号，最后仅调用一次squeue接收检查。旧71453链独立运行，不重提、不取消、不挤占。提交后不继续查日志、等待首步或后台轮询，等用户通知后核验收据与指标。
