# 固定权重机制与8维HDBSCAN读出对照

用户于2026-10-18明确授权：执行67050/67292固定权重机制对照，并比较“渲染8维实例特征→L2 Normalize→HDBSCAN→2D实例Mask”与原Membership Mask。无训练、无优化器、无新增训练组、无官方128。

## 固定源与数据

- 基线67050：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt11_unified_instance_grounding_short_node13_4gpu/checkpoint_latest.pt`，`unified_instance_grounding`。
- 新组67292：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt15_gaussian_instance_consistency_short_node13_4gpu/checkpoint_latest.pt`，`unified_gaussian_consistency`。
- 都是原256更新终点；不从smoke、initial或旧完整训练权重代替。预检记录checkpoint SHA256，核对原生模型文件与各自执行快照逐文件一致、资产hash/数据清单/配置/真实RGB和GT。
- 与既有探针完全相同的4train+4heldout场景及context帧，每窗口两视图：两模型16个窗口、32张图。聚类按图独立，跨视图不关联cluster ID。
- 独立执行快照预定`/space/mawb/ssst_fixed_instance_readout_code_20261018`。
- 独立运行目录`/space/mawb/ssst/group_plus/fixed_instance_readout_v1/attempt01_67050_67292`；拒绝重复提交。

## 机制对照

同次原生前向后才使用GT。按实际渲染贡献给Gaussian分配诊断归属，stuff进入有效像素贡献分母；统一采用thing纯度≥0.7、贡献质量≥0.001。保存可靠覆盖与每GT样本数量；不是mesh真实3D归属。

两组使用完全相同的采样代码和随机种子，分别对Child256与实例8维统计：同GT同父/跨父Gaussian余弦距离、异GT距离、全局及局部kNN同异GT AUROC、shuffle对照。局部kNN沿用旧诊断最多8近邻；另以新loss原配方重算跨父原型pull、半径0.1场景RMS内异GT最多8邻居push及margin违反率，二者分开报告。两组Gaussian集合不同，不能声称逐Gaussian对应或固定样本因果对照。

按原来源匹配获得每GT的候选Query，记录自身/局部异GT可靠Gaussian上的候选最大Membership响应，原阈值0.5及诊断阈值0.1/0.3/0.7的最佳IoU、正确类别/no-object数量、NMS保留数量；记录所有Query最佳IoU作读出上界。GT仅用于事后定位，不挑选部署输出。保存原始响应与中间缓存，重点核对scene0030_01。

门禁：checkpoint按名逐张量恢复一致、原生有符号8维特征同次重渲染一致、RGB和GT输入与旧缓存相同、预测有限；旧RGB数值误差记录并以最大0.02保护，不能宣称跨运行逐位确定性。开始/结束可训练参数SHA与源checkpoint一致、所有参数含冻结资产版本号不变、无任何参数grad；两名独立worker对同checkpoint源SHA一致。

## HDBSCAN配方

- 直接使用原生有符号8维特征渲染结果，**先渲染、再逐像素L2 Normalize**，不先归一化Gaussian，不裁负数、不经过概率截断。
- 使用现有`scikit-learn 1.5.2`的`sklearn.cluster.HDBSCAN`，不安装替代包。完整256×256分辨率，所有alpha>1e-6且8维范数>1e-6像素参与，无抽样、无缩图。
- 固定`min_cluster_size=100`（约全图0.15%）、`min_samples=10`（sklearn包含点自身）、L2单位球上Euclidean、EOM、epsilon0、`allow_single_cluster=False`、kd_tree、leaf40、每fit4CPU。
- 只用8维特征，不添加XY/RGB/GT，不指定真实实例数、不用heldout调参。一个固定配方不能排除HDBSCAN超参数敏感性。
- 每簇形成一个2D Mask；噪声/无覆盖/零向量均保持未分配，不做最近簇填充、连通域拆分或GT实例数裁剪。

## 统一评测

主要指标：每张图全部非空有效thing GT，含来源不支持的GT；void忽略，stuff有效像素保留，用于惩罚误扩张。类别无关2D Mask指标：GT最佳IoU（上界）、一对一Hungarian GT平均IoU（漏检计0）、IoU严格大于0.5的最大匹配数、Precision/Recall/F1。匹配优先最大化合格匹配数，IoU和仅用于平局；重复簇/重复Query计FP。保留每图/每场景明细，scope汇总按GT计IoU并微平均P/R，报告原始分母。无GT图像不伪造Recall。

同一指标分别评测：

1. `membership_raw`：全部原Query的Membership≥0.5，无类别/NMS筛选，允许原有重叠。
2. `membership_nms`：原生NMS eligible Query的原Mask。
3. `membership_assembled`：原生类别/alpha/NMS及像素竞争后的组装实例Mask。
4. `hdbscan8d`：所有可聚类像素的簇，包括背景簇；不凭GT删除背景簇。
5. `membership_raw_common_coverage`：原Mask限制在HDBSCAN可输入覆盖，仍在全部有效GT画布上计漏检。
6. `membership_raw_predicted_thing`与`hdbscan8d_predicted_thing`：双方共同用可输入覆盖与原生预测语义thing区域过滤输出，聚类标签不重拟合。无GT前景泄漏，单独标注为共享后处理对照。

另报来源支持GT子集的最佳IoU作为覆盖诊断；保存历史两视图、指定来源Query匹配的原探针IoU以核对。它与本轮每图/全GT指标不同，不能直接混用。没有类别/置信度排序的HDBSCAN簇不冒充官方AP/PQ；本轮不跑官方完整数据评测。

## 执行、产物与停止轮询

单一Slurm作业，拟用3dimage-13、4×3090、16CPU、128G、时限2小时。四worker各负责一个模型的四窗口，先完成GPU前向/机制/缓存导出，退出后四CPU进程各4CPU并行执行32张图的全分辨率HDBSCAN、统一评测、汇总和配对图片。GPU计算估计数分钟量级；全分辨率HDBSCAN耗时未实测，不保证准确ETA。

预检、针对性CPU单测及集成验证、独立干净快照均在提交前完成。提交后唯一状态核对只确认接收和调度状态；不等待GPU门禁、首窗口或聚类完成。阶段异常写`JOB_FAILED.json`，不自动重训或改聚类参数重试。

关键产物：`preflight.json`、`VALIDATION_EVIDENCE.json`、`submission.json`、`SCHEDULING_CHECK.json`、`STARTED.json`、各模型`FIXED_WEIGHTS_rankXX.json`、逐窗口`*_mechanism.json`/`*_inference.pt`/`*_clustering.json`/`*_cluster_maps.npz`、`GPU_COMPLETE.json`、`SUMMARY.json`、`figures/`、`COMPLETE.json`。原始GPU缓存可复用，CPU汇总失败不重新跑训练。用户通知完成后核实最终收据、固定参数证明、有效覆盖与每场景P/R/IoU，再给机制结论。

## 提交收据与当前交接

已提交 **67349**，执行SHA `8803dd4e3562c4727ebe6cfd1c5969c4e9aae2bc`。唯一调度核对：北京时间2026-10-18 14:58:16，**RUNNING，3dimage-13，gres/gpu:3090:4，16CPU，128G**。该状态只表示调度运行，真实CUDA门禁、固定参数结束证明、聚类和评测结果均尚未核实；未读取提交后的日志，不继续轮询。

提交前完成12项CPU测试（新聚类/计分6项、已有边界数学6项）、32张合成图完整CPU流程（16窗口、28行分方法汇总、8张配对图片）、Python编译/shell语法检查、真实源模型/资产/RGB/GT预检、独立干净快照。未重训任何模型。

训练作业：无；GPU诊断、CPU聚类评测及汇总顺序包含在67349中，无额外依赖作业。GPU部分参考旧诊断应为数分钟量级，全分辨率HDBSCAN真实耗时未测，不给精确ETA；2小时为申请上限。已停止轮询，等用户通知完成后再检查。提交和唯一调度证据另归档于`docs/fixed_instance_readout_submission_20261018.json`。
