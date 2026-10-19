# Concat共享Decoder/GFM：17服务器八卡full4交接

北京时间2026-10-19 **11:12:02**，已提交唯一训练作业 **67973**。

- 申请资源：**3dimage-17 / 8×4090 / 4090分区 / 32 CPU / 192G / 24小时上限**。
- 提交后唯一一次调度核对：`67973|PENDING||(None)`。这是11:12:02的历史状态，不能当作后续实时状态。
- 当次未见JOB_FAILED；GPU前向/反向、首步更新及八卡同步均**尚未核实**。
- 没有单独评测或汇总作业，没有后台监控；新配置尚无可靠GPU吞吐，不把24小时时限当作耗时预测。
- 运行目录：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt22_concat_shared_full_node17_8gpu`。
- 固定代码快照：`/space/mawb/ssst_concat_shared_full_code_20261019`。HEAD `58e86fa3b17ea044edbca46906546417c825102e` + 13个文件的 `SOURCE_MANIFEST.json` SHA清单。
- 日志：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt22_concat_shared_full_node17_8gpu/slurm-67973.out`。

## 用户授权与最终结构

用户要求：重建token仍按点云采样初始化1024个，理解token仍由超点特征初始化，
两类token Concat 后用共享Decoder/GFM Adapter更新，提交17服务器八卡训练。
注册键 `concat_shared`；架构 `OMEGA_POINT1024_SUPERPOINT_CONCAT_SHARED_DECODER_GFM_V1`。

固定1024个重建锚点，保留原点云FPS、基础token、几何/RGB及对应来源GFM特征投影。
理解thing token仍以融合点云上的超点邻域编码初始化，256→1024投影；两个stuff token同属理解侧。
序列严格为 `[1024 reconstruction | N superpoint thing | 2 stuff]`。
**只有一套12层1024维Decoder和一套共享GFM K/V Adapter**，self attention不隔断两类token，梯度双向传播。
原A的独立理解Decoder被删除；原独立语义GFM投影和模型gfm_projection被删除。
共享GFM投影网格经同一1024→256特征读出，用于点云超点编码及像素Mask特征，保留局部RGB/Mask读出头。
初始化中的来源GFM投影不是第二套Decoder K/V Adapter。

重建锚点保留原位置/半径更新。理解/stuff几何固定为初始化参考，内容每层更新。
每层重建监督只看前1024个token，最终只有这1024个token生成GS：每锚点64个，共65536个。
理解部分LN/1024→256任务读出，沿用A分类、Gaussian子特征/像素证据混合、Mask点积、原渲染、匹配及损失。
GS父关系使用gaussian_parent_anchor，不伪造对理解query的一对一归属。

这是新的共享结构；与A点云双Decoder版本相比，也改变了理解Decoder、理解特征来源、几何更新方式和参数量。
17服务器是4090，67952的13服务器是3090。不能声称只改变Concat的严格单变量或把硬件差异忽略。

## 初始化和提交前证据

仅读取attempt21的**零步**文件：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt21_A_dual_point1024_full_node13_8gpu/checkpoint_untrained_initial.pt`，SHA256 `0ea7d0a97743f06bf2e38595c602d6b2a6557262dbe2791db96060a1833a6e64`；
没有读取其训练中权重或Adam，没有查询其训练进度。
新模型489个张量，其中479个同名同形状共享张量与该零步文件逐值相同；
删除185个旧分支张量，新增10个投影张量（共享特征投影、理解输入/输出投影），seed42初始化。
progress全0，Adam为空，初始8rank共同RNG经严格1→8展开后与来源完全相同。

新初始文件：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt22_concat_shared_full_node17_8gpu/checkpoint_untrained_initial.pt`；SHA256 `1f568ec3939671ce35d3e343917cda54d6b944d6ea0455cd7c522ec075c11ac5`。
保存前后validate_zero_initial均通过，所有489个张量保存前后逐值相同且有限。
实际冻结Omega/SAM资产SHA、训练与heldout清单SHA均核对，
A历史33376次曝光顺序完全重建，实际首末训练窗口和heldout样本构造成功。

本轮最终 **13项CPU测试通过，38.438秒**（不含导入耗时）：4项新结构检查、9项原A/点云1024回归。
覆盖一套Adapter/Decoder、原K/V逐值一致、双向token梯度、理解几何固定、
12层均实际观察到1024+N+2长度、仅1024个token生成65536 GS及动态理解类别/Mask读出、完整无重复参数分组。
Python编译、shell语法、git diff --check、所有13个overlay SHA和启动入口verify-only均通过。
CPU使用合成Omega/SAM和替代renderer，不代表真实GPU/八卡验证通过；本地主机旧CUDA/glibc告警不作为GPU证据。

## 训练预算和后续检查

沿用A full4：1191场景/8337窗口；4epoch、4172更新、batch8、8rank每rank一场景；
AdamW LR1e-4、100步LR warmup/cosine、weight decay.05（1D为0）、clip1；
理解权重.05、1043步ramp；每50步checkpoint、epoch权重和初始/epoch末原固定探针。
不增加损失，不改官方后处理，没有额外短测或适配训练，没有自动官方128评测。

用户通知完成/检查后，先只读核对现有67973和该run的：
submission/一次核对记录、JOB_FAILED、STARTED（尤其concat_shared元数据）、
8份FIRST_GRADIENTS、FIRST_UPDATE_VERIFIED、TRAINING_COMPLETE、checkpoint及八rank连续日志。
审计4172更新、实际初始化一致、最终参数/Adam更新、数据顺序、有限性和固定探针。
缺报告或指标先定位产物与工程故障，不重复训练。若仍在运行，报告一次后停止轮询。

本轮没有取消、重提或恢复67952/67940/67931，没有修改任何历史运行目录或固定快照，
没有commit/push。当前workspace保留之前未提交修改，不得reset/clean。
新增核心为tokengs/models/concat_shared_decoder.py；准备/启动/提交脚本为prepare_concat_shared_full.py、
run_concat_shared_full.sh、submit_concat_shared_full.py，注册和训练入口只新增独立架构。

**已停止轮询，等你通知完成后再检查。**
