# 完整重建→Object几何→语义：18服务器八卡full4交接

北京时间2026-10-19 **11:27:14**，已提交唯一训练作业 **67976**。

- 申请：**3dimage-18 / 8×4090 / 4090分区 / 32 CPU / 192G / 24小时上限**。
- 18节点共10张4090，本作业只申请8张。
- 提交后唯一一次调度核对：`67976|PENDING||(None)`。这是11:27:14的历史状态，不代表后续仍在排队。
- 当次未见JOB_FAILED；实际GPU前向/反向、首步更新、八卡同步均尚未核实。
- 没有独立评测/汇总作业，没有后台监控。新配置暂无可靠GPU吞吐，24小时只是调度时限，不作耗时预测。
- 输出目录：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt23_recon_then_object_full_node18_8gpu`。
- 固定快照：`/space/mawb/ssst_recon_then_object_full_code_20261019`，HEAD `58e86fa3b17ea044edbca46906546417c825102e` 加14个文件的SOURCE_MANIFEST.json SHA清单。
- 日志：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt23_recon_then_object_full_node18_8gpu/slurm-67976.out`。

## 本轮执行解释和真实区别

用户要求：重建token点云采样初始化1024个，理解token仍由超点特征初始化，先用重建特征更新重建token，
再让每个Object Query从更新后的1024个token读相关几何，最后利用语义特征更新，并授权18服务器八卡训练。

已有67952本来就在L6/L8/L10/L12间执行“更新重建→Object读几何→读语义”。
本轮已向用户说明该重合并询问时序；在未收到进一步修订时，按已在对话中说明的解释实现为：
**先完整完成全部12层重建，再做4层Object几何→语义更新**。
不是声称用户另行确认了每个实现细节，也不是重复提交已有层间交替组。

注册键 `recon_then_object`；架构 `OMEGA_POINT1024_RECON_THEN_OBJECT_GEOMETRY_SEMANTIC_V1`。

1. 重建侧保持原点云网格/置信度/FPS1024锚点、基础token+几何/RGB+来源GFM初始化，1024维12层Decoder读取重建GFM K/V。
2. 理解侧保持原融合点云超点邻域编码，256维动态thing Query，加两个stuff Query；不由重建token替代初始Query。
3. 所有重建层先执行完，再执行保留原L6/L8/L10/L12参数键的4个Object层；四层全部读取最终L12的1024个token及最终位置/半径。
4. 每个Object层顺序为几何cross attention→语义image cross attention→Query self attention→FFN，带原几何bias及动态Query几何更新。
5. 保持原A L6几何定义的归一化尺度；证据本身来自最终L12，审计字段reconstruction_read_layer=12、reconstruction_read_count=1024。
6. 没有detach或stop-gradient，理解损失可以反传重建特征；没有Object→重建Write。语义前向不会修改任何重建token/位置/半径。
7. 每层保留原重建状态供原监督；最终1024个锚点各64个GS，共65536 GS。分类、Mask、匹配、损失、渲染及官方后处理保持A逻辑。

本版本保留原A独立重建/语义Adapter和Object Decoder，**不是**67973的Concat共享Decoder/GFM。
与67952相比，参数布局/初始值完全一致，主要改变Object读取几何的层次及执行时序；硬件由3090变为4090，需单列说明。

## 初始化与真实预检

仅读取attempt21的零步文件：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt21_A_dual_point1024_full_node13_8gpu/checkpoint_untrained_initial.pt`，SHA256 `0ea7d0a97743f06bf2e38595c602d6b2a6557262dbe2791db96060a1833a6e64`。
没有读取其训练中权重或Adam，也没有查询已有作业进度。
**全部664个张量逐值相同，无增删键，无新增模块参数**；progress全0，Adam为空。
共同零步RNG显式1→8扩展，8rank与来源逐值一致。

新初始文件：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt23_recon_then_object_full_node18_8gpu/checkpoint_untrained_initial.pt`；SHA256 `f3cb086bf7cd96293dc2e65a2942e117c7426db64bfa8633b8dd8526c7603833`。
保存前后validate_zero_initial均通过，全部664张量保存前后逐值相同且有限。
冻结Omega/SAM真实资产SHA、训练/heldout清单SHA已核对，完整A历史33376次曝光顺序全部重建；实际首末训练窗口和heldout样本构造成功。

最终 **13项唯一CPU检查通过，37.553秒**（不含环境导入）：4项新结构检查+9项原A/点云1024回归。
实际记录全部12层重建先于四次几何/语义读取，各Object层均读取最终1024个token，
验证理解损失对重建/语义/超点Query的非零梯度、无前向回写、没有替换超点初始Query、完整664训练张量布局及无重复参数分组。
测试使用完整配置检查参数布局，按实际checkpoint口径排除冻结假模型权重。
原A的注意力残差输出层保持零初始化；梯度通路单元检查显式激活这些输出投影以验证其学习后的通路，没有因此修改训练初始化。

Python编译、shell语法、git diff --check、14文件快照SHA与启动入口verify-only均通过。
CPU合成Omega/SAM和替代renderer不代表真实GPU或八卡通过；本地主机CUDA/glibc告警不作GPU证据。

## 预算及后续接手

保持A full4：1191场景/8337窗口、4epoch/4172更新、batch8、8rank每rank一场景；
AdamW LR1e-4、100步warmup/cosine、weight decay.05（1D为0）、clip1；理解权重.05和1043步ramp。
每50步checkpoint、epoch权重及初始/epoch末固定探针。没有额外训练短测、适配或自动正式128评测。

用户通知完成/检查后，核对现有67976、JOB_FAILED、STARTED中的reconstruction_then_object元数据、
8份FIRST_GRADIENTS、FIRST_UPDATE_VERIFIED、TRAINING_COMPLETE、八rank连续日志及checkpoint。
审计4172更新、实际初始664张量一致、最终有限性、数据顺序、参数/Adam更新和固定探针。
四个Write=0的W_inject参数为布局对齐而保留，输出不参与重建；正式审计应区分未用/零梯度参数，不机械要求全部664参数都存在Adam状态或数值变化。
若仍在运行，报告一次后停止。缺报告先只读定位，不重训，不重提，不因本交接PENDING记录而恢复任何旧作业。

相关实验只作身份索引，以下均非本轮新查询的实时状态：

- 67952：attempt21，13节点8×3090，点云1024双分支，Object在重建L6/L8/L10/L12之间读几何→语义。
- 67973：attempt22，17节点8×4090，Concat共享Decoder/GFM。
- 67976：attempt23，18节点8×4090，完整重建后四层Object均读最终L12几何→语义。

本轮没有修改、取消、恢复或重新提交67952/67973/67940/67931，没有修改任何历史运行目录或快照。
workspace保留既有未提交改动，不得reset/clean；本轮没有commit或push。
新增模型reconstruction_then_object.py；新准备/启动/提交脚本prepare_recon_then_object_full.py、run_recon_then_object_full.sh、submit_recon_then_object_full.py。
原SuperpointDualCore只提取decode_streams接口，默认层间交替逻辑保持不变；原A回归已通过。

**已停止轮询，等你通知完成后再检查。**
