# A双分支三维超点初始化：完整数据4 epoch

用户明确授权在13服务器训练A。独立run：`/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu`；独立不可变快照：`/space/mawb/ssst_superpoint_dual_full_code_20261017`。使用13节点8×3090、32CPU/192G/24小时上限，不影响66432/66433，不重跑短验证或其他已完成任务。

## 不变的结构与完整训练配方

保持66502已通过真实GPU短验证的A结构：冻结官方VGGT-Omega与SAM ViT-B，同一3D超点分别经独立3D邻域编码/池化初始化两组Query；1024维重建Decoder与256维理解交互层L6/L8/L10/L12，旧Gaussian特征与Object Query相似度读出Mask。Write=0，正常联合反向，非梯度隔离，不增加结构或损失项。约193.8M可训练参数。

数据与66432一致：1191场景/8337窗口，补齐后每epoch8344曝光，1043更新；4epoch4172更新、33376曝光，全局批8，8rank每rank一个场景。AdamW LR1e-4、100步LR warmup/cosine、weight decay0.05（1D参数为0）、clip1；理解总权重0.05，1043步ramp。没有额外重建适配2epoch。每50步checkpoint，每epoch保留权重，初始和每epoch末运行相同4train/4heldout固定探针；关注A短测中高BCE个例、分类与Mask不一致，不因loss波动自动改配方。

## 严格零步起点与卡数变化

读取66502短测的`checkpoint_initial.pt`（0更新、空Adam），不读取256步终点。模型所有693张量逐位保持，不继承过拟合16场景的权重或优化器状态。

短测4rank、完整训练8rank，不能直接冒用8rank恢复。提交前显式检查源4rank Python/NumPy/Torch/CUDA RNG状态完全相同，才将这些相同零步状态复制为8rank；只修改初始化产物的world_size与rank_rng，原始短测文件不改动。新文件独立保存为run/checkpoint_untrained_initial.pt，记录原始/新文件SHA和转换过程；训练入口原有严格世界大小/资产/数据/config/零步验证保持不变。转换工具拒绝非零更新、非空Adam、不同源rank RNG或不合法卡数。

## 自动评测及比较

训练成功后afterok同节点1×3090任务：完整128固定heldout导出，原官方评分与11有限指标/2文本空列检查，与旧VGGT+MASt3R C及已完成V4隔离/权重0.05比较。若原Unified全量组已有COMPLETE.json，追加同表比较；若尚未有完整结果，则不等待或假造其结果。不改变官方评分器/数据/后处理。

预检锁定真实Omega/SAM SHA、数据SHA、完整训练/heldout数量与实际样本、评测commit/src状态。新增测试覆盖零步4→8rank转换及拒绝已训练/不一致源，比较工具兼容Unified的嵌套TRAINING_COMPLETE收据且继续拒绝短测和不完整训练；旧V4的冻结校验仍保留。修复的是汇总收据兼容性，不是修改指标。

训练耗时粗估约5–9小时，评测另计：依据A在4×3090短测7.58秒/更新，8卡取消两次梯度累积，但完整数据Query数量、节点负载及A较大的参数量会影响实际吞吐；不承诺线性加速。最终以新run日志为准，提交后只核对一次调度状态，不等待首步。

具体执行SHA、验证项数、作业号及提交状态以preflight.json、VALIDATION_EVIDENCE.json、submission.json与SUBMISSION_STATUS_ONCE.json为准。已完成短测的GPU同步证据不能替代新全量的首步证据。

提交前验证已完成：真实零步693张量逐位一致，源4rank RNG完全一致并显式扩展到8rank，空Adam/0更新、实际冻结文件及完整数据校验通过；28项CPU回归通过，shell语法、Python编译及diff检查通过。新全量GPU首步仍未核实。
