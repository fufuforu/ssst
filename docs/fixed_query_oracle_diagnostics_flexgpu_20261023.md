# 固定权重诊断改用可变卡数（2026-10-23）

用户授权不再等待整机8卡。原71215仍PENDING时已hold防止准备期间启动；原71216为依赖等待。新链提交前取消旧两项并核实终态，保留v1快照与全部提交记录。不得并行重复导出。

新执行快照 /space/mawb/ssst_fixed_oracle_diagnostics_code_20261023_v2_flexgpu；新运行目录 /space/mawb/ssst/group_plus/semantic_resolution_v1/fixed_query_oracle_diagnostics_20261023_v2_flexgpu。原v1实验定义、4epoch源权重SHA、模型/Oracle/评分源码、1860窗和312场景、Context GT限制、E0复现门禁全部相同；没有训练或新实验组。

唯一执行变化为launcher按SLURM_GPUS_ON_NODE启动进程（1–10卡），原range(rank,len(windows),world)分片和all_gather支持任意world，完整性校验仍要求1860唯一窗。此次选择18单节点6卡4090，24CPU/192G，12h时限；评分汇总afterok导出，1卡8CPU/96G，允许3090/4090分区、不再排除11。调度器当前11的8卡均已预留，18的10卡已分配4卡；GPU看起来空闲不等于Slurm可分配。

重用已核验且未运行的v1源权重和输入预检，针对可变卡数补做实际launcher参数、1860窗无重无漏分片、提交依赖与重复拒绝测试及Slurm test-only；再次核对源权重SHA和本轮代码一致性。CPU不代表GPU执行成功。不重做耗时的未变数据扫描，也不重复训练或评测旧三链。

按AGENTS，提交后只做一次有界接收检查并记新作业号，然后停止轮询。运行根HANDOFF.md保存真实提交和旧链取消记录。旧三条8epoch链不改动。
