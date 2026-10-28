# Baseline8 A8/C8调度范围变更（用户明确授权）

用户要求同时等待11、12、13、14、17、18。此前71638–71641尚未启动，已hold且旧run无STARTED收据。使用独立新快照/运行目录替换待调度链，原代码快照及完成/提交收据不改；不改12服务器71621–71625链。

允许节点为3dimage-11/12/13/14/17/18，每个阶段仍为单节点：缓存8卡、拟合CPU、导出8卡、评分5卡。节点11–13属3090分区，14/17/18属4090分区；新提交使用两个分区并排除候选集合之外的节点，不使用含六台的--nodelist（后者要求多节点而非六选一）。

cache/export节点门禁改为允许集合，记录实际主机、GPU型号与compute capability。不同阶段可落到不同允许节点；同一阶段全部rank同节点，A8/C8共享同一缓存，正式不重新generate。原partial_1406 renderer字节不变，仍严格核对path/SHA。cuobjdump已确认同时有sm_86 cubin与PTX；可供4090驱动处理，但这不等同于已验证新4090 GPU前向或同步。节点旧glibc兼容性沿用此前13成功的同一库。

源Baseline8、8337＋1860窗口、场景划分、GS32、网络、IoU监督、头训练、校准及原评分器不变。本次只变更可调度位置和运行来源记录，源代码科学函数与训练配方不变。原数据和GPU准备已完成，新增10项针对性回归及Python/bash语法核对；新run预检会再核对源SHA和场景清单。

新代码：/space/mawb/ssst_frozen_baseline8_gs_quality_code_20261024_v2_flexnodes。
新run：/space/mawb/ssst/group_plus/semantic_resolution_v1/frozen_baseline8_gs_quality_20261024_v2_flexnodes。
新链提交前取消旧hold链（零计算），不恢复原作业；afterok依赖按新作业号重建。提交后只做一次有界接收检查，记录实际状态后停止轮询。
