# Point1024 + FM query + 隔离残差 adapter：2+4 full4

用户授权在13节点完整训练。独立运行目录：`/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_2ctx4novel_full4_20261021_v1_node13_8gpu`；独立源码快照：`/space/mawb/ssst_point1024_fmquery_2ctx4novel_full4_code_20261021_v1_node13`。

从原A4零步权重严格迁移，SHA 3c7f1860...b25744，657张量保留，36个旧重建初始化张量删除，16adapter张量新增；不用smoke或任何训练后权重。使用原8rank RNG各自rank，零Adam状态；配置取原preflight.config显式传入，运行期严格匹配源meta。

8337窗/1191场景；2context+4novel，256²；原始context/旧novel保留。实际清单SHA 8fe72589a2b8f1b52fa517cde39a0fce37d104adc2ee3a28585d1747292313f1。8卡×1、4epoch/4172updates；LR1e-4/warmup100/cosine1e-5，LU权重.05×min(1,(step+1)/1043)，L6/L12权重1/3与2/3。AdamW/clip1保持原训练入口；Write0，无颜色先验或可学习anchor基底。理解FM query，4个隔离残差adapter；1024anchor×64GS。与通过69992 smoke的训练/模型/数据代码无diff。

专用run/submit脚本不复用FPS作业。8×3090@3dimage-13，32CPU/192GB，24h时限，epoch probes及原checkpoint-every50/各epoch保留不变。完整复用旧probe口径，probe不等于官方评测。预计权重/优化器保留约十余GB，不拷贝冻结模型或图像，源零步hardlink。只提交训练；没有自动1860或128官方评测依赖。完成后写TRAINING_COMPLETE.json；首步FIRST_UPDATE_VERIFIED.json验证8rank完整权重hash一致且均发生更新，FIRST_GRADIENTS_rankXX记录各rank梯度。

提交前标准库7项测试及Python/shell语法通过，复算smoke receipt和数据/源权重hash、逐窗数据兼容验证通过。单卡smoke allocated峰值11.05GiB，第二步8次累积34.87s；理想8卡约4.36s/update即约5h纯训练，真实场景差异、通信、探针/保存会增加时间，仅可粗估5–10h、不含排队。8rank同步与全场景峰值尚未验证，不能由单卡smoke替代。提交后只做一次有界squeue并停止轮询，实际job见run/SUBMITTED.json。
