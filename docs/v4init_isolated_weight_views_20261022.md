# V4init隔离：理解权重与2+4训练视角消融（2026-10-22）

用户授权新增两组，明确确认低权重采用0.05。两组均以当前71000→71001→71002的V4init NoCA隔离配方为基线；不改动或重提基线。

|组|理解loss基础权重|Context|Novel|重建监督视角|相对71001的唯一实验轴|
|---|---:|---:|---:|---:|---|
|基线71001|1.0|2|2|4|已有作业|
|U005|0.05|2|2|4|理解权重|
|Views2ctx4|1.0|2|4|6|训练重建视角|

不是把低权重与2+4组合成一组，也没有额外扫描或新增第四组。

## 共同配方与同一起点

- 完整V4重建初始化、1024 Anchor/65536GS；原持久基底、7D内容MLP和原始GFM2048视觉投影。
- 去Object→Anchor CA、保留FM CA；显式隔离重建tokens、几何及mask渲染几何。Write=0、无adapter。
- 理解仍旧A4投影点云→动态超点query，FM原生128，source_many_to_one/support_purity=.5；理解监督仅两个context。
- L6/L12监督[1/3,2/3]；主LR1e-4，refine_mu/refine_rho基础LR1e-5；warmup100/cosine乘数floor.1；理解权重都乘min(1,(update+1)/1043)。AdamW/weight decay规则及全参数global clip1保持基线。
- 每组8337窗/1191场景，8卡×1/global8，4epoch4172更新。沿用原epoch采样顺序与8rank RNG。
- 完全相同的V4init零步checkpoint_initial.pt，SHA `bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397`。空Adam、source updates0；不接着71001或70715的训练权重续训。
- 632保留张量逐值相同、32个CA张量精确移除、无新增参数；每组195961456可训练参数，48个mu/rho头张量进入geometry LR组。
- 关键8个模型/loss/decoder/采样源文件与基线执行快照逐字节相同，仅增加架构身份和训练入口参数支持，不能因新代码SHA而误认为底层重建结构不同。

## 2+4清单与可比性

复用已存在的冻结2+4清单，并复制到新运行根作为独立输入。清单loader验证全部8337窗口：原scene/context、原novel、顺序和其它parent metadata保留；只增加两个来自原context时间区间内的不同novel。没有重采样context，没有新数据场景。

真实首训练窗CPU batch对比通过：2+2 frame_ids=[122,174,142,156]；2+4=[122,174,142,156,160,166]。原四视角RGB、语义/实例GT及mask逐值一致，images_input逐值相同；真实images_all分别[1,4,3,256,256]和[1,6,3,256,256]，images_output分别2和4个novel。

生成仍只读两个context；增加的是重建监督视角。实际六视角重建loss会响应新增novel图像，新增novel的GT不进入理解loss，均已CPU测试。2+4保持4172更新，窗口曝光33376不变，重建监督视角曝光由133504增至200256；六视角loss仍沿既有目标均值归一化，不能把它解释为相同计算量实验。

## 验证和依赖

9项针对性CPU测试通过，11.211秒；包含真实core前向一致、双向梯度隔离、几何LR比、2+4清单拒绝非法修改、实际六视角RGB监督、context-only理解监督、权重梯度缩放及两组单轴身份。

真实源审计各拒绝8类非法迁移。真实数据batch可比性已验证。CPU未代替GPU：每组独立单卡2步smoke使用真实256²/1024/65536与对应2+2或2+4，global8通过8次累积实现；检查实际帧和calibration视角、双向隔离、9个初始化梯度及实际LR/理解warmup。

每组：smoke → afterok 8卡完整训练 → afterok 8卡正式1860评测。两组不互相等待，不依赖71001才能启动；正式训练从共同零步重新加载，不继承smoke。

另有CPU结果汇总，afterok依赖已有基线评测71002和两组新评测。它只读取完成结果、核对source/预算/输入/GT及权重SHA，生成相对基线CSV、逐类PQ差值和报告；不启动GPU、训练或重复评测。基线失败时汇总被阻断，新组结果仍各自保留。

评测固定1860窗/312场景、target-all6含context、SIU3R commit `8ea80166be76854f938e90521f1a5b688b755c87`，SIU3R/src不改。

## 执行路径与停止轮询

代码：`/space/mawb/ssst_v4init_weight_views_code_20261022_v1`。
运行根：`/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_weight_views_20261022_v1`，子目录U005和Views2ctx4各有smoke/train/eval。
GitHub分支：`experiments/v4init-isolated-weight-views-20261022`。

提交后只一次核对新作业调度接收状态，随后停止轮询；不查71000–71002的当前进度、不等待新组smoke或首步。GPU执行未核实的内容必须标注。用户通知完成后再读取相关收据、日志和正式指标，不重复提交。

完成入口：各组train/TRAINING_COMPLETE.json、FIRST_UPDATE_VERIFIED.json；eval/EVALUATION_VERIFIED.json和evaluation/metrics.json；总comparison/SUMMARY_VERIFIED.json、REPORT.md、vs_isolated_baseline.csv、per_class_pq.csv。单seed结论不等于稳定因果效应。
