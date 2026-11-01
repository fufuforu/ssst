# 重建 A/B/C 与 QuerySplat 迁移 D：执行规格

用户授权来自 2026-10-25 交接及“读取并继续执行”。不增加实验组，不使用 C8Cal 或新质量头，不使用 LPIPS 训练/TTO。A/B/C 各追加一轮是交接建议的助手默认；D 从零四轮是用户指定。

|组|源|Token/GS|训练范围|新增更新|
|---|---|---|---|---|
|A|Baseline8，8344 updates|1024 / 65536|重建；理解/共享输入冻结|1043|
|B|同A；显式参数/Adam扩容|4096 / 262144|同A|1043|
|C|同A；零输出RGB残差模块|1024 / 65536|仅新外观模块|1043|
|D|零步公共理解参数；重建全新初始化|1024 / 65536|新任务模块；冻结VGGT/SAM|4172|

ABC endpoint epoch9/9387，D endpoint epoch4/4172。ABC 对 A，D 对原 Baseline4 Native；两条线不作同起点因果比较。global8，八卡每卡一窗，8337窗/1191场景，2Context+2Novel，256²，L6/L12权重1/3、2/3。没有自动扩大训练预算。

## 源、参数和优化器

Baseline8 source SHA256 `bd1afe7e27ca01533ff5bfef6cd218a99c4f2294de08399a40cd2acf6eb98bc5`；632存储张量、604Adam状态。ABC保留所有原权重及Adam槽位，冻结参数的槽位保持原值。B仅扩容 `core.v4_initialization.gs_tokens`；旧1024行原值保留，新增3072行为原表三次重复+seed20261025/std1e-4扰动；Adam新增moment行零，旧step保留8344。所有其它形状或键差异均报错。

ABC原槽位保留源终点LR：main3e-6、geometry3e-7，保持常数。C新外观Adam从空开始，3e-5/100步warmup。此LR选择为助手默认。D主LR1e-4/warmup100/cosine下限1e-5，理解warmup1043、clip1；无旧局部几何refinement，不套旧48张量分组。

D公共参数从Baseline4引用的零步文件 `bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397`读取，progress0、Adam空。文件实际上是旧joint V4init架构664张量，原始A4零步来源 `3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744`。D只复制 gfm_projection、initializer.understanding、semantic_memory、panoptic 中实际存在且同形同义的张量；不复制已移除的Anchor CA，不复制任何重建/适配参数，不宣称复制张量是本次新随机。VGGT/SAM身份同基线。新重建部分seed42，新Adam空、progress0。逐张量来源在CPU及每rank初始化收据记录。

## A/B共同种子候选与C保护

ABC共同使用48×48/视角的原候选规则。若去重后不足4096，三组都从剩余有限像素按稳定confidence降序补到4096；仍不足则明确失败。没有复制补齐，不改点云max_points4096。补池是已披露的共同输入规则改动：极端输入下A/C种子可能与历史Baseline8不同；零残差等价相对相同候选池的源几何。每窗强制唯一来源数=Token数，父索引顺序=逐Anchor64GS。

C宽度1024/六层外观，RGB+世界坐标Plücker射线patch16，归一化patch相加后投影KV；仅Context。geometry token detached，独立learned query偏置，输出零初始化的`.25*tanh(delta)`RGB残差。原xyz/alpha/scale/rotation不变。理解分支child descriptor也读取原RGB，避免颜色残差暗中改变语义。GPU gate验证零残差GS逐值相同、更新后保护11字段与实际语义输出逐值相同。无C8头迁移。

## D：明确的官方结构适配

参考本地QuerySplat commit3465a1d2c789d8ebe71f4f9c0bae3f2c2fd726ae，Apache-2.0；移植 decoder cross→self→MLP次序、独立RGB+Plücker memory、geometry query+learned appearance query、geometry/appearance激活公式。沿用本仓库同源DecoderBlock原语。源码公式源文件为官方scripts/models/querysplat.py、enc_dec.py、activations.py；仓库LICENSE保留，不加载发布权重。

这是宽度1024适配，非发布2048/8192/512配置：12层几何、6层外观、1024query、每query64GS、SH1完整23维。几何从纯learned query读取Context Omega四层memory，memory投影沿基线接口并新随机；没有FPS颜色几何种子、原PE/逐层局部refinement或ray-bias。RGB输入沿基线[0,1]，patch16的RGB/Plücker卷积求和后LayerNorm，再归一化投影KV（此patch归一化顺序与发布代码分别归一化后相加不同，属于适配配置）。保留官方appearance→geometry梯度，理解读取仍detach。

几何signed-expm1坐标，z_offset1.0只加一次；scale cap.075，归一化四元数。SH DC来自`.5*tanh(raw)/C0`，高阶`.1*tanh(raw)`，alpha=sigmoid(raw−2)。不混入原Anchor相对坐标偏移。第一相机世界系沿基线，相机预处理/监督标定不变，必须通过真实GPU完整步检查。

L6与L12分别使用同一个新几何头和共享六层外观decoder做深监督，此项为训练适配。理解适配使用同一组GS的64子中心均值及尺度均值作为父几何，原1024维token给理解；DC RGB给child描述，实际RGB渲染完整SH1。原Anchor decoder仅复用detached语义读取，其未使用的PE/refinement辅助参数保持新随机且冻结，计入模型总参数，不计入训练参数。

SH1在PyTorch中对每个相机用camera→GS方向显式求值，再调用同一个固定RGB rasterizer。所有一阶系数可反传，不丢弃高阶、无backend切换。CPU检查解析SH/视角依赖/梯度；GPU gate检查真实GS23、完整反向、LU→重建零梯度与更新。VGGT/SAM冻结，不使用Novel特征重建；all-view pass仅做原相机标定。

## 验收、资源和依赖

提交前CPU验证完整实际模型/Adam迁移、键/形状白名单、参数分组、SH激活与射线memory、原core损失双向梯度回归；代码独立worktree和干净SHA归档。GPU smoke八卡两次完整更新，各组独立目录，smoke权重不进入正式训练；正式各rank重新读取固定源。gate强制有限梯度、实际参数变化、冻结参数无变化、DDP逐张量副本哈希一致、真实GS/父索引/唯一来源。C还检查更新后语义逐值不变。

资源只用3090/4090中11/12/13/14/17/18，单节点8卡；评分单卡，CPU汇总。依赖为四个smoke全部afterok → 每组训练 → 各自固定checkpoint导出 → 各自官方评分及纯Novel评分 → 总汇总。任一smoke失败则四组正式训练不启动，不修改Token/尺寸或重复起点。后续工程修复沿收据继续。

导出每组1860窗/312场景，不做参数更新，输入/GT与相应原Native参考逐项比对。正式11项、百分制理解六项、Target=all6；纯Novel额外用frame_ids[2:]四帧PNG相对链接调用同一官方重建评分，不混称all6为Novel。评分器SIU3R commit8ea80166be76854f938e90521f1a5b688b755c87，src不修改。

提交后仅一次有界调度接收核对；不等待GPU首步或smoke、不后台监控。RUNNING不能证明反向/更新/多卡通过。用户通知完成后检查实际收据，禁止重提已完成作业。
