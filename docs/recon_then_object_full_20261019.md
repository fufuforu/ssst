# 完整重建后：Object Query先读几何、再读语义

用户授权：1024个重建token由点云采样初始化，理解token继续超点特征初始化，
先更新重建token，让Object Query读更新后的几何信息，再读语义特征，18服务器八卡训练。

已有67952实际在重建L6/L8/L10/L12之间执行“Object读几何→读语义”。
本轮已在对话中说明并询问时序区别，当前按**全部12层重建完成后，再串行执行4层Object更新**实现，
与已有层间交替实验区分；不是再跑一次相同的67952，也不是Concat共享Decoder版本。

## 明确数据流

- 冻结Omega/SAM；context-only原点云网格/置信度/FPS1024锚点。
- 保留原点云token初始化：学习基础token、几何/RGB内容、来源GFM特征；每锚点64 GS，总65536 GS。
- 理解token继续由融合点云上的SAM超点邻域编码初始化，保持256维和动态thing数，另有两个stuff token。
- 用重建GFM Adapter的K/V先完成全部12层1024维重建Decoder。重建位置和半径按原A方式更新。
- 四个Object层保留原参数键L6/L8/L10/L12，但全部在重建完成后执行；
  四层均读取最终L12的1024个重建token、最终位置和半径，经原几何编码/geometry bias作几何cross attention，
  再对独立语义特征作image cross attention，随后Query self attention与FFN。
- 保持A原L6锚点几何定义的归一化尺度；实际读取证据始终为L12，审计字段reconstruction_read_layer=12。
- 不用重建token替代超点Query初始值；没有stop-gradient或detach；理解损失可以反传重建分支。
- 没有Object→重建的Write注入；语义前向不会改写重建token/位置/半径。
- 重建监督读取原12层状态，理解分类与Mask继续原A读出/匹配/损失/渲染/后处理。

注册键 `recon_then_object`，架构 `OMEGA_POINT1024_RECON_THEN_OBJECT_GEOMETRY_SEMANTIC_V1`。
核心 `tokengs/models/reconstruction_then_object.py`。
原SuperpointDualCore仅提取decode_streams可重载接口，原层间交替实现默认运算不变。

## 可比性与训练预算

与attempt21点云1024版本参数键、维度和数量相同，**664个张量全部逐值使用其零步初始文件**，没有新增或删除参数，
不读取在训权重或Adam，不查询已有作业进度。空Adam、progress全0、共同8rank零步RNG显式1→8。
具体SHA和实际预检结果以新run/preflight.json为准。

完整1191场景/8337窗口，4epoch、4172更新、batch8、LR1e-4、LR warmup100、
理解权重.05和1043步ramp、每50步checkpoint、初始及epoch末固定探针。
18服务器实际为10张4090的节点，本次只申请其中8张；此前67952在13服务器3090，需注明硬件差异。

本轮CPU测试验证实际全12层重建先执行、四次几何/语义读取顺序、每次证据为最终1024个token、
没有前向回写或初始Query替换、理解损失对重建/语义特征的非零梯度及完整参数布局。
原A注意力残差输出层零初始化，梯度路径单元检查显式激活这些残差投影以验证其学习后的通路。
合成CPU renderer不代替真实GPU渲染/八卡首步验证。

提交后记录和接手入口为 `recon_then_object_full_submission_20261019.md`。不修改67952、67973或历史快照。
