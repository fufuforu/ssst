# Point1024 FM-query：2 context + 4 novel 训练数据版本

用户最新要求：训练窗口从2+2扩到2+4，与评测每窗视图配置对齐。本轮只准备代码、冻结数据清单、CPU验证；未提交新GPU smoke/训练/评测，没有取消或修改旧69969。

## 当前已知作业状态

用户说“跑完了”后进行的唯一一次核对显示69969仍PENDING、无实际节点、无启动日志或收据。本次不再次轮询。它仍使用独立2+2快照，不是本目录的2+4版本，也不能用其后续结果证明2+4显存或训练通过。

## 数据变化

保留原A4的8337窗/1191场景、每条记录顺序、context ID、原两张novel和context-only标签/匹配信息。按原processed depth帧清单，取两个context之间的帧作为候选，排除原novel后用固定sha种子均匀无放回抽2张。四张novel按时间排序，最终frame_ids=context+novel，共六张不同图。

原始metadata、sampler RNG身份保持；不使用预测或指标挑新帧。新增帧RGB/depth/extrinsic/panoptic及intrinsic逐项检查，不复制图片。原始manifest不改，只写独立新清单及SHA。

官方SIU3R代码 src/data/components/scannet_dataset.py 在两个context之间random.sample额外帧；本版本对齐novel候选区间和无放回采样、2+4视图数。**context仍是原冻结A4 context集合**，并未按官方num_extra_target_views=4的候选数量门槛重新采样context，因此不能声称完整context联合分布严格相同。训练split与验证split也保持分离。

4epoch仍4172更新，global8、LR1e-4/warmup100/cosine1e-5、理解.05/1043、L6/L12、1024anchor/65536GS、隔离adapter/FM初始化均不改。重建监督从4个总视图增加到6个，监督图像预算与计算量增加；此臂不再是原A4相同数据预算。LU仍只用两个context的原实例/语义监督，不额外添加target语义loss。

## 代码入口

- scripts/prepare_training_windows_2ctx4novel.py：原数据清单→新六帧清单；拒绝覆盖，检查选中帧文件。
- scripts/training_windows_2ctx4novel.py：确定性扩窗、严格父清单/context/metadata验证、数据identity。
- scripts/train_unified_query_v1.py：dual_point1024_isoadapter在本快照默认读取独立2+4清单，opt.num_views=6；可用--training-window-manifest传同协议清单。其它架构路径不变。原训练probe的chosen索引保留，新训练probe读取对应六帧记录，验证probe原来就2+4。
- scripts/export_unified_query_checkpoint.py：验证新训练清单SHA及协议字段，opt.num_views=6；官方cohort/GT/评分不改。旧无2+4身份checkpoint不可误当本版本训练产物。
- 专用smoke与receipt verifier在本目录改为六帧契约，但**未提交**；旧submitter明确退出，防误写旧69969目录。未来提交需要独立2+4 run和授权。

模型源码相对154cd70保持不变，不新增可学习参数。provider原pinned_pair路径本就支持显式不同长度novel，原build_batch按window.context+novel核帧序；因此无需放宽旧随机sampler“2novel”的断言或修改基础数据provider。

## 身份与零步迁移

原零步657个保留张量/36删除/16adapter新增规则不变。原meta.manifest_sha256仍是**父清单/源零步兼容身份**；实际训练数据另用training_window_manifest_sha256、training_view_protocol、parent_training_manifest_sha256及context/novel/总视图数明确记录。

迁移收据新增training_data_extension字段，显式说明从旧零步权重开始使用新数据；不是放宽旧零步验证器接受任意数据。resume逐项比较新增identity，export也核对应清单内容哈希。新旧数据协议不能因模型张量形状相同而混用。

## 新数据路径

`/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_fmquery_training2ctx4novel_20261021_v1/training_windows.json`

同目录MANIFEST_RECEIPT.json记录8337窗/1191场景、SHA、选中帧资产校验。仅写小JSON，不写图片、特征缓存、大权重或训练结果。

## 验证边界

标准库7项测试覆盖扩窗确定性、context/旧novel保持、严格协议/metadata/帧序拒绝、六帧smoke契约与收据规则。真实数据CPU检查需确认images_input=[1,2,3,256,256]、images_all=[1,6,3,256,256]、images_output=[1,4,3,256,256]，且与原四帧batch的context图像逐值一致。新六帧GPU前反向/显存尚未验证，不能以原CPU6anchor测试或旧2+2 smoke替代。

## 冻结清单结果

生成成功：8337窗/1191场景，新增16674次novel观测，选中49021个不同的(scene,frame)并检查全部数据文件；JSON约7.7MB。SHA256=`8fe72589a2b8f1b52fa517cde39a0fce37d104adc2ee3a28585d1747292313f1`。原context/旧novel与顺序、metadata已全量逐窗核对。4轮仍33376次窗口呈现（含global8补齐），总监督图像从133504增加到200256，即+50%；loss仍沿原视图均值计算，不额外乘1.5。

## 已完成验证

7项标准库测试通过。真实数据CPU回放两窗（parent index0与原smoke index8205）通过：六帧shape/帧序正确，两个context的RGB、语义/实例标签、相机、深度均与父2+2 batch逐值一致；原两张novel的RGB也逐值保留。日志与JSON：training_2ctx4novel_cpu_data_audit.log/json。模型/数据provider源码相对154cd70无diff，未新增参数，未做GPU验证或新提交。
