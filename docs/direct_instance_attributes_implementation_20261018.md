# 直接Gaussian实例属性、Query原型归属与聚类读出

后续用户已授权短测，提交作业67369；见`docs/direct_instance_attributes_short_20261018.md`及对应提交收据。下文“本轮未提交”描述实现轮历史状态，当前不得据此重复提交。

用户授权实现“先直接预测Gaussian实例属性，再由该属性同时支持原型归属和聚类推理”。本轮只实现/验证代码，未启动训练、未提交GPU作业、未重复旧实验；真实CUDA行为与训练收益尚未验证。

## 架构

新注册入口`unified_direct_instance_attributes`，架构身份`OMEGA_UNIFIED_DIRECT_INSTANCE_ATTRIBUTES_V1`。原VGGT-Omega/SAM资产、三维超点初始化、共享Unified Decoder、几何/RGB解码、类别头保留。

最终Unified Query `Q[B,Q,D]`直接分支输出：

- 原几何/RGB路径：重建adapter与Gaussian head输出`[B,Q*P,14]`。
- 新实例属性头：`LayerNorm(D)→Linear(D,P*8)`，reshape为`[B,Q*P,8]`。每个Child有独立输出行/偏置，与几何头的query-major/child-minor顺序一致，不复制一个父Query向量。
- 新原型头：`LayerNorm(D)→Linear(D,8)`，输出`[B,Q,8]`。
- 新stuff属性头：`LayerNorm(D)→Linear(D,P*2)`，输出每Gaussian两个stuff logits。

8维属性直接由共享Query预测，不读Gaussian xyz、center、radius，也不先构造几何相关Child256再映射。删除本架构中原`child_feature`、`child_index`、`query_mask`、`stuff_feature`和旧`instance_head`，避免留下未使用的参数。旧架构类、已提交执行快照和历史checkpoint不改。

为保持原Renderer的14通道ABI，8维属性与14维几何/RGB张量并行存储，均一一对应相同Gaussian；这是存储接口选择，8维仍是直接预测的Gaussian属性。`gaussian_instance_attributes`与兼容字段`gaussian_instance_features`指向同一张量，不是两套实例头。

新增头初始化在CPU RNG fork内，不改变共享模块的初始参数和后续RNG。针对性测试核对新旧同名参数、初始Gaussian与类别输出逐值一致；因删除/新增输出头，不能再声称与旧组共同305张量完整一致，也不能加载67050/67292训练终点作为新架构。

## 两种归属/Mask读出

### 原型点积

FP32中分别L2 Normalize每Gaussian的8维属性`e_g`和全部Query的8维原型`p_q`，计算

`thing_logit[g,q] = exp(logit_scale).clamp(1,100) * dot(e_g,p_q) + logit_bias`。

scale初始10（等效温度0.1），bias初始0，两者可训练。每Gaussian对**所有**Query计算，独立sigmoid，允许当前source_many_to_one中同GT多个Query同时激活；不做所有Query互斥softmax、不限制仅归父Query。

拼接两个直接预测的stuff logits后sigmoid，交给原有Membership可微渲染、alpha归一化、语义合成、来源匹配Mask损失、类别/NMS/最终组装。原Mask概率阈值仍0.5；并不假设它无需后续验证。学习scale/bias用于提供响应尺度自由度，不代表已证实校准成功。

### 8维聚类

同一份**有符号原始属性**经现有Renderer得到8通道Feature Map；不裁负数、不先单位化Gaussian、不经过Membership概率截断。推理时再对渲染后像素向量L2 Normalize并HDBSCAN，沿用67349固定全分辨率配方，noise/无效像素保持未分配。聚类只用预测、alpha和8维特征，无GT/XY，不指定真实实例数。

`model.infer_instances(prediction, method='prototype'|'cluster', ...)`在一个生成结果上选择读出，可传目标相机做新视图读出。两者返回CPU实例标签和每图Mask列表；0表示背景/未分配。prototype还返回原始Membership候选、原生NMS eligible及最终组装语义。cluster是类别无关/逐视图ID，没有假造Query对应或跨视图实例ID。

HDBSCAN公共实现抽到`tokengs/instance_clustering.py`，原诊断脚本保持兼容导入，模型训练不导入或执行HDBSCAN；训练梯度不经过聚类。

## 训练监督与梯度

沿用67050渲染实例监督：

`L = Lrec + u*(Lunder_native + 0.01*Lrender_instance)`。

其中`Lrender_instance`保留原Pull/Push/Cross（2/1/2权重及原margin、两context视图、按实例平衡）。渲染后特征损失本身按像素L2 Normalize。**本架构不叠加67292直接Gaussian跨父一致性项**，避免把其异GT压近问题和新结构同时引入。此选择已记录在checkpoint配方中。

- 原Membership监督可更新属性头、原型头、可学习scale/bias、stuff头、共享Decoder及渲染几何。
- 渲染Pull/Push/Cross直接更新属性头及共享Query，通过Renderer贡献权重仍可更新几何；它不直接依赖Prototype参数。
- RGB重建可更新几何/RGB分支与共享Query，不直接更新实例输出头。
- 从属性预测中移除显式几何输入不等于冻结几何或重建；两任务仍共享Query，渲染仍可微。

参数分组覆盖全部可训练参数且不重复；已增加属性/原型范数、有效logit scale/bias日志，后续需要观察范数对特征混合的影响。所有新配方写入`direct_instance_attribute_recipe`，与原`instance_grounding_recipe`一起由初始化/恢复/导出校验。旧架构checkpoint拒绝隐式迁移。

## 训练与评测入口

训练器已接入`--architecture unified_direct_instance_attributes`。实现并不自动授权运行下面的训练命令；提交前仍需新架构专用真实资产预检和GPU门禁，不得直接套用硬编码旧`instance_head`/305参数的旧smoke、preflight和提交脚本。

固定checkpoint评测入口：

```bash
python scripts/evaluate_direct_instance_attributes.py \
  --checkpoint /absolute/path/checkpoint_latest.pt \
  --checkpoint-sha256 CHECKPOINT_SHA256 \
  --output /absolute/path/new_output_directory \
  --readout both --evaluate
```

该CLI不提交作业、不建优化器、不更新参数；支持torchrun多worker。核对架构/recipe/资产/数据清单，固定4train+4heldout探针，从同一生成结果评prototype raw/NMS/assembled与HDBSCAN，全GT、每图、类别无关一对一IoU及P/R。缓存两种读出、8维图和直接属性；保存参数前后SHA/版本/grad核验、逐图指标和汇总。只做8窗口诊断，不冒充官方128/mAP/PQ。

## 针对性验证

CPU测试覆盖：逐Child直出、明确几何隔离、初始共享/重建不变、全部Query余弦归属与stuff、Membership与渲染特征两条独立梯度、RGB不直接更新实例头、负值渲染、两种无GT推理/新视图、真实HDBSCAN、参数组/全部参数参与、checkpoint与recipe拒绝错误迁移。另做真实两rank CPU/Gloo、不同Query数量、checkpointed Decoder、两次优化更新/全参数SHA同步验证。

最终验证：45项CPU检查通过（新架构10项、原实例渲染11项、Unified18项、聚类/评测6项）；另1项真实两rank Gloo两步同步检查通过，使用`find_unused_parameters=False`，所有可训练参数均有有限梯度、两rank全参数SHA相同。Gloo初次被沙箱本地网络限制阻断，改为获准的本机loopback执行后通过，无GPU资源申请。Python编译、git diff检查、训练/评测CLI帮助入口通过。

CPU/合成Renderer结果不能代表真实CUDA渲染、真实数据覆盖或训练收益。校验收据及代码文件hash见`docs/direct_instance_attributes_validation_20261018.json`。

## 文件

- 模型：`tokengs/models/direct_instance_attributes.py`
- 公共聚类：`tokengs/instance_clustering.py`
- 注册/训练：`tokengs/models/superpoint_structural_variants.py`、`scripts/train_unified_query_v1.py`
- 双读出评测：`scripts/evaluate_direct_instance_attributes.py`
- CPU/分布式测试：`tests/test_direct_instance_attributes.py`、`tests/test_direct_instance_attributes_ddp.py`
