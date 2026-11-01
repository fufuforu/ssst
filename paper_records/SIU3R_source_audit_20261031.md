# SIU3R 与当前 QPanchor1 GT v2 源码核对

结论：SIU3R 没有当前 v2 的“GT Context 外参反投影 + 预测/GT基线比值校准”过程。当前版本的 GT Target 相机来源和矩阵还原正确，但生成输入条件及渲染流程没有完整对齐 SIU3R。上一轮 v2 修正改善了内部几何接口，却引入了已知 Context 外参条件；不能把它作为 SIU3R unposed 条件下的严格对齐复现。

本次是源码/CPU审计，未修改、取消或提交训练作业，未执行额外 GPU 质量实验。SIU3R 固定提交 8ea80166be76854f938e90521f1a5b688b755c87；src/configs 无修改。QP当前冻结代码 a631fe468edb6bcc567b14f5355966cb5644ae0f 未改动。

## SIU3R 怎样在“不输入外参”情况下训练和评测

“不需要相机外参”指生成 GS 时的输入条件，不表示训练/评测数据没有 GT 位姿。

1. src/data/components/scannet_dataset.py:194 从文件读取 GT K/c2w，relative_pose:90 用 inv(T_context0) @ T_view 统一到首 Context 坐标，不做逐场景深度中位数归一化或预测基线比例拟合。K前两行除256。
2. src/pipeline.py:55 step_wo_lift 只把 Context RGB、Context K 和分割训练标签传给模型。Context c2w 没有传进生成模型；Target c2w/K在 gaussians 生成后才传给渲染器。验证/测试的 step_w_query_class_logit_lift 使用同样分离。未发现标准 pipeline 中调用 Target pose optimization、PnP、Sim3 或 baseline calibration。
3. src/models/model.py:314 forward 参数没有外参；src/models/backbone_croco.py:278 把 K 编码成 intrinsic token。src/models/model.py:212 两个可训练点头输出 pts3d，直接堆叠后作为 GS means 传给 GaussianAdapter；不是从冻结单目深度配合 GT Context 外参反投影。坐标/尺度由可训练重建模块适配 GT 渲染约束，模型以首图为公共坐标参考。
4. src/models/gaussian_renderer.py:20 设置物理 near=.1、far=100、黑背景。其固定10倍缩放同时作用于相机中心、GS中心/协方差和裁剪，是统一数值单位变换，不是每个场景的基线估计。训练 RGB监督和验证/测试都用 GT Target K/c2w。
5. src/pipeline.py:338 重建损失为 MSE + .5 * 半分辨率LPIPS，另有分割和实例内深度平滑。没有我们原生配方的 L1+.2SSIM+visibility/early Chamfer。此差别本身是模型配方选择，不能归咎为程序错误，也不是本轮 GT/预测版本之间的唯一差异。

## 当前 v2 的问题分级

### P1：生成输入条件没有对齐（确定）

当前 scripts/gt_camera_protocol.py:26 绑定前两 Context 的 GT cam_view；:52 用它们重新反投影。tokengs/models/qp_anchor_gt_camera.py:27/37 强制 generate_gt 获取这些参数。GT Context 外参因此决定点云、锚点、native相机、语义投影和最终生成结果。

CPU执行了 SIU3R 原始 step_wo_lift 方法（模型/渲染器是确定性接口 fixture）：修改 Context 外参不改变生成输入/GS；修改 Target 外参只改变渲染输入，GS不变。当前实际 reframe_omega 则在修改 Context 旋转后改变生成点云。这是确定的协议差异，不是“GT Target source”这一个变量。

### P1：物理裁剪范围没有按统一单位规则对齐（确定）

当前用 a 作为 GT-provider单位到生成单位的比例，生成单位/米=.15*a，native单位/米=.6*a。但 native renderer 仍固定 near=.025、far=125；对应物理范围为 near=.025/(.6*a)、far=125/(.6*a)。理解分支的4倍单位桥接保持同样物理范围。

| a | 当前近裁剪/米 | 当前远裁剪/米 | SIU3R/米 |
|---:|---:|---:|---|
| .25 | .16667 | 833.33 | .1–100 |
| 1 | .04167 | 208.33 | .1–100 |
| 4 | .01042 | 52.08 | .1–100 |

所以当前相机/GS单位一致，不等于物理渲染协议一致。若以 .1–100m 为目标，native near/far应随 .6*a 换算，语义渲染器随 .15*a 换算。当前沿用灰背景，SIU3R为黑背景。这些是保留原 QP配方的选择，但阻止“完整对齐 SIU3R”的表述。没有量化它们对现有 PSNR差距的贡献。

### P2：反投影改变点云几何，不能只称作坐标变换（确定的设计变化，质量影响待测）

当前保留 frozen Omega 原始预测深度，把其 K/相对位姿替换为 GT 后重新 unproject。若两套 K、相对旋转/平移不同，这一般不是原点云的一次全局 Sim3，尤其两视图使用不同相对位姿时可能改变跨视图一致性。新点云仍作为固定锚点和早期 Chamfer 参照；SIU3R没有这条冻结深度锚定链。

CPU构造例验证了“原点云→重新反投影点云”无法用一个全局Sim3精确解释。此例不是对真实当前模型的重建精度测量，不能拿其残差大小解释实际0.5dB差距。

当前 Context_reprojection_max_px 检查只验证 unproject 后再用同一相机 project 能返回原像素；这是必要的数值自洽检查，不能证明预测深度正确，也不能证明两视图重建同一物体时一致。之前不应把它当作排除所有几何问题的证据。

### P2：尺度规则不是 SIU3R 方法（确定；不自动等于算术错误）

当前 a=predicted_Context_baseline / GT_provider_Context_baseline，并加[.25,4]界限与退化回退。SIU3R没有这种场景校准。它可定义一种内部归一化坐标，但依赖预测基线误差并改变物理先验/裁剪解释，不能用“SIU3R也如此”解释其必要性。单纯把 clip 再调宽不能解决生成输入条件差异。

### P3：主点与投影实现尚未严格对齐（确定源码差异，像素影响未量化）

SIU3R 的 src/models/cuda_splatting.py:70 从归一化K得到FoV，再构建对称 frustum，投影矩阵 x/y 偏移为零。我们 native gsplat 渲染器传完整像素K，包括 cx/cy。真实 scene0011_00 的 cx=127.536657、cy=129.610743，不恰好位于图像中心。两种实现不能仅凭同一K文件就宣称逐像素相同；还应核对两套rasterizer的半像素约定。这里没有 GPU 合成渲染，因此不声称具体像素误差或解释了多少 PSNR。不能为了匹配源码而未经验证地偷偷改GT主点。

### 配方差异：损失函数与可训练3D表示

SIU3R的点头/解码器可训练；我们 frozen Omega 深度和固定QP锚点是较强几何先验，XYZ头虽可学习局部偏移，并非完全不能纠正几何，但适应GT约束的能力/速度不同。两者重建目标函数也不同。它们应作为方法差异披露，不该为追求 PSNR直接混改所有损失后仍称原QPanchor1控制实验。

## 核对通过的部分

- 当前 cohort 与 SIU3R data/scannet/val_pair.json 完整比较：1860个Context窗口一致，每个 Target 帧集合一致。文件内视图次序可不同，但当前按frame_id导出，GT/预测视图关联未发现错位。
- 用 SIU3R 实际 relative_pose 和 intrinsics_normalize 源码在真实窗口执行，对比当前GT decoder：逆尺度还原后的Target c2w最大误差1.1921e-7，K归一化/像素转换一致。没有证据表明是 GT 位姿文件读错、c2w/w2c颠倒或256倍K错误。
- 目前没有 Target RGB或GT深度参与生成的证据；但 Context GT外参确实参与，不能混淆这两个断言。
- 既有GPU smoke/梯度隔离通过只说明训练执行和部分数值接口正确，不证明与SIU3R实验条件相同。

## 修正方向（本次未实施/提交）

若目标是 SIU3R 的 unposed 输入条件：生成接口应为 Context RGB+K，移除 GT Context 外参对点云/锚点/射线的依赖。让可训练GS位置/尺度在一个明确固定的首相机/米制坐标约定下输出，GT外参只用于渲染监督与评测。VGGT可作为特征/几何初始化，但其任意尺度必须由模型自身的可学习3D/尺度映射解决，而不是生成时读取GT Context位姿。一个 Context RGB/K预测的正尺度头是可研究的适配办法，非SIU3R本身实现，不能未经验证保证解决性能差距。

随后分别校验目标相机、物理裁剪/背景、投影主点/像素约定和序列化，复用同一评分器。固定模型和图像分别做相机/投影诊断，拆开Context与Novel指标；当前日志不能将差距唯一归因为GT监督本身或某一处程序错误。

当前 v2 应标为“GT Context pose-conditioned + GT Target rendering”实验，可保留其结果，但不能作为 SIU3R unposed条件的最终公平对比。此次未擅自再停训或重提实验。

证据：diagnostics/siu3r_vs_qpanchor1_GT_v2_20261031/ 内的源方法测试、cohort核对、投影差异和源码SHA记录。
