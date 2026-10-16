# V4 从头联合4epoch最终结果：2026-10-16

65761训练COMPLETED/4172更新/4epoch/4:12:31，65762官方评测COMPLETED/9:56，65763比较COMPLETED。只加载冻结官方Omega/SAM，source checkpoint=null、adaptation_updates=0、short_updates=0，8337窗口完整覆盖，全部主要参数族有梯度和更新证据。无训练或评测失败。

| model | AbsRel | RMSE | PSNR | SSIM | LPIPS | context_mIoU_s | context_mAP | context_PQ | context_mIoU_t | novel_mIoU_s | novel_mAP | novel_PQ | novel_mIoU_t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_C | 0.08730 | 0.21321 | 22.59658 | 0.78076 | 0.36805 | 0.56024 | 0.30799 | 0.64552 |  | 0.55265 | 0.27376 | 0.60979 |  |
| v3_frozen_reconstruction | 0.06700 | 0.19638 | 21.96171 | 0.72416 | 0.47539 | 0.49606 | 0.22730 | 0.60014 |  | 0.49516 | 0.21495 | 0.57722 |  |
| v4_adapt2_frozen_reconstruction | 0.06695 | 0.19633 | 21.97076 | 0.72397 | 0.47532 | 0.51503 | 0.21754 | 0.58242 |  | 0.51443 | 0.19868 | 0.55452 |  |
| v4_fresh_direct_joint4 | 0.22148 | 0.45588 | 13.20795 | 0.51119 | 0.73823 | 0.40873 | 0.10221 | 0.50778 |  | 0.38543 | 0.07695 | 0.46129 |  |

Official native scores, same heldout128. Invalid depth stays NaN; text mIoU stays blank. No frame exclusion or metric substitution.

## 结论与限制

官方128场景：V4从头joint PSNR13.20795、AbsRel0.22148、RMSE0.45588、target mAP0.07695/PQ0.46129，明显落后于V4适配2后冻结组（21.97076、0.06695、0.19633、0.19868/0.55452）。11项指标全部有限、两项text留空。深度有限不等于精度正常；官方深度仍是逐图scale+shift对齐后的误差。

固定8验证PSNR：initial11.130→epoch1 13.407→epoch2 11.742→epoch3 12.387→epoch4 12.541；理解loss epoch1 .652→epoch2 7.429→epoch3 1.768→epoch4 1.819，体现明显波动，没有稳定学好重建。Mask匹配IoU末期 .631，不能代替官方mAP .077，分类与最终组装仍影响结果。

V4从头joint4与adapt2+冻结理解4的训练预算/初始化/冻结范围都不同，不能仅由该对照认定梯度冲突为唯一原因。结合既有V3适配后联合退化证据，优化干扰值得优先验证；下一步建议同样从头joint、仅隔离理解loss到重建参数的梯度，保持本轮其它设置，允许重建由自身loss继续优化。该建议未实现或提交。不能把本次结果解释为所有联合训练或几何初始化必然无效。

## 深度导出零值审计

两组各768帧，均无整帧全零深度；GT有效位置上，导出深度PNG为零的像素比例：V4冻结0.1207%，V4从头joint9.9850%。这是导出量化结果，不等于直接测量raw alpha空洞面积；提示覆盖/几何退化需进一步定位，不能只凭有限AbsRel/RMSE宣称深度正常。审计JSON见comparison_v4_fresh_joint_vs_frozen_20261016/depth_zero_pixel_audit.json。
