# 原Unified完整训练完成审计，正式评测尚在运行

用户通知后只做有界检查：66432 COMPLETED/0:0，2026-10-17北京时间12:50:08至17:17:05、耗时4h26m57s，11节点8×3090。66433在17:17:05自动启动，检查时RUNNING约57秒，正式128导出/评分尚未产出；不能把下列探针当正式mAP/PQ。

## 训练完整性

4epoch/4172更新、33376场景曝光（含padding），8rank各4172条连续日志、每rank4172曝光；所有日志浮点标量有限，峰值allocated GPU约7.4GiB。checkpoint_initial为0更新/空Adam；latest为epoch4/4172，301/301可训练张量均相对initial变化且全有限，301个Adam状态均step4172，8rank RNG齐全，冻结资产/数据元信息一致。首更新参数SHA八rank同步通过。机器证据run/FULL_TRAINING_AUDIT.json。

## 固定探针走势

每列仅4个场景。Mask IoU对每个有来源支持的GT，取正匹配Query中的最佳Mask后按GT等权汇总；train20/24个GT有来源支持、heldout20/20。尚未经过分类/no-object/像素组装，非最终实例IoU/mAP/PQ。

| 检查点 | Train PSNR | Heldout PSNR | Train最佳Mask IoU | Heldout最佳Mask IoU | Heldout正Query分类正确率 |
|---|---:|---:|---:|---:|---:|
| initial | 10.935 | 11.625 | 0.1346 | 0.1202 | 0% |
| epoch1 | 17.231 | 16.755 | 0.4516 | 0.4716 | 79.31% |
| epoch2 | 17.694 | 17.367 | 0.4973 | 0.5633 | 80.00% |
| epoch3 | 18.555 | 17.790 | 0.5268 | 0.6241 | 81.38% |
| epoch4 | 18.877 | 17.956 | 0.5308 | 0.6067 | 84.14% |

Train/heldout理解loss初始2.079/2.078降到0.834/0.552。Heldout Mask IoU在epoch3略高于epoch4，非全指标单调改善；仅探针不据此改用epoch3正式评测。按预定方案评分epoch4。

Heldout scene0019_01：最佳Mask IoU0.9107，正Query分类71.4%，最终有2个实例ID，而支持GT仅1个；有残余重复/假阳性风险，需正式匹配才能定性。scene0011_00 IoU0.5523/分类67.4%；scene0030_00 IoU0.6255/分类97.1%。简单和复杂场景质量仍有差距。

全量训练的heldout Mask/PSNR明显高于其16场景短测，因此不能用A/B短测排名当最终排名。也不能据4场景PSNR17.956直接比较旧C的正式128 PSNR。等待同协议评测后再作结论。

## 已确认的自动汇总兼容性修复

原66433不可变代码09063a1的compare_gfm_native_runs仍读取旧V4平铺`full_updates/frozen_unchanged`，而原Unified实际TRAINING_COMPLETE使用嵌套progress；若官方导出/评分成功，最后compare阶段会因KeyError失败。这不影响已经产生的官方指标，但Slurm最终状态可能FAILED，需区分阶段。

嵌套收据兼容代码已在A/B快照584e814存在且经过测试。挂接无GPU的CPU补汇总 **66528**（11节点1CPU/8G，afterany:66433），先严格检查COMPLETE/128场景/11指标与原官方结果一致，再用修复的比较工具读取已有结果。它不做训练、导出或评分；若原评测本身未成功，则检查会拒绝，不伪造成功。唯一一次核对PENDING(Dependency)。run/COMPARISON_RECEIPT_FIX_SUBMISSION.json记录原因和代码，原失败收据保留。

当前无正式128质量结论，停止轮询，待用户通知评测完成后再检查66433/66528及指标。A/B训练未改动、未重复提交。
