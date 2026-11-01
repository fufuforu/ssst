# ABCD v3：初始化及Adam审计的CPU/CUDA比较修复

用户通知看不到v2作业。71710/71711/71712/71713均FAILED/1:0，分别4:24/4:23/4:31/4:32，节点12/13；训练71714/71717/71720/71723及后续均因依赖失败取消。各rank在initialize的torch.equal(value,old)报CPU/CUDA设备不一致；四组smoke/train目录仍为空，没有前向、反向或参数更新。v1的启动参数修复有效，v2已真正启动各rank、装载模型/源/固定渲染器后才失败。

根因是CPU测试未覆盖实际GPU目标张量，审计直接torch.equal(CUDA模型,CPU源)，B原1024行检查也有相同缺陷。v3增加exact_equal：先核对shape/dtype，再将两侧detach().cpu()后逐值严格比较；不使用容差、不改变dtype、不降低迁移白名单。普通模型、B旧行和Adam检查统一使用该函数；Adam检查实际搬移后的state，而非搬移前的CPU临时副本，新B moment行仍严格零。

本次新增CPU回归覆盖两侧归一化边界、值/shape/dtype损坏拒绝、B参数/Adam前缀和新行检查，并复测真实torchrun参数解析。测试设备包装器仅覆盖调用约定，不冒称实际CUDA测试。完整四组实际模型/权重/Adam迁移的CPU审计重新执行，保留原14项科学测试结果；模型/loss/数据/初始化策略/Adam数值/训练预算均未改，tokengs无diff。

独立v3代码与run，旧v1/v2执行SHA及失败日志不改。恢复相同四组smoke→训练→导出→Native/纯Novel评分→汇总，禁止重复已完成阶段（本次无已完成训练）。ABC各追加1epoch，D1024Token/1024维独立零训4epoch保持原提交规格；用户关于4096的提问仅解释，未授权尺寸变更。

继续沿AGENTS：提交前封存全部准备；提交后仅一次接收核对，停止轮询，不等GPU首步或smoke。CPU通过不代表GPU前向/反向/优化器/DDP已经通过。
