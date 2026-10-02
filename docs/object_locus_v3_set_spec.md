# Object-Locus V3-Set

# Object-Locus V3-Set

本版保留 V2.1 的 token → anchor → object state → Gaussian child membership 路径：L6/L8/L10/L12、1024 anchors、100 thing + 2 stuff states、independent sigmoid masks、64 个 Gaussian children，以及 alpha-normalized pixel memberships。仅最终 L12 的分类和渲染 masks 接受理解监督。

## 监督

训练 GT 只由两个 context frame 的 semantic/instance labels 构造。Scene-global 的同一 instance ID 合并为一个 GT，按 ID 排序；thing 类别取有效像素 semantic 众数，平票取较小类。valid pixel 为 semantic `0..19` 且（stuff 或 instance ID > 0）。不以 predicted alpha、depth 或 anchor support 筛 GT。

一次 final Hungarian 只匹配 100 个 thing queries 与完整 GT 集合，成本为 `2*(-P(correct class)) + 5*BCE_cost + 5*Dice_cost`；BCE/Dice 基于两个 context 的 rendered probability masks 和全部 valid pixels。无 anchor matching 项。matched query 用 GT class-2 监督；unmatched query 的分类目标为 no-object `18`，该类 CE weight 固定 `0.1`。理解标量为：

```text
L_understanding = 0.1 * (2*L_cls + 5*L_thing_BCE + 5*L_thing_Dice
                          + 5*L_stuff_BCE + 5*L_stuff_Dice)
```

不含 anchor、identity、独立 semantic、auxiliary、objectness、competition 或额外正则 loss。像素 BCE 使用概率域 `binary_cross_entropy`，mask Dice 使用原 probability。

## Readout

分类器为共享的融合头和单一 19-channel class head（18 thing + no-object）。L6/L8/L10 分类由 anchor-mask pooling 读出；L12 分类在 Gaussian-mask pooling 后重算并用于 Hungarian。Gaussian child residual 与 V2.1 保持一致，mask probabilities 不做 object-channel softmax。

独立候选是逐 query 的重叠 masks，不进行 NMS 或 winner competition。Panoptic readout 独立执行资格阈值、像素最高分归属和低于 50% 保留面积过滤。官方 SIU3R evaluator 只接收 panoptic packed PNG；独立 candidate AP 是 local 指标，不冒充 official AP。

## Fresh training recipe

从 SHA256 `5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f` 的 pretrained reconstruction step 47500 fresh transfer；object branch seed 31415，global seed 42，optimizer fresh。固定 8 scenes、56 windows、64 epochs（3584 updates），每个窗口恰好 64 次 exposure。AdamW object/reconstruction peak LR 分别为 `1e-4 / 1e-6`，weight decay `0.05/0`，FP32，batch 1，GC alpha `0.01`，global grad clip `1.0`。Understanding 与 LR 前 200 updates 线性 warm-up，之后按注册 cosine schedule 到 10% peak LR。

结果是固定小规模训练集验收，不是跨场景泛化保证，也不是完整 unposed SIU3R benchmark。训练后无论 gate 如何，均停止，不自动扩大训练。
