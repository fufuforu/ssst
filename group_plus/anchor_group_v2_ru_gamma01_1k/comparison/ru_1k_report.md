# Anchor-Group V2-RU gamma=0.1 — Paired 1k report

## Provenance / recipe

- GPU: NVIDIA GeForce RTX 3090; pretrained SHA: 5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f; manifest SHA: 1f37d08c2941920d94a172d62f95dd494b1126374215833999dc9d75ad9fc483; plan SHA: ffd97c5e018fd5075650a5718503b26c3e566f0e3b516e392ebc657aa01c1323 (first 1000 steps).
- Only scientific variable: query update gamma, Control=1.0 and RU=0.1. Both used GC alpha=.01, unmatched_noobj_scale=1.0, identical optimizer/LR/warm-up/clip and fresh seed-42/seed-31415 initialization.
- No new parameter, gate, normalization, detach, loss, Hungarian or aggregation change.

## Table 1 — Structural primary

|Step|Arm|L6 q-in PR|L6 q-in cos p90|L6 q-out PR|PR retention|L6 q-out cos p90|L6 q-out norm median|L6 u-out PR|L12 q-out PR|L12 q-out cos p90|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|control|71.0176|0.0819|2.5216|0.0355|0.9690|15.9999|2.5507|2.9889|0.9679|
|0|ru|71.0176|0.0819|3.5422|0.0499|0.9303|1.6307|3.2716|3.1670|0.9801|
|200|control|71.0176|0.0819|2.5657|0.0361|0.9669|15.9999|2.5665|3.0512|0.9656|
|200|ru|71.0176|0.0819|3.5462|0.0499|0.9294|1.6304|3.3212|3.2182|0.9794|
|500|control|70.7694|0.0992|2.3479|0.0332|0.9977|16.0022|2.1551|2.1174|0.9989|
|500|ru|70.8779|0.0999|3.2457|0.0458|0.9550|1.6519|1.9586|2.1949|0.9928|
|1000|control|69.8809|0.1412|2.1042|0.0301|0.9991|16.0002|1.8565|1.9602|0.9998|
|1000|ru|70.0173|0.1459|2.3420|0.0334|0.9668|1.6791|1.5023|1.4631|0.9957|

## Table 2 — GT specialization

|Step|Arm|L6 best Dice median|L6 hard-correct median|L6 bestQ effQ|L6 bestQ top5|L12 best Dice median|L12 hard-correct median|L12 bestQ effQ|L12 bestQ top5|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|control|0.1127|0.3044|2.984|0.9922|0.0925|0.2778|2.814|1.0000|
|0|ru|0.1210|0.3322|2.821|0.9922|0.0848|0.2060|2.675|1.0000|
|200|control|0.1106|0.2624|2.911|0.9922|0.0909|0.2768|2.827|0.9922|
|200|ru|0.1081|0.3177|2.710|0.9922|0.0702|0.2701|2.621|1.0000|
|500|control|0.1763|0.1550|2.207|1.0000|0.2307|0.1658|2.679|1.0000|
|500|ru|0.2055|0.2928|2.444|1.0000|0.2056|0.2250|2.436|1.0000|
|1000|control|0.1841|0.2313|2.412|1.0000|0.2710|0.3320|2.645|1.0000|
|1000|ru|0.2415|0.3228|2.599|1.0000|0.2472|0.2133|2.509|1.0000|

## Table 3 — Assignment

|Step|Arm|L6 Apost effQ|L6 Apost Gini|L12 Apost effQ|L12 Apost Gini|
|---:|---|---:|---:|---:|---:|
|0|control|73.008|0.4045|84.176|0.3171|
|0|ru|72.058|0.4184|93.226|0.1994|
|200|control|72.540|0.4074|84.064|0.3216|
|200|ru|70.742|0.4268|94.065|0.1866|
|500|control|38.813|0.5779|4.601|0.9486|
|500|ru|7.275|0.9061|2.795|0.9725|
|1000|control|37.674|0.5408|4.179|0.9608|
|1000|ru|5.243|0.9227|4.681|0.9494|

## Table 4 — Online positive exposure

|Step|Arm|Matches|Unique|Never|Top5|Top10|Gini|Effective Q|First positive coverage|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|200|control|805|83|17|0.2932|0.4621|0.6459|46.697|83/100|
|200|ru|805|82|18|0.2584|0.4037|0.5933|53.111|82/100|
|500|control|1959|91|9|0.4661|0.6161|0.7265|33.112|91/100|
|500|ru|1959|91|9|0.4257|0.5758|0.6910|37.420|91/100|
|1000|control|3966|93|7|0.5802|0.7524|0.7995|22.776|93/100|
|1000|ru|3966|96|4|0.5764|0.7231|0.7847|24.249|96/100|

## Table 5 — Train1024 utilization

|Step|Arm|Matches|Unique|Never|Top5|Top10|Gini|Effective Q|Ownership Gini|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|control|4062|95|5|0.2740|0.4572|0.6305|49.082|0.2061|
|0|ru|4062|94|6|0.2671|0.4249|0.6085|51.678|0.1202|
|200|control|4062|95|5|0.3013|0.4586|0.6212|49.537|0.2154|
|200|ru|4062|94|6|0.2575|0.4163|0.5903|53.903|0.1112|
|500|control|4062|61|39|0.7395|0.9121|0.9124|11.214|0.9503|
|500|ru|4062|65|35|0.7681|0.8917|0.9053|11.536|0.9724|
|1000|control|4062|42|58|0.7432|0.9242|0.9180|10.906|0.9617|
|1000|ru|4062|56|44|0.7632|0.9087|0.9132|10.992|0.9511|

## Table 6 — Val32 task and direct grouping

|Step|Arm|Thing mIoU ctx/tgt|ca-R50 ctx/tgt|class-aware ctx/tgt|ca TP/FP/FN ctx|ca TP/FP/FN target|active queries ctx/tgt|Supported-GT recall50|PSNR ctx/tgt|
|---:|---|---|---|---|---|---|---:|---:|---|
|0|control|0.00133/0.00146|0.00000/0.00000|0.00000/0.00000|0/0/163|0/0/164|100.00/100.00|0.10638|25.6333/24.4012|
|0|ru|0.00132/0.00144|0.00000/0.00000|0.00000/0.00000|0/0/163|0/0/164|100.00/100.00|0.07801|25.6333/24.4012|
|200|control|0.00235/0.00248|0.00000/0.00000|0.00000/0.00000|0/0/163|0/0/164|100.00/100.00|0.07746|25.3201/24.0892|
|200|ru|0.00133/0.00145|0.00000/0.00000|0.00000/0.00000|0/0/163|0/0/164|100.00/100.00|0.04895|25.3587/24.2127|
|500|control|0.00257/0.00209|0.00000/0.00000|0.00000/0.00000|0/16/163|0/17/164|7.25/7.25|0.04196|25.3449/24.2419|
|500|ru|0.00156/0.00167|0.00613/0.00610|0.00000/0.00000|1/19/162|1/19/163|5.12/5.12|0.13103|25.2927/24.1319|
|1000|control|0.01308/0.01111|0.01227/0.01220|0.01227/0.01220|2/40/161|2/40/162|10.53/10.53|0.11806|25.2459/24.1248|
|1000|ru|0.00429/0.00370|0.00000/0.00000|0.00000/0.00000|0/13/163|0/13/164|8.78/8.78|0.12766|25.1995/24.0744|

## Val32 context mechanism diagnostics

|Step|Arm|Anchor ownership accuracy|Thing-anchor correct|Supported-GT recall50|Assignment entropy|Thing mass mean/median/p10/p90/max/max:median|Query cosine mean/p90/max|No-object mean/max|Active queries|
|---:|---|---:|---:|---:|---:|---|---|---|---:|
|0|control|0.14818|0.27155|0.10638|4.35288|9.9352/8.5468/3.8282/17.3460/31.7598/3.80|0.81806/0.96625/0.99650|0.02232/0.06380|100.00|
|0|ru|0.12771|0.22198|0.07801|4.41471|9.7755/9.2867/5.3818/14.8161/21.5723/2.37|0.92022/0.97976/0.99210|0.05893/0.07657|100.00|
|200|control|0.13846|0.25368|0.07746|4.33273|9.9039/8.4379/3.5900/17.7817/31.7578/3.86|0.80875/0.96353/0.99618|0.02529/0.07135|100.00|
|200|ru|0.12365|0.21654|0.04895|4.41700|9.7748/9.3241/5.5898/14.4650/20.7833/2.28|0.91692/0.97903/0.99193|0.06020/0.07790|100.00|
|500|control|0.38593|0.15854|0.04196|1.35645|5.2404/0.0714/0.0536/0.8321/214.3289/4427.77|0.88714/0.99834/0.99977|0.87838/0.97204|7.25|
|500|ru|0.42350|0.32111|0.13103|1.13471|5.1640/0.0288/0.0212/0.2737/328.4635/25250.06|0.88634/0.99252/0.99573|0.87482/0.94283|5.12|
|1000|control|0.45562|0.28054|0.11806|1.26023|5.4946/0.0227/0.0215/0.3701/234.6379/18048.61|0.85587/0.99978/0.99994|0.85666/0.96138|10.53|
|1000|ru|0.47011|0.29655|0.12766|1.24596|5.0747/0.0528/0.0466/1.3107/228.8984/7363.79|0.83540/0.99578/0.99686|0.88330/0.97226|8.78|

## Paired directional deltas (RU − Control)

Values below are raw difference and relative difference, computed from the registered checkpoint artifacts; no success threshold is applied.

|Step|Metric|Control|RU|Absolute delta|Relative delta|
|---:|---|---:|---:|---:|---:|
|500|L6 q-out PR|2.34794|3.24569|+0.897751|+38.236%|
|500|L6 q-out cosine p90|0.997728|0.955026|-0.0427017|-4.280%|
|500|L6 GT best Dice|0.17634|0.20545|+0.0291099|+16.508%|
|500|L12 GT hard-correct|0.16582|0.225017|+0.0591968|+35.699%|
|500|L12 best-query effective Q|2.67874|2.43588|-0.242858|-9.066%|
|500|train1024 unique matched queries|61|65|+4|+6.557%|
|500|train1024 effective query count|11.2135|11.5364|+0.322884|+2.879%|
|500|train1024 top5 match share|0.739537|0.768095|+0.0285574|+3.862%|
|500|val32 supported-GT recall50|0.041958|0.131034|+0.0890764|+212.299%|
|500|val32 context ca-R50|0|0.00613497|+0.00613497|null|
|500|val32 context thing mIoU|0.00256846|0.00155838|-0.00101008|-39.326%|
|500|val32 context class-aware R50|0|0|+0|null|
|1000|L6 q-out PR|2.10419|2.342|+0.237816|+11.302%|
|1000|L6 q-out cosine p90|0.999131|0.966805|-0.032326|-3.235%|
|1000|L6 GT best Dice|0.184053|0.241536|+0.0574831|+31.232%|
|1000|L12 GT hard-correct|0.332044|0.2133|-0.118744|-35.762%|
|1000|L12 best-query effective Q|2.64534|2.50906|-0.136284|-5.152%|
|1000|train1024 unique matched queries|42|56|+14|+33.333%|
|1000|train1024 effective query count|10.9064|10.9918|+0.0854172|+0.783%|
|1000|train1024 top5 match share|0.74323|0.763171|+0.0199409|+2.683%|
|1000|val32 supported-GT recall50|0.118056|0.12766|+0.00960402|+8.135%|
|1000|val32 context ca-R50|0.0122699|0|-0.0122699|-100.000%|
|1000|val32 context thing mIoU|0.0130845|0.00429234|-0.00879212|-67.195%|
|1000|val32 context class-aware R50|0.0122699|0|-0.0122699|-100.000%|

## Table 7 — Query update ratios

|Step|Arm|Layer|Observed update/candidate delta mean|max ratio error|
|---:|---|---:|---:|---:|
|0|control|6|1.000000|0|
|0|control|8|1.000000|0|
|0|control|10|1.000000|0|
|0|control|12|1.000000|0|
|0|ru|6|0.100000|1.49e-08|
|0|ru|8|0.100000|1.49e-08|
|0|ru|10|0.100000|1.49e-08|
|0|ru|12|0.100000|1.49e-08|
|200|control|6|1.000000|0|
|200|control|8|1.000000|0|
|200|control|10|1.000000|0|
|200|control|12|1.000000|0|
|200|ru|6|0.100000|1.49e-08|
|200|ru|8|0.100000|1.49e-08|
|200|ru|10|0.100000|1.49e-08|
|200|ru|12|0.100000|1.49e-08|
|500|control|6|1.000000|0|
|500|control|8|1.000000|0|
|500|control|10|1.000000|0|
|500|control|12|1.000000|0|
|500|ru|6|0.100000|1.49e-08|
|500|ru|8|0.100000|1.49e-08|
|500|ru|10|0.100000|1.49e-08|
|500|ru|12|0.100000|1.49e-08|
|1000|control|6|1.000000|0|
|1000|control|8|1.000000|0|
|1000|control|10|1.000000|0|
|1000|control|12|1.000000|0|
|1000|ru|6|0.100000|1.49e-08|
|1000|ru|8|0.100000|1.49e-08|
|1000|ru|10|0.100000|1.49e-08|
|1000|ru|12|0.100000|1.49e-08|

## Classification

**RU-C** — gamma=0.1 residual interpolation is insufficient to prevent slot representation collapse.

At step1000 RU layer6 q-in PR was 70.0173, q-out PR 2.3420 (retention 0.0334). RU q-out PR is directionally higher than Control, but remains much lower than q-in, so query representation still rapidly collapses under the preregistered qualitative outcome definition.

Step1000 directional evidence: slot_diversity_directionally_better_than_control=True, slot_diversity_preserved=False, ru_layer6_qin_pr=70.01727594834779, ru_layer6_qout_pr_retention=0.033448912241061814, gt_specialization_improved=False, slot_utilization_improved=False, grouping_improved=False, all_grouping_metrics_lower=False.

No 5k or follow-up experiment was started.
