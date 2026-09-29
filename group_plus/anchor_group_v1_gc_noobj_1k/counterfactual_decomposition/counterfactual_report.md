# Projection + Evidence Aggregation + GRU/FFN Counterfactual Decomposition Audit

Read-only audit over five model states and the locked fixed16 windows. Each state/window used exactly one production context forward; all counterfactual branches reused that forward's saved tensors.

- Contracts: 15/15 PASS.
- Production forwards: 80; target builds: 80.
- Backward / autograd.grad / optimizer construction / optimizer step: 0 / 0 / 0 / 0.
- Branches are mathematical replays under `torch.no_grad()`; no branch is a trained model or performance result.

## Table 1 — Projection decomposition (medians of per-window metrics)

|State|Layer|Stage|Cos p90|PR rank|Entropy rank|Feature variance|
|---|---|---|---|---|---|---|
|fresh_step0|6|q_raw|0.0819|71.0176|81.3460|0.0004|
|fresh_step0|6|q_ln|0.0815|71.4028|81.6024|0.9637|
|fresh_step0|6|u_linear|0.3385|12.7112|14.1421|1.8616|
|fresh_step0|6|u_norm|0.3385|13.2916|14.4966|0.0617|
|fresh_step0|8|q_raw|0.9689|2.5087|3.9625|0.2269|
|fresh_step0|8|q_ln|0.9689|2.5087|3.9625|0.2269|
|fresh_step0|8|u_linear|0.9736|2.7026|3.7591|0.3463|
|fresh_step0|8|u_norm|0.9736|2.6315|3.8132|0.0139|
|fresh_step0|10|q_raw|0.9672|2.8612|4.6709|0.2069|
|fresh_step0|10|q_ln|0.9672|2.8612|4.6709|0.2069|
|fresh_step0|10|u_linear|0.9793|2.3639|3.6310|0.3612|
|fresh_step0|10|u_norm|0.9793|2.4670|3.7129|0.0115|
|fresh_step0|12|q_raw|0.9657|3.0398|4.8877|0.2053|
|fresh_step0|12|q_ln|0.9657|3.0398|4.8877|0.2053|
|fresh_step0|12|u_linear|0.9765|2.1823|3.4337|0.3529|
|fresh_step0|12|u_norm|0.9765|2.2002|3.3883|0.0107|
|control_step500|6|q_raw|0.1022|70.7541|81.1848|0.0004|
|control_step500|6|q_ln|0.1025|71.1212|81.4329|0.9477|
|control_step500|6|u_linear|0.6457|11.4788|13.4167|1.6316|
|control_step500|6|u_norm|0.6457|11.4983|13.5163|0.0446|
|control_step500|8|q_raw|0.9961|2.1887|3.2635|0.1006|
|control_step500|8|q_ln|0.9961|2.1883|3.2625|0.1006|
|control_step500|8|u_linear|0.9970|2.0144|2.7792|0.2324|
|control_step500|8|u_norm|0.9970|1.9379|2.7327|0.0054|
|control_step500|10|q_raw|0.9977|2.0787|3.0888|0.0875|
|control_step500|10|q_ln|0.9977|2.0784|3.0882|0.0874|
|control_step500|10|u_linear|0.9981|1.6727|2.2942|0.3546|
|control_step500|10|u_norm|0.9981|1.6849|2.3418|0.0062|
|control_step500|12|q_raw|0.9985|1.7881|2.7134|0.0827|
|control_step500|12|q_ln|0.9985|1.7879|2.7131|0.0826|
|control_step500|12|u_linear|0.9992|1.5155|2.0653|0.4113|
|control_step500|12|u_norm|0.9992|1.5286|2.0843|0.0063|
|ablation_step500|6|q_raw|0.0934|71.0515|81.3954|0.0004|
|ablation_step500|6|q_ln|0.0930|71.4341|81.6485|0.9555|
|ablation_step500|6|u_linear|0.5072|12.6386|14.0797|1.8053|
|ablation_step500|6|u_norm|0.5072|13.0695|14.3619|0.0515|
|ablation_step500|8|q_raw|0.9810|2.7910|4.1583|0.1699|
|ablation_step500|8|q_ln|0.9810|2.7909|4.1577|0.1698|
|ablation_step500|8|u_linear|0.9808|2.4225|3.4332|0.4009|
|ablation_step500|8|u_norm|0.9808|2.5972|3.5361|0.0126|
|ablation_step500|10|q_raw|0.9877|2.8236|4.1856|0.1454|
|ablation_step500|10|q_ln|0.9877|2.8236|4.1853|0.1453|
|ablation_step500|10|u_linear|0.9895|2.3316|3.1906|0.5719|
|ablation_step500|10|u_norm|0.9895|2.4919|3.2912|0.0102|
|ablation_step500|12|q_raw|0.9909|2.7712|4.0525|0.1323|
|ablation_step500|12|q_ln|0.9909|2.7710|4.0519|0.1321|
|ablation_step500|12|u_linear|0.9947|1.9415|2.7230|0.7173|
|ablation_step500|12|u_norm|0.9947|2.1284|2.8985|0.0096|
|control_step1000|6|q_raw|0.1474|69.9667|80.6425|0.0004|
|control_step1000|6|q_ln|0.1495|70.3422|80.8996|0.9107|
|control_step1000|6|u_linear|0.9467|5.3619|9.0971|1.0477|
|control_step1000|6|u_norm|0.9467|5.4372|9.2104|0.0218|
|control_step1000|8|q_raw|0.9991|1.9742|2.9002|0.1006|
|control_step1000|8|q_ln|0.9991|1.9725|2.8980|0.1006|
|control_step1000|8|u_linear|0.9994|1.6350|2.2263|0.3701|
|control_step1000|8|u_norm|0.9994|1.5832|2.1675|0.0085|
|control_step1000|10|q_raw|0.9995|1.8856|2.7287|0.0995|
|control_step1000|10|q_ln|0.9995|1.8838|2.7266|0.0994|
|control_step1000|10|u_linear|0.9997|1.4541|1.9410|0.6574|
|control_step1000|10|u_norm|0.9997|1.4187|1.8904|0.0092|
|control_step1000|12|q_raw|0.9996|1.8239|2.5825|0.1057|
|control_step1000|12|q_ln|0.9996|1.8224|2.5809|0.1056|
|control_step1000|12|u_linear|0.9998|1.4971|1.9434|0.7898|
|control_step1000|12|u_norm|0.9998|1.4157|1.8431|0.0091|
|ablation_step1000|6|q_raw|0.1089|71.0879|81.4197|0.0004|
|ablation_step1000|6|q_ln|0.1099|71.4783|81.6737|0.9408|
|ablation_step1000|6|u_linear|0.6946|11.5357|13.4893|1.6709|
|ablation_step1000|6|u_norm|0.6946|11.9739|13.7381|0.0393|
|ablation_step1000|8|q_raw|0.9933|2.6623|3.9038|0.1342|
|ablation_step1000|8|q_ln|0.9933|2.6620|3.9032|0.1341|
|ablation_step1000|8|u_linear|0.9953|2.1876|3.2222|0.3926|
|ablation_step1000|8|u_norm|0.9953|2.3194|3.3441|0.0089|
|ablation_step1000|10|q_raw|0.9947|2.6428|4.2125|0.1238|
|ablation_step1000|10|q_ln|0.9947|2.6424|4.2125|0.1237|
|ablation_step1000|10|u_linear|0.9980|1.7639|2.6330|0.6098|
|ablation_step1000|10|u_norm|0.9980|1.8424|2.7705|0.0069|
|ablation_step1000|12|q_raw|0.9957|2.3758|3.8639|0.1214|
|ablation_step1000|12|q_ln|0.9957|2.3753|3.8629|0.1213|
|ablation_step1000|12|u_linear|0.9985|1.5726|2.2654|0.7259|
|ablation_step1000|12|u_norm|0.9985|1.6117|2.3414|0.0072|

## proj_u weight singular spectrum (same matrix across layers)

|State|Singular PR|Entropy rank|Top1 energy|Top4 energy|Top8 energy|Condition|
|---|---|---|---|---|---|---|
|fresh_step0|14.8955|15.4177|0.0963|0.3403|0.6130|1.6372|
|control_step500|14.8917|15.4156|0.0964|0.3405|0.6134|1.6382|
|ablation_step500|14.8893|15.4144|0.0963|0.3405|0.6135|1.6392|
|control_step1000|14.9047|15.4230|0.0957|0.3401|0.6128|1.6312|
|ablation_step1000|14.8962|15.4182|0.0960|0.3401|0.6132|1.6332|

The projection output dimension is 16, so its theoretical rank ceiling is 16. Raw-q PR near 70 versus u PR must be interpreted relative to that ceiling; low empirical u rank below 16 and a concentrated weight singular spectrum are the evidence relevant to learned compression.

## Table 2 — Local layer-6 counterfactual branches

|State|Branch|z cos p90|z PR|GRU PR|q_out PR|q_out cos p90|Apost effQ|GT Dice|GT bestQ effQ|
|---|---|---|---|---|---|---|---|---|---|
|fresh_step0|PROD|0.9765|2.2743|2.5464|2.5087|0.9689|75.9605|0.0848|2.9142|
|fresh_step0|ZERO|0.0000|0.0000|64.6628|60.8187|0.1390|32.8032|0.2007|2.8284|
|fresh_step0|GLOBAL_MEAN|1.0000|0.0000|61.6777|54.5142|0.9987|99.7215|0.0410|3.0000|
|fresh_step0|QUERY_CENTRIC|0.9779|2.2380|2.5903|2.5739|0.9702|75.5993|0.0863|2.8284|
|control_step500|PROD|0.9958|2.0360|2.2788|2.1887|0.9961|51.1916|0.1547|2.0000|
|control_step500|ZERO|0.0000|0.0000|64.6012|60.6360|0.1669|9.4122|0.2282|2.8284|
|control_step500|GLOBAL_MEAN|1.0000|0.0000|61.0622|54.3550|0.9991|99.8715|0.0415|2.6679|
|control_step500|QUERY_CENTRIC|0.9968|1.6958|1.8661|1.8328|0.9965|50.3964|0.1503|2.0000|
|ablation_step500|PROD|0.9785|2.4209|2.8218|2.7910|0.9810|61.3525|0.1357|2.9142|
|ablation_step500|ZERO|0.0000|0.0000|64.7925|60.8781|0.1565|12.7697|0.2181|2.9142|
|ablation_step500|GLOBAL_MEAN|1.0000|0.0000|61.2192|54.6361|0.9990|99.8573|0.0394|2.0000|
|ablation_step500|QUERY_CENTRIC|0.9903|2.0983|2.5444|2.5794|0.9890|47.1638|0.1335|2.8284|
|control_step1000|PROD|0.9997|1.8068|2.1008|1.9742|0.9991|19.5728|0.1986|2.0000|
|control_step1000|ZERO|0.0000|0.0000|64.1836|59.9455|0.1967|4.5143|0.2473|2.0000|
|control_step1000|GLOBAL_MEAN|1.0000|0.0000|60.4153|53.7370|0.9991|99.9320|0.0414|2.0000|
|control_step1000|QUERY_CENTRIC|0.9998|1.6350|1.9442|1.9009|0.9991|19.3906|0.1994|2.0000|
|ablation_step1000|PROD|0.9936|2.3294|2.7703|2.6623|0.9933|39.5938|0.1866|2.7889|
|ablation_step1000|ZERO|0.0000|0.0000|64.8824|60.9535|0.1773|10.9962|0.2389|2.9142|
|ablation_step1000|GLOBAL_MEAN|1.0000|0.0000|61.1761|54.7441|0.9991|99.8952|0.0396|2.9142|
|ablation_step1000|QUERY_CENTRIC|0.9941|2.1837|2.6520|2.5307|0.9936|26.5632|0.2035|2.6679|

## Table 3 — Evidence-weight geometry

|State|Layer|Weights|Weight cos p90|Eff anchors|Entropy|Radius / ell|e resultant|
|---|---|---|---|---|---|---|---|
|fresh_step0|6|PROD|0.5667|303.2071|0.8930|0.9853|0.8362|
|fresh_step0|6|QUERY_CENTRIC|0.5751|303.5417|0.8956|0.9775|0.8442|
|fresh_step0|8|PROD|0.9192|693.7096|0.9673|1.0036|0.5779|
|fresh_step0|8|QUERY_CENTRIC|0.9596|128.7739|0.7953|0.8028|0.7130|
|fresh_step0|10|PROD|0.9361|755.7774|0.9755|1.0203|0.5100|
|fresh_step0|10|QUERY_CENTRIC|0.9607|131.3184|0.8177|0.9243|0.6313|
|fresh_step0|12|PROD|0.9606|853.5599|0.9858|0.9824|0.7630|
|fresh_step0|12|QUERY_CENTRIC|0.9754|448.8588|0.9297|0.8545|0.8341|
|control_step500|6|PROD|0.9266|53.8700|0.6785|1.0580|0.9034|
|control_step500|6|QUERY_CENTRIC|0.9450|94.6778|0.7744|1.0453|0.8835|
|control_step500|8|PROD|0.9913|592.1722|0.9406|1.0018|0.6983|
|control_step500|8|QUERY_CENTRIC|0.9945|161.9059|0.7984|0.9082|0.7834|
|control_step500|10|PROD|0.9941|515.3294|0.9300|1.0226|0.6846|
|control_step500|10|QUERY_CENTRIC|0.9960|56.7242|0.7472|0.8148|0.6564|
|control_step500|12|PROD|0.9996|173.8948|0.8248|1.0146|0.7879|
|control_step500|12|QUERY_CENTRIC|0.9994|339.6988|0.9111|0.9364|0.7121|
|ablation_step500|6|PROD|0.7837|61.1975|0.7326|1.1299|0.7544|
|ablation_step500|6|QUERY_CENTRIC|0.8456|157.4859|0.8029|1.0037|0.8066|
|ablation_step500|8|PROD|0.9506|479.1083|0.9237|0.9776|0.5973|
|ablation_step500|8|QUERY_CENTRIC|0.9598|87.5309|0.7696|0.7777|0.6717|
|ablation_step500|10|PROD|0.9624|317.7952|0.8848|0.9665|0.4570|
|ablation_step500|10|QUERY_CENTRIC|0.9818|45.0625|0.6984|0.6721|0.5885|
|ablation_step500|12|PROD|0.9962|123.8147|0.7976|0.8522|0.7982|
|ablation_step500|12|QUERY_CENTRIC|0.9916|271.4856|0.8884|0.7590|0.6851|
|control_step1000|6|PROD|0.9930|21.6335|0.5186|1.2663|0.8689|
|control_step1000|6|QUERY_CENTRIC|0.9971|10.5690|0.4739|1.1487|0.8831|
|control_step1000|8|PROD|0.9984|285.5924|0.8594|1.0496|0.5464|
|control_step1000|8|QUERY_CENTRIC|0.9992|123.4147|0.8079|1.0018|0.6954|
|control_step1000|10|PROD|0.9992|139.7570|0.7749|0.9845|0.5912|
|control_step1000|10|QUERY_CENTRIC|0.9993|56.5637|0.7154|0.8621|0.5766|
|control_step1000|12|PROD|0.9999|92.9624|0.7680|0.9732|0.7021|
|control_step1000|12|QUERY_CENTRIC|0.9997|288.8789|0.9094|0.9845|0.7107|
|ablation_step1000|6|PROD|0.9208|44.2858|0.6519|1.2617|0.8001|
|ablation_step1000|6|QUERY_CENTRIC|0.9402|49.9664|0.7056|1.1615|0.7763|
|ablation_step1000|8|PROD|0.9887|160.7948|0.8087|0.7000|0.6637|
|ablation_step1000|8|QUERY_CENTRIC|0.9930|50.3797|0.7080|0.5457|0.7705|
|ablation_step1000|10|PROD|0.9934|142.4182|0.8255|1.0555|0.4482|
|ablation_step1000|10|QUERY_CENTRIC|0.9963|84.1809|0.7610|0.7065|0.7119|
|ablation_step1000|12|PROD|0.9971|196.8995|0.8780|1.0682|0.6668|
|ablation_step1000|12|QUERY_CENTRIC|0.9960|364.0665|0.9160|0.9490|0.7005|

## Table 4 — GRU versus FFN contraction

|State|Layer|Branch|q_in PR|GRU PR|FFN/final PR|GRU ratio|FFN ratio|
|---|---|---|---|---|---|---|---|
|fresh_step0|6|PROD|71.0176|2.5464|2.5087|0.0359|1.0058|
|fresh_step0|6|ZERO|71.0176|64.6628|60.8187|0.9105|0.9406|
|fresh_step0|6|GLOBAL_MEAN|71.0176|61.6777|54.5142|0.8685|0.8841|
|fresh_step0|6|QUERY_CENTRIC|71.0176|2.5903|2.5739|0.0365|1.0080|
|fresh_step0|8|PROD|2.5087|2.8182|2.8612|1.0825|1.0162|
|fresh_step0|8|ZERO|2.5087|2.7192|2.8233|1.0648|1.0327|
|fresh_step0|8|GLOBAL_MEAN|2.5087|2.6709|2.7988|1.0441|1.0349|
|fresh_step0|8|QUERY_CENTRIC|2.5087|2.7607|2.8488|1.0670|1.0230|
|fresh_step0|10|PROD|2.8612|3.0193|3.0398|1.0553|1.0032|
|fresh_step0|10|ZERO|2.8612|2.9779|3.0053|1.0249|1.0113|
|fresh_step0|10|GLOBAL_MEAN|2.8612|2.9626|3.0102|1.0236|1.0092|
|fresh_step0|10|QUERY_CENTRIC|2.8612|3.0241|3.0282|1.0446|1.0022|
|fresh_step0|12|PROD|3.0398|3.1239|3.1112|1.0307|0.9960|
|fresh_step0|12|ZERO|3.0398|3.0508|3.0104|1.0211|1.0001|
|fresh_step0|12|GLOBAL_MEAN|3.0398|3.0378|3.0343|1.0146|0.9940|
|fresh_step0|12|QUERY_CENTRIC|3.0398|3.1012|3.0980|1.0220|0.9968|
|control_step500|6|PROD|70.7541|2.2788|2.1887|0.0322|0.9598|
|control_step500|6|ZERO|70.7541|64.6012|60.6360|0.9130|0.9386|
|control_step500|6|GLOBAL_MEAN|70.7541|61.0622|54.3550|0.8630|0.8879|
|control_step500|6|QUERY_CENTRIC|70.7541|1.8661|1.8328|0.0264|0.9702|
|control_step500|8|PROD|2.1887|2.3489|2.0787|1.0476|0.9014|
|control_step500|8|ZERO|2.1887|2.3128|2.1207|1.0407|0.9342|
|control_step500|8|GLOBAL_MEAN|2.1887|2.2213|2.0117|1.0078|0.9086|
|control_step500|8|QUERY_CENTRIC|2.1887|2.2831|2.0618|1.0225|0.9073|
|control_step500|10|PROD|2.0787|2.0621|1.7881|0.9919|0.8820|
|control_step500|10|ZERO|2.0787|2.0389|1.8204|0.9819|0.9153|
|control_step500|10|GLOBAL_MEAN|2.0787|1.9845|1.7521|0.9670|0.9003|
|control_step500|10|QUERY_CENTRIC|2.0787|2.0397|1.7842|0.9821|0.8936|
|control_step500|12|PROD|1.7881|1.7621|1.6148|0.9956|0.9185|
|control_step500|12|ZERO|1.7881|1.7680|1.6202|0.9971|0.9175|
|control_step500|12|GLOBAL_MEAN|1.7881|1.7269|1.5658|0.9783|0.9050|
|control_step500|12|QUERY_CENTRIC|1.7881|1.7855|1.6210|1.0054|0.9067|
|ablation_step500|6|PROD|71.0515|2.8218|2.7910|0.0397|0.9898|
|ablation_step500|6|ZERO|71.0515|64.7925|60.8781|0.9119|0.9396|
|ablation_step500|6|GLOBAL_MEAN|71.0515|61.2192|54.6361|0.8616|0.8896|
|ablation_step500|6|QUERY_CENTRIC|71.0515|2.5444|2.5794|0.0358|1.0018|
|ablation_step500|8|PROD|2.7910|3.1170|2.8236|1.0929|0.9146|
|ablation_step500|8|ZERO|2.7910|2.8995|2.7703|1.0428|0.9468|
|ablation_step500|8|GLOBAL_MEAN|2.7910|2.8852|2.7087|1.0410|0.9318|
|ablation_step500|8|QUERY_CENTRIC|2.7910|3.0105|2.7898|1.0663|0.9266|
|ablation_step500|10|PROD|2.8236|3.0891|2.7712|1.0613|0.8897|
|ablation_step500|10|ZERO|2.8236|2.8182|2.6295|0.9913|0.9116|
|ablation_step500|10|GLOBAL_MEAN|2.8236|2.8069|2.5644|0.9761|0.9015|
|ablation_step500|10|QUERY_CENTRIC|2.8236|2.9778|2.6926|1.0272|0.9034|
|ablation_step500|12|PROD|2.7712|2.8519|2.6982|1.0409|0.9319|
|ablation_step500|12|ZERO|2.7712|2.7278|2.5082|0.9873|0.9179|
|ablation_step500|12|GLOBAL_MEAN|2.7712|2.7441|2.5124|0.9890|0.9149|
|ablation_step500|12|QUERY_CENTRIC|2.7712|2.8777|2.7166|1.0511|0.9227|
|control_step1000|6|PROD|69.9667|2.1008|1.9742|0.0300|0.9435|
|control_step1000|6|ZERO|69.9667|64.1836|59.9455|0.9173|0.9340|
|control_step1000|6|GLOBAL_MEAN|69.9667|60.4153|53.7370|0.8635|0.8887|
|control_step1000|6|QUERY_CENTRIC|69.9667|1.9442|1.9009|0.0278|0.9619|
|control_step1000|8|PROD|1.9742|2.1001|1.8856|1.0465|0.9312|
|control_step1000|8|ZERO|1.9742|2.0215|1.8784|1.0323|0.9445|
|control_step1000|8|GLOBAL_MEAN|1.9742|1.9200|1.7958|0.9807|0.9401|
|control_step1000|8|QUERY_CENTRIC|1.9742|2.0394|1.8836|1.0197|0.9402|
|control_step1000|10|PROD|1.8856|1.9187|1.8239|1.0148|0.9573|
|control_step1000|10|ZERO|1.8856|1.9259|1.8344|1.0215|0.9472|
|control_step1000|10|GLOBAL_MEAN|1.8856|1.8430|1.7415|0.9838|0.9400|
|control_step1000|10|QUERY_CENTRIC|1.8856|1.8929|1.8030|1.0054|0.9618|
|control_step1000|12|PROD|1.8239|1.7914|1.7069|0.9669|0.9576|
|control_step1000|12|ZERO|1.8239|1.8806|1.8180|1.0297|0.9655|
|control_step1000|12|GLOBAL_MEAN|1.8239|1.7608|1.6762|0.9654|0.9491|
|control_step1000|12|QUERY_CENTRIC|1.8239|1.7835|1.6923|0.9718|0.9539|
|ablation_step1000|6|PROD|71.0879|2.7703|2.6623|0.0390|0.9325|
|ablation_step1000|6|ZERO|71.0879|64.8824|60.9535|0.9127|0.9394|
|ablation_step1000|6|GLOBAL_MEAN|71.0879|61.1761|54.7441|0.8606|0.8929|
|ablation_step1000|6|QUERY_CENTRIC|71.0879|2.6520|2.5307|0.0373|0.9508|
|ablation_step1000|8|PROD|2.6623|3.0981|2.6428|1.1764|0.8639|
|ablation_step1000|8|ZERO|2.6623|2.7617|2.4299|1.0492|0.9074|
|ablation_step1000|8|GLOBAL_MEAN|2.6623|2.6884|2.3287|1.0219|0.8925|
|ablation_step1000|8|QUERY_CENTRIC|2.6623|2.9029|2.4554|1.1120|0.8742|
|ablation_step1000|10|PROD|2.6428|2.7643|2.3758|1.0523|0.8750|
|ablation_step1000|10|ZERO|2.6428|2.6005|2.2451|0.9989|0.8834|
|ablation_step1000|10|GLOBAL_MEAN|2.6428|2.5719|2.2184|0.9845|0.8878|
|ablation_step1000|10|QUERY_CENTRIC|2.6428|2.6902|2.3238|1.0351|0.8758|
|ablation_step1000|12|PROD|2.3758|2.5606|2.3330|1.0943|0.9097|
|ablation_step1000|12|ZERO|2.3758|2.4023|2.1422|1.0111|0.8949|
|ablation_step1000|12|GLOBAL_MEAN|2.3758|2.3961|2.1713|1.0140|0.9034|
|ablation_step1000|12|QUERY_CENTRIC|2.3758|2.5598|2.3179|1.0792|0.9050|

## Table 5 — Sequential four-layer counterfactual replay

|State|Branch|Layer|q PR|z PR|Apost effQ|GT best Dice|GT bestQ effQ|
|---|---|---|---|---|---|---|---|
|fresh_step0|PROD|6|2.5087|2.2743|75.9605|0.0848|2.9142|
|fresh_step0|PROD|8|2.8612|2.5139|91.8384|0.0689|3.2321|
|fresh_step0|PROD|10|3.0398|2.7286|93.3254|0.0626|3.0000|
|fresh_step0|PROD|12|3.1112|1.4486|84.4994|0.0673|2.8284|
|fresh_step0|ZERO|6|60.8187|0.0000|32.8032|0.2007|2.8284|
|fresh_step0|ZERO|8|50.7570|0.0000|63.5502|0.1534|3.0000|
|fresh_step0|ZERO|10|44.2888|0.0000|62.8548|0.1391|3.0000|
|fresh_step0|ZERO|12|39.6681|0.0000|35.8687|0.1687|3.0000|
|fresh_step0|GLOBAL_MEAN|6|54.5142|0.0000|99.7215|0.0410|3.0000|
|fresh_step0|GLOBAL_MEAN|8|39.0297|0.0000|99.8572|0.0414|2.8284|
|fresh_step0|GLOBAL_MEAN|10|28.5654|0.0000|99.8410|0.0409|2.8284|
|fresh_step0|GLOBAL_MEAN|12|21.6101|0.0000|99.7324|0.0409|2.4142|
|fresh_step0|QUERY_CENTRIC|6|2.5739|2.2380|75.5993|0.0863|2.8284|
|fresh_step0|QUERY_CENTRIC|8|2.8856|2.1618|91.2505|0.0653|3.0000|
|fresh_step0|QUERY_CENTRIC|10|3.0035|2.2199|93.6907|0.0627|3.0000|
|fresh_step0|QUERY_CENTRIC|12|3.0510|1.3384|85.4746|0.0657|2.8284|
|control_step500|PROD|6|2.1887|2.0360|51.1916|0.1547|2.0000|
|control_step500|PROD|8|2.0787|1.6179|44.2193|0.1638|2.0000|
|control_step500|PROD|10|1.7881|1.6303|35.6941|0.1873|2.8284|
|control_step500|PROD|12|1.6148|1.0894|4.9116|0.1972|2.8284|
|control_step500|ZERO|6|60.6360|0.0000|9.4122|0.2282|2.8284|
|control_step500|ZERO|8|50.3235|0.0000|49.7922|0.1766|2.8284|
|control_step500|ZERO|10|43.0290|0.0000|42.9865|0.1990|2.8284|
|control_step500|ZERO|12|37.3490|0.0000|29.4453|0.2023|2.4142|
|control_step500|GLOBAL_MEAN|6|54.3550|0.0000|99.8715|0.0415|2.6679|
|control_step500|GLOBAL_MEAN|8|36.8434|0.0000|99.9770|0.0400|2.8284|
|control_step500|GLOBAL_MEAN|10|25.5310|0.0000|99.9826|0.0396|2.8284|
|control_step500|GLOBAL_MEAN|12|18.3141|0.0000|99.9761|0.0396|2.5000|
|control_step500|QUERY_CENTRIC|6|1.8328|1.6958|50.3964|0.1503|2.0000|
|control_step500|QUERY_CENTRIC|8|1.6744|1.2443|42.5493|0.1514|3.0000|
|control_step500|QUERY_CENTRIC|10|1.5067|1.3431|36.2621|0.1655|2.0000|
|control_step500|QUERY_CENTRIC|12|1.4441|1.0942|8.1835|0.1866|2.7889|
|ablation_step500|PROD|6|2.7910|2.4209|61.3525|0.1357|2.9142|
|ablation_step500|PROD|8|2.8236|1.8670|19.3682|0.1725|2.7889|
|ablation_step500|PROD|10|2.7712|1.8075|18.3188|0.1835|2.0000|
|ablation_step500|PROD|12|2.6982|1.3437|4.5020|0.2088|2.4142|
|ablation_step500|ZERO|6|60.8781|0.0000|12.7697|0.2181|2.9142|
|ablation_step500|ZERO|8|50.4083|0.0000|46.9954|0.1831|3.0000|
|ablation_step500|ZERO|10|42.9716|0.0000|47.4883|0.1892|2.9142|
|ablation_step500|ZERO|12|37.2718|0.0000|32.3388|0.2081|3.0000|
|ablation_step500|GLOBAL_MEAN|6|54.6361|0.0000|99.8573|0.0394|2.0000|
|ablation_step500|GLOBAL_MEAN|8|38.0512|0.0000|99.9622|0.0365|2.9142|
|ablation_step500|GLOBAL_MEAN|10|26.4001|0.0000|99.9631|0.0364|2.8284|
|ablation_step500|GLOBAL_MEAN|12|18.5715|0.0000|99.9710|0.0370|2.0000|
|ablation_step500|QUERY_CENTRIC|6|2.5794|2.0983|47.1638|0.1335|2.8284|
|ablation_step500|QUERY_CENTRIC|8|2.5963|1.9431|21.8386|0.1773|2.7889|
|ablation_step500|QUERY_CENTRIC|10|2.4705|2.0337|22.1521|0.1773|2.8284|
|ablation_step500|QUERY_CENTRIC|12|2.5330|1.3137|8.3998|0.1840|2.3747|
|control_step1000|PROD|6|1.9742|1.8068|19.5728|0.1986|2.0000|
|control_step1000|PROD|8|1.8856|1.8882|18.1129|0.1973|2.8284|
|control_step1000|PROD|10|1.8239|1.4726|11.6590|0.1970|2.7889|
|control_step1000|PROD|12|1.7069|1.8550|3.8161|0.2456|2.7889|
|control_step1000|ZERO|6|59.9455|0.0000|4.5143|0.2473|2.0000|
|control_step1000|ZERO|8|49.5053|0.0000|34.8071|0.2024|2.9142|
|control_step1000|ZERO|10|42.2113|0.0000|33.3629|0.1858|2.9142|
|control_step1000|ZERO|12|36.1787|0.0000|21.1191|0.2097|2.8284|
|control_step1000|GLOBAL_MEAN|6|53.7370|0.0000|99.9320|0.0414|2.0000|
|control_step1000|GLOBAL_MEAN|8|36.1559|0.0000|99.9879|0.0395|2.8284|
|control_step1000|GLOBAL_MEAN|10|23.1341|0.0000|99.9914|0.0392|2.5000|
|control_step1000|GLOBAL_MEAN|12|15.2752|0.0000|99.9959|0.0388|3.0000|
|control_step1000|QUERY_CENTRIC|6|1.9009|1.6350|19.3906|0.1994|2.0000|
|control_step1000|QUERY_CENTRIC|8|1.8445|1.4179|16.9195|0.2191|3.0000|
|control_step1000|QUERY_CENTRIC|10|1.7754|1.3296|13.5046|0.2058|2.8284|
|control_step1000|QUERY_CENTRIC|12|1.6667|1.8157|5.3599|0.2249|2.0000|
|ablation_step1000|PROD|6|2.6623|2.3294|39.5938|0.1866|2.7889|
|ablation_step1000|PROD|8|2.6428|1.9344|9.2832|0.1919|2.8284|
|ablation_step1000|PROD|10|2.3758|2.0805|7.6631|0.2042|2.7889|
|ablation_step1000|PROD|12|2.3330|1.4011|3.8705|0.2377|2.8284|
|ablation_step1000|ZERO|6|60.9535|0.0000|10.9962|0.2389|2.9142|
|ablation_step1000|ZERO|8|50.1657|0.0000|38.6100|0.2123|2.9142|
|ablation_step1000|ZERO|10|42.1703|0.0000|31.7702|0.2371|2.8284|
|ablation_step1000|ZERO|12|35.6257|0.0000|23.3371|0.2253|2.8284|
|ablation_step1000|GLOBAL_MEAN|6|54.7441|0.0000|99.8952|0.0396|2.9142|
|ablation_step1000|GLOBAL_MEAN|8|37.0225|0.0000|99.9849|0.0376|2.9142|
|ablation_step1000|GLOBAL_MEAN|10|24.5584|0.0000|99.9866|0.0375|2.9142|
|ablation_step1000|GLOBAL_MEAN|12|16.6609|0.0000|99.9870|0.0377|2.8284|
|ablation_step1000|QUERY_CENTRIC|6|2.5307|2.1837|26.5632|0.2035|2.6679|
|ablation_step1000|QUERY_CENTRIC|8|2.6210|1.9814|9.1493|0.1912|3.0000|
|ablation_step1000|QUERY_CENTRIC|10|2.5289|1.6873|9.0319|0.1988|2.8284|
|ablation_step1000|QUERY_CENTRIC|12|2.4420|1.3638|5.6763|0.2422|2.8284|

## Localization

Ratings use the fixed diagnostic labels in the audit specification; they describe evidence in these windows/checkpoints, not causal proof or a model recommendation.

- **PD-A_projection_bottleneck: moderate evidence**
  - state=control_step500, q_ln_pr=71.1212, u_linear_pr=11.4788, u_norm_pr=11.4983, proj_u_singular_pr=14.8917, top4_energy=0.3405
  - state=ablation_step500, q_ln_pr=71.4341, u_linear_pr=12.6386, u_norm_pr=13.0695, proj_u_singular_pr=14.8893, top4_energy=0.3405
  - state=control_step1000, q_ln_pr=70.3422, u_linear_pr=5.3619, u_norm_pr=5.4372, proj_u_singular_pr=14.9047, top4_energy=0.3401
  - state=ablation_step1000, q_ln_pr=71.4783, u_linear_pr=11.5357, u_norm_pr=11.9739, proj_u_singular_pr=14.8962, top4_energy=0.3401
- **PD-B_production_aggregation_bottleneck: weak/no evidence**
  - state=control_step500, prod_z_cos_p90=0.9958, qc_z_cos_p90=0.9968, prod_qout_pr=2.1887, qc_qout_pr=1.8328, prod_GT_best_dice=0.1547, qc_GT_best_dice=0.1503
  - state=ablation_step500, prod_z_cos_p90=0.9785, qc_z_cos_p90=0.9903, prod_qout_pr=2.7910, qc_qout_pr=2.5794, prod_GT_best_dice=0.1357, qc_GT_best_dice=0.1335
  - state=control_step1000, prod_z_cos_p90=0.9997, qc_z_cos_p90=0.9998, prod_qout_pr=1.9742, qc_qout_pr=1.9009, prod_GT_best_dice=0.1986, qc_GT_best_dice=0.1994
  - state=ablation_step1000, prod_z_cos_p90=0.9936, qc_z_cos_p90=0.9941, prod_qout_pr=2.6623, qc_qout_pr=2.5307, prod_GT_best_dice=0.1866, qc_GT_best_dice=0.2035
- **PD-C_shared_evidence_overwrite: moderate evidence**
  - state=control_step500, zero_qout_pr=60.6360, global_mean_qout_pr=54.3550, zero_qin_pr=70.7541, zero_qout_cos_p90=0.1669, global_mean_qout_cos_p90=0.9991, zero_GT_best_dice=0.2282, global_mean_GT_best_dice=0.0415
  - state=ablation_step500, zero_qout_pr=60.8781, global_mean_qout_pr=54.6361, zero_qin_pr=71.0515, zero_qout_cos_p90=0.1565, global_mean_qout_cos_p90=0.9990, zero_GT_best_dice=0.2181, global_mean_GT_best_dice=0.0394
  - state=control_step1000, zero_qout_pr=59.9455, global_mean_qout_pr=53.7370, zero_qin_pr=69.9667, zero_qout_cos_p90=0.1967, global_mean_qout_cos_p90=0.9991, zero_GT_best_dice=0.2473, global_mean_GT_best_dice=0.0414
  - state=ablation_step1000, zero_qout_pr=60.9535, global_mean_qout_pr=54.7441, zero_qin_pr=71.0879, zero_qout_cos_p90=0.1773, global_mean_qout_cos_p90=0.9991, zero_GT_best_dice=0.2389, global_mean_GT_best_dice=0.0396
- **PD-D_intrinsic_recurrent_update_contraction: weak/no evidence**
  - state=control_step500, zero_qin_pr=70.7541, zero_vgru_pr=64.6012, zero_qout_pr=60.6360
  - state=ablation_step500, zero_qin_pr=71.0515, zero_vgru_pr=64.7925, zero_qout_pr=60.8781
  - state=control_step1000, zero_qin_pr=69.9667, zero_vgru_pr=64.1836, zero_qout_pr=59.9455
  - state=ablation_step1000, zero_qin_pr=71.0879, zero_vgru_pr=64.8824, zero_qout_pr=60.9535

### Counterfactual answers

- ZERO evidence at layer 6: q_in PR 70.7541 → v_gru PR 64.6012 → q_out PR 60.6360 for Control500.
- GLOBAL_MEAN at layer 6: q_in PR 70.7541 → q_out PR 54.3550 for Control500.
- QUERY_CENTRIC versus PROD at layer 6, Control500: z PR 2.0360 → 1.6958; q_out PR 2.1887 → 1.8328; GT best Dice 0.1547 → 0.1503.
- Direct answers: No. ZERO preserves most rank: Control500 q_in→v_gru→q_out PR is approximately 70.75→64.60→60.64, not 70→2–3. No. At Control1000 layer6 QC versus PROD q_out PR is 1.90 vs 1.97, ownership effective-Q 19.39 vs 19.57, and GT best Dice 0.199 vs 0.199; at Control500 QC does not improve these metrics either.
- Primary localization: **mixed**. Empirical q→u compression is present below the 16-D ceiling, but proj_u's singular spectrum is near full-rank. ZERO rules out severe intrinsic GRU/FFN collapse; shared nonzero evidence sharply raises query cosine. QC does not restore q-out diversity or GT specialization, so changing the production normalization alone is not supported as a remedy.
- These are fixed-checkpoint counterfactual replays only; they are not trained performance estimates.

## Audit boundary

This was the final read-only mechanism decomposition before a structural intervention experiment. No training was run. No backward or autograd.grad was run. No optimizer was constructed. No model, loss, Hungarian, or checkpoint was modified. No corrective mechanism was implemented. No next training experiment was started.
