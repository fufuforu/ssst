"""Fixed targets and deterministic additional RGB contexts; stdlib only."""
import bisect,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
PROTOCOL='qpanchor1_2vs4_context_same_supervision_ft1_v1'
def key(row):return row['scene']+'|'+','.join(map(str,row['context']))+'|'+','.join(map(str,row['novel']))
def select_extra(row,available):
    context=list(row['context']);novel=list(row['novel']);assert len(context)==2 and context[0]<context[1]
    forbidden=set(context+novel);pool=[f for f in available if f not in forbidden]
    if len(pool)<2:raise ValueError('Fewer than two additional non-target source frames: '+key(row))
    inner=[f for f in pool if context[0]<f<context[1]];fallback=len(inner)<2
    candidates=pool if fallback else inner
    chosen=[]
    for fraction in (1/3,2/3):
        target=context[0]+(context[1]-context[0])*fraction
        chosen.append(min((f for f in candidates if f not in chosen),key=lambda f:(abs(f-target),f)))
    chosen=sorted(chosen);assert len(set(chosen))==2 and not set(chosen)&forbidden
    return chosen,fallback
_TABLE=None
def additions(row):
    global _TABLE
    if _TABLE is None:_TABLE=json.loads((BASE/'view_manifest.json').read_text())['additional_contexts']
    extra=_TABLE[key(row)];assert len(extra)==2 and not set(extra)&set(row['context']+row['novel'])
    return extra
