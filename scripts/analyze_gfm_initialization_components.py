"""CPU-only component and matched/unmatched gradient analysis of saved audit records."""
import argparse,json,pathlib,numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
root=args.output;summary=json.loads((root/'summary.json').read_text());extra={}
for stage in ['zero','epoch1','epoch2','epoch4']:
 data={};gradroles={};layerclasses={};regions=[]
 def extend(k,v):data.setdefault(k,[]).extend(v)
 for rank in range(4):
  r=root/stage/f'rank{rank:02}';done=json.loads((r/'COMPLETE.json').read_text())
  for idx in done['records']:
   d=json.loads((r/str(idx)/'native.json').read_text());n=d['proposal_queries'];init=d['initial_components']
   for k in ['content_norm','geometry_encoding_norm','learned_base_norm','content_geometry_cosine','seed_learned_cosine']:extend(k,init[k])
   reg=init.get('geometric_pooling_regions',[]);regions+=reg
   for l,s in d['layers'].items():
    for name in ['scales_over_scene','anchors_within_3sigma','attention_mass_within_3sigma','attention_effective_anchors','attention_entropy']:
     v=np.asarray(s[name]);extend(l+'_'+name+'_SAM',v[:n].ravel().tolist());extend(l+'_'+name+'_fallback',v[n:].ravel().tolist())
    pc=np.asarray(s['class_probabilities']);layerclasses[l+'_SAM_total']=layerclasses.get(l+'_SAM_total',0)+n;layerclasses[l+'_SAM_noobj']=layerclasses.get(l+'_SAM_noobj',0)+int((pc[:n].argmax(1)==18).sum())
   nobj=np.asarray(d['actual_initial_q_class']).argmax(1)==18;layerclasses['initial_SAM_noobj']=layerclasses.get('initial_SAM_noobj',0)+int(nobj[:n].sum())
  for i in done['training_gradient_windows']:
   g=json.loads((r/f'train_gradient_{i:02}.json').read_text());n=g['proposal_queries'];matched=np.zeros(100,bool);matched[g['matched_query']]=1
   for comp,values in g['gradients'].items():
    q=np.asarray(values['q_pre']['per_query'])
    for sam,typ in [(True,'SAM'),(False,'fallback')]:
     select=np.arange(100)<n if sam else np.arange(100)>=n
     for positive,name in [(True,'matched'),(False,'unmatched')]:
      mask=select&(matched if positive else ~matched);gradroles.setdefault(comp+'_'+typ+'_'+name,[]).extend(q[mask].tolist())
 data_s={k:dict(N=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.percentile(v,90))) for k,v in data.items() if len(v)}
 regionstats={k:dict(mean=float(np.mean([r[k] for r in regions])),median=float(np.median([r[k] for r in regions])),p90=float(np.percentile([r[k] for r in regions],90))) for k in ['sigmoid_weight_mean','sigmoid_weight_saturated_fraction','shape_projection_norm','weighted_semantic_norm','shape_semantic_cosine']}
 extra[stage]=dict(stats=data_s,geometric_pooling=regionstats,layer_class_counts=layerclasses,gradient_roles={k:dict(N=len(v),mean=float(np.mean(v)) if v else None) for k,v in gradroles.items()})
 print(stage,'components',{k:data_s[k]['median'] for k in ['content_norm','geometry_encoding_norm','learned_base_norm','seed_learned_cosine']});print('pool',regionstats)
 print('L6',{k:data_s[k] for k in ['L6_anchors_within_3sigma_SAM','L6_anchors_within_3sigma_fallback','L6_attention_effective_anchors_SAM','L6_attention_effective_anchors_fallback','L6_attention_mass_within_3sigma_SAM']});print('class',layerclasses);print('gradroles',extra[stage]['gradient_roles'])
tracks=summary['reference_candidate_identity_tracks']
for stage in ['zero','epoch1','epoch2','epoch4']:
 present=[t['stages'][stage] for t in tracks if t['stages'][stage]['candidate_present']]
 print('fixed_candidate',stage,'present',len(present),'raw50',sum(r['IoU']['raw']>=.5 for r in present),'initial_P_noobj_mean',np.mean([r['no_object_probability']['initial'] for r in present]),'final_P_noobj_mean',np.mean([r['no_object_probability']['final'] for r in present]))
(root/'detailed_component_audit.json').write_text(json.dumps(extra,indent=2,allow_nan=False)+'\n')
