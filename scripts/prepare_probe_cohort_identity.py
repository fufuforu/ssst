#!/usr/bin/env python3
"""Prepare immutable dev/test cohort identities before parallel A/B jobs."""
import argparse,hashlib,json
from pathlib import Path
DATA=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_v1_8gpu/data_manifest.json')
COHORT=Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/four_arm_evaluation_retry01/cohort_manifest.json')
EXPECTED='a03e6842e9657d512a0de2b9dd20ed73aa9bcc61ec8f4fd8d33ed07e8cc11249'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--attempt',type=Path,required=True);r=a.parse_args().attempt
 for n in ('public_cohort_identity_receipt.json','cohort_identity.json'):
  if (r/n).exists():raise RuntimeError(f'refusing to overwrite {r/n}')
 if sha(DATA)!=EXPECTED:raise RuntimeError('data manifest SHA mismatch')
 d=json.loads(DATA.read_text());f=json.loads(COHORT.read_text());old=json.loads((COHORT.parent/'window_identities.json').read_text())['gc001']
 dev=d['dev8'];test=f['val32_excluding_dev8_scenes'];
 if len(dev)!=8 or len(test)!=24:raise RuntimeError('fixed cohort count mismatch')
 ids=lambda xs:{(w['scene'],tuple(w['context']),tuple(w['novel'])) for w in xs}
 oldtest=old['val32_excluding_dev8_scenes']['windows'];olddev=old['dev8']['windows']
 if ids(test)!=ids(oldtest) or ids(dev)!=ids(olddev):raise RuntimeError('fixed test/dev identities mismatch')
 if len({w['scene'] for w in test})!=24 or len({w['scene'] for w in dev})!=8:raise RuntimeError('cohort scenes not unique')
 identity={'dev':dev,'test':test,'source_manifest_sha256':EXPECTED,'four_arm_cohort_sha256':sha(COHORT),
   'four_arm_window_identity_sha256':sha(COHORT.parent/'window_identities.json'),'dev_test_identity_matches_registered_four_arm':True,
   'public_cohort_only':True,'windows':32,'scenes':32}
 (r/'cohort_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
 (r/'public_cohort_identity_receipt.json').write_text(json.dumps({'status':'PASS','windows':32,'dev':8,'test':24,'source_manifest_sha256':EXPECTED,
   'four_arm_identity':'PASS','written_before_gpu_jobs':True},indent=2)+'\n')
if __name__=='__main__':main()
