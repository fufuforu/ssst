"""Recovery-specific receipt gate; official scoring and comparisons remain unchanged."""
from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.olda4_ablation_pipeline import read,write,digest
from scripts.v4init_extend8_pipeline import train_gate,finish
from scripts.v4init_extend8_policy import END,START,check_meta


def recovery_gate(root):
    cfg=read(root/'configuration.json')
    assert not (root/'STARTED.json').exists() and not (root/'checkpoint_latest.pt').exists(),'Refuse duplicate recovery'
    assert digest(root/'checkpoint_recovery_source.pt')==cfg['recovery_sha256']
    assert digest(root/'source_epoch4_meta.json')==cfg['source_epoch4_meta_sha256']
    assert read(root/'PREFLIGHT.json')['all_adam_moments_and_steps_restored_exact']
    write(root/'RECOVERY_SOURCE.json',dict(status='PASSED',source=cfg['recovery_source'],frozen_checkpoint=str(root/'checkpoint_recovery_source.pt'),sha256=cfg['recovery_sha256'],progress=cfg['recovery_progress'],source_code_sha=cfg['recovery_source_code_sha'],original_epoch4_sha256=cfg['continuation_sha256']))


def final_gate(root):
    cfg=read(root/'configuration.json');train=Path(cfg['training_root']);tcfg=read(train/'configuration.json')
    cfg['continuation_sha256']=tcfg['continuation_sha256'];write(root/'configuration.json',cfg)
    assert digest(cfg['cohort'])==cfg['cohort_sha256']
    complete=read(train/'TRAINING_COMPLETE.json')
    assert complete['architecture']==cfg['architecture'] and complete['progress']==dict(epoch=8,update_in_epoch=0,full_updates=END)
    check_meta(read(train/'STARTED.json'),cfg)
    recover=cfg['restart_mode']=='recover';kind='RECOVERY_LOADED' if recover else 'CONTINUATION_LOADED'
    progress=cfg['recovery_progress'] if recover else dict(epoch=4,update_in_epoch=0,full_updates=START)
    sha=cfg['recovery_sha256'] if recover else cfg['continuation_sha256']
    rows=[read(train/f'{kind}_rank{i:02}.json') for i in range(8)]
    assert all(x['status']=='PASSED' and x['source_sha256']==sha and x['progress']==progress and x['rank_rng_restored'] and x['all_trainable_parameters_exact'] and x['adam_steps']==[float(progress['full_updates'])] for x in rows)
    assert len({x['model_parameter_sha256'] for x in rows})==1
    first=read(train/'FIRST_UPDATE_VERIFIED.json')
    assert first['status']=='PASSED' and len(first['ranks'])==8 and len({x['sha256'] for x in first['ranks']})==1 and all(x['changed'] for x in first['ranks'])
    assert not (root/'source_checkpoint.pt').exists()
    # Epoch5/6/7 are deliberately not retained; final latest is frozen after completion.
    os.link(train/'checkpoint_latest.pt',root/'source_checkpoint.pt')
    cfg['checkpoint_sha256']=digest(root/'source_checkpoint.pt');write(root/'configuration.json',cfg)
    write(root/'FINAL_CHECKPOINT_FROZEN.json',dict(status='PASSED',sha256=cfg['checkpoint_sha256'],hardlink=True,progress=complete['progress'],source_epoch4_sha256=cfg['continuation_sha256'],recovery_progress=progress))


if __name__=='__main__':
    root=Path(sys.argv[1]);{'recovery_gate':recovery_gate,'train_gate':train_gate,'eval_gate':final_gate,'finish':finish}[sys.argv[2]](root)
