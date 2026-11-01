"""Prepare reviewable independent runs, then submit sealed afterok DAG once."""
from pathlib import Path
import argparse,json,os,subprocess,hashlib,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if sys.version_info<(3,9):raise RuntimeError('Use /space/mawb/anaconda3/envs/tokengs/bin/python for AST gate provenance')
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
B4=BASE/'v4init_isolated_noca_v4recipe_20261022_v1';B8=BASE/'v4init_isolated_extend8_recovery_20261023_v2_jsonfix/Baseline'
RUN=BASE/'reconstruction_abcd_querysplat_scratch4_20261025_v4_cgatefix'
OLD=BASE/'reconstruction_abcd_querysplat_scratch4_20261025_v3_devicefix'
OLD_CODE=Path('/space/mawb/ssst_reconstruction_abcd_code_20261025_v3_devicefix')
COHORT='/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json'
B8_SHA='bd1afe7e27ca01533ff5bfef6cd218a99c4f2294de08399a40cd2acf6eb98bc5';ZERO_SHA='bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397'

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def reuse_gate(arm,run,cfg,code):
    from scripts.reconstruction_abcd_gate_reuse import update_signature,task_signature
    old=OLD/arm;smoke=old/'smoke';gate=read(smoke/'SMOKE_PASSED.json');previous=read(old/'configuration.json')
    expected='86b420758b4b9d37966eedb4a9cbf71eeb568a84'
    assert arm in ('A','B','D') and gate['status']=='PASSED' and gate['code_sha']==expected and gate['world_size']==8 and gate['added_updates']==2
    assert len(gate['ranks'])==8 and all(x['finite'] and x['protected_unchanged'] and x['updates']==2 for x in gate['ranks'])
    for k,v in previous.items():
        if k not in ('code_sha','code_snapshot'):assert cfg[k]==v,(arm,k)
    assert update_signature(OLD_CODE/'scripts/train_reconstruction_abcd.py')==update_signature(ROOT/'scripts/train_reconstruction_abcd.py')
    assert task_signature(OLD_CODE/'scripts/train_reconstruction_abcd.py')==task_signature(ROOT/'scripts/train_reconstruction_abcd.py')
    unchanged={}
    for folder in ('tokengs','scripts'):
        for path in (OLD_CODE/folder).rglob('*.py'):
            relative=str(path.relative_to(OLD_CODE))
            if relative in ('scripts/train_reconstruction_abcd.py','scripts/prepare_submit_reconstruction_abcd.py'):continue
            assert sha(path)==sha(ROOT/relative),relative
            unchanged[relative]=sha(path)
    files={'SMOKE_PASSED.json':sha(smoke/'SMOKE_PASSED.json')}
    for i in range(8):
        init=read(smoke/f'INITIALIZATION_rank{i:02d}.json');assert init['status']=='PASSED' and init['source_sha256']==cfg['source_sha256']
        proof=read(smoke/f'UPDATE_AUDIT_2_rank{i:02d}.json');assert proof['status']=='PASSED' and proof['protected_tensors_equal'] and proof['ddp_replicas_equal'] and proof['changed_names']
        shape=read(smoke/f'SHAPES_rank{i:02d}.json');n=cfg['token_count'];assert shape['gaussians']==[1,n*64,23 if arm=='D' else 14] and (arm=='D' or shape['unique_sources']==n)
        gradients=read(smoke/f'TARGETED_GRADIENTS_2_rank{i:02d}.json');assert gradients['status']=='PASSED' and all(v>0 for v in gradients['norms'].values())
        names=[f'INITIALIZATION_rank{i:02d}.json',f'UPDATE_AUDIT_2_rank{i:02d}.json',f'SHAPES_rank{i:02d}.json',f'TARGETED_GRADIENTS_2_rank{i:02d}.json']
        if arm=='D':
            iso=read(smoke/f'ISOLATION_rank{i:02d}.json');assert iso['status']=='PASSED' and iso['LU_reconstruction_nonzero']==0;names.append(f'ISOLATION_rank{i:02d}.json')
        files.update({name:sha(smoke/name) for name in names})
    return dict(status='PASSED',source_run=str(old),source_job_id={'A':'71727','B':'71728','D':'71730'}[arm],source_code_sha=expected,target_code_sha=code,source_checkpoint_sha256=cfg['source_sha256'],unchanged_science_files=unchanged,evidence_files=files,non_C_update_ast_sha256=update_signature(ROOT/'scripts/train_reconstruction_abcd.py'),task_ast_sha256=task_signature(ROOT/'scripts/train_reconstruction_abcd.py'),smoke_weights_reused=False)

def prepare():
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert 'Ran 14 tests' in Path('/space/mawb/ssst_reconstruction_abcd_cpu_tests_20261025.log').read_text() and '\nOK\n' in Path('/space/mawb/ssst_reconstruction_abcd_cpu_tests_20261025.log').read_text()
    launch=read('/space/mawb/ssst_reconstruction_abcd_launch_validation_20261025.json');assert launch['status']=='PASSED'
    assert '\nOK\n' in Path('/space/mawb/ssst_reconstruction_abcd_launch_tests_20261025.log').read_text()
    old=OLD
    for arm in 'ABCD':
        assert not list((old/arm/'train').iterdir())
    cpu=read('/space/mawb/ssst_reconstruction_abcd_cpu_preflight_v3_20261025.json');assert cpu['status']=='PASSED'
    if RUN.exists():raise ValueError('Run already exists; inspect receipts, never reprepare or resubmit')
    assert '\nOK\n' in Path('/space/mawb/ssst_reconstruction_abcd_device_tests_20261025.log').read_text()
    assert '\nOK\n' in Path('/space/mawb/ssst_reconstruction_c_protection_tests_20261025.log').read_text()
    RUN.mkdir()
    shutil.copyfile('/space/mawb/ssst_reconstruction_c_protection_tests_20261025.log',RUN/'C_PROTECTION_CPU_TESTS.log')
    shutil.copyfile('/space/mawb/ssst_reconstruction_abcd_device_tests_20261025.log',RUN/'DEVICE_TESTS.log')
    shutil.copyfile('/space/mawb/ssst_reconstruction_abcd_cpu_preflight_v3_20261025.log',RUN/'CPU_MIGRATION_RERUN.log')
    write(RUN/'LAUNCH_VALIDATION.json',launch)
    shutil.copyfile('/space/mawb/ssst_reconstruction_abcd_launch_tests_20261025.log',RUN/'LAUNCH_TESTS.log')
    write(RUN/'RECOVERY_SOURCE.json',dict(previous_run=str(OLD),previous_code_sha='86b420758b4b9d37966eedb4a9cbf71eeb568a84',reason='C protection gate failed after two smoke updates; A/B/D successful gates reused',science_changes=[],completed_training_updates=0))
    shutil.copyfile('/space/mawb/ssst_reconstruction_abcd_cpu_preflight_v3_20261025.json',RUN/'CPU_PREFLIGHT.json')
    shutil.copyfile('/space/mawb/ssst_reconstruction_abcd_cpu_tests_20261025.log',RUN/'CPU_TESTS.log')
    manifest=read(B4/'train/configuration.json')
    assert read(B4/'eval/FINAL_CHECKPOINT_FROZEN.json')['sha256']=='694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc'
    assert read(B8/'eval/FINAL_CHECKPOINT_FROZEN.json')['sha256']==B8_SHA
    assert read(B4/'eval/evaluation/metrics.json')['windows']==1860 and read(B8/'eval/evaluation/metrics.json')['windows']==1860
    write(RUN/'REFERENCE_RECEIPTS.json',dict(baseline4=read(B4/'eval/FINAL_CHECKPOINT_FROZEN.json'),baseline8=read(B8/'eval/FINAL_CHECKPOINT_FROZEN.json')))
    specs={}
    for arm in 'ABCD':
        run=RUN/arm;run.mkdir();(run/'eval').mkdir();(run/'train').mkdir();(run/'smoke').mkdir()
        source=B4/'train/checkpoint_untrained_initial.pt' if arm=='D' else B8/'eval/source_checkpoint.pt'
        cfg=dict(recipe='reconstruction_abcd_'+arm,arm=arm,code_sha=code,code_snapshot=str(ROOT),
            source_checkpoint=str(source),source_sha256=ZERO_SHA if arm=='D' else B8_SHA,
            manifest=manifest['manifest'],manifest_sha256=manifest['manifest_sha256'],heldout_sha256=manifest['heldout_sha256'],cohort=COHORT,cohort_sha256=manifest['cohort_sha256'],
            baseline4_root=str(B4),baseline8_root=str(B8),source_epoch=0 if arm=='D' else 8,
            added_epochs=4 if arm=='D' else 1,added_updates=4172 if arm=='D' else 1043,full_updates=4172 if arm=='D' else 9387,
            epoch_budget_origin='user specified four epochs' if arm=='D' else 'assistant pilot default from handoff',
            token_count=4096 if arm=='B' else 1024,token_dim=1024,children_per_token=64,global_batch=8,world_size=8,context_views=2,novel_views=2,image_hw=[256,256],
            trainable_scope='all new task modules except frozen semantic-replay geometry helpers' if arm=='D' else 'new RGB residual appearance only' if arm=='C' else 'reconstruction only; understanding/shared frozen',
            original_gradient_isolation=True,supervised_layers=[6,12],layer_weights=[1/3,2/3],
            learning_rate_policy='baseline1e-4, warmup100, cosine floor1e-5; understanding warmup1043' if arm=='D' else 'original Adam retained at terminal main3e-6/geometry3e-7; C appearance fresh3e-5 warmup100',
            ABC_common_pool='48grid top-per-cell; if <4096 unique, supplement descending finite pixel confidence to4096 for ALL ABC; no duplicate anchors',
            B_expansion='retain1024 rows; three repeats plus deterministic1e-4 noise seed20261025; new moment rows zero, table Adam step retained8344',
            C_protection='original RGB also retained for semantic child descriptors; new residual used only for RGB rendering/loss',
            D_adaptations=dict(querysplat_commit='3465a1d2c789d8ebe71f4f9c0bae3f2c2fd726ae',geometry_depth=12,appearance_depth=6,width=1024,original_release_width=2048,heads=16,mlp_ratio=4,query_count=1024,sh_degree=1,patch_size=16,geometry_memory='same four Omega-layer adapter as baseline, fresh random',activation='official signed-expm1 XYZ+z1.0,scale cap.075,normalized quaternion,alpha sigmoid(raw-2)',midlayer_supervision='shared appearance decoder at geometry L6 and L12; no local refinement',semantic_interface='geometry means/radii over64 children; detached semantic reads; DC RGB descriptor',pretrained_task_weights=False) if arm=='D' else None,
            renderer=str(Path('/space/mawb/.cache/torch_extensions_backup/gsplat_cuda_partial_1406/gsplat_cuda.so')),renderer_sha256='82403af8b36fb4b64774afc7462ea6611896fcfe7a02785098bce9ff50d8d751',
            formal_comparison='Native only; no C8 calibration head',training_windows=8337,training_scenes=1191,formal_windows=1860,formal_scenes=312,target_protocol='all6; extra pure Novel metrics',storage_policy='initial and rolling latest; no per-epoch retained copies')
        if arm in ('A','B','D'):
            receipt=reuse_gate(arm,run,cfg,code);write(run/'REUSED_SMOKE.json',receipt)
            (run/'smoke').rmdir();(run/'smoke').symlink_to(os.path.relpath(OLD/arm/'smoke',run))
        else:
            cfg['C_semantic_protection']='matched no_grad/RNG, measured repeat roundoff; absolute cap1e-5; frozen weights and geometry/alpha exact'
        for dest in (run,run/'eval'):write(dest/'configuration.json',cfg)
        (run/'code.sha').write_text(code+'\n');specs[arm]=cfg
    write(RUN/'EXPERIMENT_SPECIFICATIONS.json',specs)
    (RUN/'code.sha').write_text(code+'\n')
    # Tracked source archive excludes LFS assets (training uses external pinned backbones).
    subprocess.run(['git','archive','--format=tar.gz','-o',str(RUN/'source_snapshot.tar.gz'),'HEAD','tokengs','scripts','tests','docs/reconstruction_abcd_execution_20261025.md','docs/reconstruction_abcd_launchfix_20261025.md','docs/reconstruction_abcd_devicefix_20261025.md','docs/reconstruction_abcd_cgatefix_20261025.md'],cwd=ROOT,check=True)
    write(RUN/'PREPARED.json',dict(status='PREPARED_NOT_SUBMITTED',code_sha=code,cpu_status='PASSED',source_archive_sha256=sha(RUN/'source_snapshot.tar.gz'),run=str(RUN)))
    print(RUN)

def submit():
    if (RUN/'SUBMISSION_ATTEMPT.json').exists():raise ValueError('Submission already attempted; inspect individual receipts, no replay')
    prepared=read(RUN/'PREPARED.json');assert prepared['status']=='PREPARED_NOT_SUBMITTED'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==prepared['code_sha']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    # No status polling. Only the caller's one bounded post-submit check follows.
    write(RUN/'SUBMISSION_ATTEMPT.json',dict(code_sha=prepared['code_sha']))
    jobs=[]
    def job(arm,mode,gpus,dep=None,summary=False):
        run=RUN/arm if arm else RUN
        command=['sbatch','--parsable','--partition=3090,4090','--exclude=3dimage-15,3dimage-16,3dimage-20,3dimage-21','--nodes=1',f'--job-name=reconABCDv4_{arm}_{mode}',f'--cpus-per-task={32 if gpus==8 else 4}','--mem=180G' if gpus==8 else '--mem=32G','--time=3-00:00:00' if mode=='train' else '--time=12:00:00',f'--output={run}/slurm_{mode}_%j.out',f'--chdir={ROOT}']
        if gpus:command.append(f'--gres=gpu:{gpus}')
        if dep:command+=['--dependency=afterok:'+':'.join(map(str,dep)),'--kill-on-invalid-dep=yes']
        if summary:
            command+=['--wrap',f'/usr/bin/python3 {ROOT}/scripts/reconstruction_abcd_pipeline.py {RUN} summarize']
        else:command+=[str(ROOT/'scripts/run_reconstruction_abcd.sh'),str(run),mode]
        receipt=dict(arm=arm,mode=mode,gpus=gpus,nodes=1,dependencies=dep or [],command=command)
        write(run/f'SUBMIT_COMMAND_{mode}.json',receipt)
        result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        receipt.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
        if result.returncode:
            write(run/f'SUBMISSION_FAILED_{mode}.json',receipt);raise RuntimeError(result.stderr)
        receipt['job_id']=result.stdout.strip().split(';')[0]
        write(run/f'SUBMITTED_{mode}.json',receipt);jobs.append(receipt);write(RUN/'SUBMITTED.json',dict(code_sha=prepared['code_sha'],jobs=jobs))
        return receipt['job_id']
    smokes=[job('C','smoke',8)]
    scores=[]
    for arm in 'ABCD':
        train=job(arm,'train',8,smokes);export=job(arm,'export',8,[train]);scores.append(job(arm,'score',1,[export]))
    job('','summary',0,scores,summary=True)
    print(json.dumps(jobs,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','submit']);a=p.parse_args();{'prepare':prepare,'submit':submit}[a.mode]()
