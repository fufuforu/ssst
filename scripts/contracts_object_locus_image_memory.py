"""Only contracts for image memory, full plan, source load and grouped accumulation."""
import tempfile
from scripts.object_locus_image_memory_runtime import *


def main():
    torch.set_num_threads(4)
    manifest,plan,plan_sha=prepare_plan()
    fm=torch.arange(2*256*128*128,dtype=torch.float32).reshape(1,2,256,128,128)
    old=torch.nn.functional.interpolate(fm.flatten(0,1),size=(32,32),mode='bilinear',align_corners=False)
    old=old.reshape(1,2,256,1024).permute(0,1,3,2).reshape(1,2048,256)
    assert torch.equal(object_image_memory(fm,32),old)
    full=object_image_memory(fm,128);assert full.shape==(1,32768,256)
    assert torch.equal(full,fm.permute(0,1,3,4,2).reshape(1,32768,256))
    try:object_image_memory(fm,64)
    except ValueError:pass
    else:raise AssertionError('unsupported resolution accepted')
    del fm,old,full
    assert (plan['U'],plan['P'],plan['total_updates'],plan['total_exposures'])==(1043,7,8344,66752)
    assert multiplier(199,8344)==1 and abs(multiplier(8343,8344)-.1)<1e-12
    train={w['scene'] for w in manifest['windows']}
    monitors=manifest['monitor_splits']
    for split in ('dev8','val32'):
        values=monitors[split]
        if isinstance(values,dict):values=values.get('windows',values.get('entries'))
        assert not train.intersection(w['scene'] for w in values)
    # Independent eight-sample numeric reference versus two-local/four-rank accumulator.
    names=['reconstruction.weight','understanding.weight','panoptic.weight'];reference=[];ranks=[]
    recs=torch.arange(8*3*2,dtype=torch.float64).reshape(8,3,2)/17
    unders=torch.flip(recs,[0])*0.37
    for rank in range(4):
        acc=[None]*3
        for micro in range(2):
            i=4*micro+rank;accumulate(acc,names,list(recs[i]),list(unders[i]),2)
        ranks.append(torch.stack(acc))
    weights=torch.tensor([.01,1,1],dtype=torch.float64).reshape(1,3,1)
    reference=(recs+weights*unders).mean(0)
    assert torch.allclose(torch.stack(ranks).mean(0),reference,rtol=0,atol=1e-14)
    # Verify the source and complete strict state once; configuration introduces no state.
    model,opt,optimizer,identity=construct('c32','cpu')
    initial=[(name,tuple(value.shape),value.dtype) for name,value in model.state_dict().items()]
    groups=[(g['name'],g['param_names'],g['weight_decay'],g['peak_lr']) for g in optimizer.param_groups]
    opt.object_image_memory_size=128;model.opt.object_image_memory_size=128;model.anchor_decoder.opt.object_image_memory_size=128
    other=base.build_optimizer(model)
    assert initial==[(name,tuple(value.shape),value.dtype) for name,value in model.state_dict().items()]
    assert groups==[(g['name'],g['param_names'],g['weight_decay'],g['peak_lr']) for g in other.param_groups]
    levels=[(name,g[2],g[3]) for g in groups for name in g[1] if name.endswith('.level_embed')]
    assert len(levels)==2 and all(wd==0 and lr==1e-5 for _,wd,lr in levels)
    blob=torch.load(base.SOURCE_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    assert all(torch.equal(v.cpu(),blob['model'][name]) for name,v in model.state_dict().items())
    # Small atomic I/O contract checks registered metadata without saving another full model.
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'latest.pt';payload={'model':{'p':torch.ones(2)},'completed_new_updates':1043,'plan_sha256':plan_sha,'git_sha':'test','object_image_memory_size':128,'arm':'u128'}
        atomic_checkpoint(path,payload);read=torch.load(path,weights_only=False)
        assert read['completed_new_updates']==1043 and read['object_image_memory_size']==128
    write_json(ROOT/'cpu_contracts.json',{'status':'PASS','plan_sha256':plan_sha,'source':identity,'memory32_exact':True,'memory128_order':True,'state_compatible':True,
        'identical_initialization':True,'optimizer_groups_identical':True,'grouped_gradient_8_sample_mean':True,'S':1191,'N':8337,'U':1043,'P':7,
        'checkpoint_metadata_roundtrip':True,'level_embed':levels})
    print('CPU contracts PASS',flush=True)

if __name__=='__main__':main()
