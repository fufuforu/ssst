import json
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from scripts.native_joint4_support import FrozenStateAudit,FrozenStateViolation


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.frozen_omega=torch.nn.Linear(4,4).requires_grad_(False)
        self.frozen_omega.register_buffer('constant',torch.tensor(1.))
        self.head=torch.nn.Linear(4,2)
    def forward(self,x):return self.head(self.frozen_omega(x))


def ddp_worker(rank,folder):
    torch.set_num_threads(1);torch.manual_seed(42)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+str(folder/'rendezvous'))
    model=Tiny();audit=FrozenStateAudit(model)
    ddp=DistributedDataParallel(model,broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
    initial=audit.after_ddp();assert len(initial['constructor_version_bumps'])==2
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True);ddp(torch.randn(3,4)).square().mean().backward();optimizer.step();audit.verify_step()
    result=audit.finish();assert result['before_ddp_sha256']==result['endpoint_sha256']
    (folder/f'rank{rank}.json').write_text(json.dumps(dict(initial=initial,result=result)))
    dist.destroy_process_group()


def test_two_rank_real_ddp_constructor_and_steps_preserve_frozen_content(tmp_path):
    torch.set_num_threads(1)
    torch.multiprocessing.start_processes(ddp_worker,args=(tmp_path,),nprocs=2,start_method='fork')
    rows=[json.loads((tmp_path/f'rank{i}.json').read_text()) for i in range(2)]
    assert all(r['result']['contents_equal'] for r in rows)
    assert rows[0]['result']['endpoint_sha256']==rows[1]['result']['endpoint_sha256']


def test_real_content_change_during_ddp_initialization_is_rejected():
    model=Tiny();audit=FrozenStateAudit(model)
    with torch.no_grad():model.frozen_omega.weight.add_(1.)
    with pytest.raises(FrozenStateViolation,match='during DDP construction'):audit.after_ddp()


def test_training_write_is_rejected_even_when_numerical_values_unchanged():
    model=Tiny();audit=FrozenStateAudit(model);audit.after_ddp()
    with torch.no_grad():model.frozen_omega.weight.add_(0.)
    with pytest.raises(FrozenStateViolation,match='during training'):audit.verify_step()


def test_frozen_gradients_are_still_rejected():
    model=Tiny();audit=FrozenStateAudit(model);audit.after_ddp()
    model.frozen_omega.weight.grad=torch.ones_like(model.frozen_omega.weight)
    with pytest.raises(FrozenStateViolation,match='gradient_present'):audit.verify_step()


def test_endpoint_hash_rejects_data_write_that_bypasses_version_counter():
    model=Tiny();audit=FrozenStateAudit(model);audit.after_ddp()
    model.frozen_omega.weight.data.add_(1.)
    audit.verify_step()
    with pytest.raises(FrozenStateViolation,match='at endpoint'):audit.finish()


def test_frozen_buffers_are_checked_with_parameters():
    model=Tiny();audit=FrozenStateAudit(model);audit.after_ddp()
    assert 'frozen_omega.constant' in audit.tensors
    model.frozen_omega.constant.add_(1.)
    with pytest.raises(FrozenStateViolation,match='constant'):audit.verify_step()
