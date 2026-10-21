"""FP32 native AdamW with task-owned CPU master parameters and moment state."""
import torch


class CPUOffloadAdamW:
    """Keep the native AdamW formula; synchronize FP32 model parameters per step."""
    def __init__(self, groups):
        self.model_groups = groups
        self.pairs = []
        cpu_groups = []
        for group in groups:
            copied = {key:value for key,value in group.items() if key != 'params'}
            cpu_params = []
            for model_param in group['params']:
                if model_param.dtype != torch.float32:
                    raise ValueError('adaptation requires FP32 model/master parameters')
                master = torch.nn.Parameter(model_param.detach().cpu().clone())
                cpu_params.append(master)
                self.pairs.append((model_param, master))
            copied['params'] = cpu_params
            cpu_groups.append(copied)
        self.optimizer = torch.optim.AdamW(cpu_groups,betas=(.9,.95),eps=1e-8,foreach=False)
        # Public groups expose actual device parameters for clipping/provenance.
        self.param_groups = groups

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter, _ in self.pairs:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)

    @torch.no_grad()
    def step(self):
        for original, copied in zip(self.param_groups,self.optimizer.param_groups):
            for key in ('lr','weight_decay','peak_lr'):
                copied[key] = original[key]
        for parameter, master in self.pairs:
            master.grad = None if parameter.grad is None else parameter.grad.detach().cpu()
        self.optimizer.step()
        for parameter, master in self.pairs:
            parameter.copy_(master)
            master.grad = None

    def state_dict(self):
        return {'kind':'native_fp32_cpu_adamw_v1','optimizer':self.optimizer.state_dict()}

    def load_state_dict(self, state):
        if state.get('kind') != 'native_fp32_cpu_adamw_v1':
            raise ValueError('wrong adaptation optimizer kind')
        self.optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():
            for parameter, master in self.pairs:
                master.copy_(parameter.detach().cpu())
        for original, copied in zip(self.param_groups,self.optimizer.param_groups):
            for key in ('lr','weight_decay','peak_lr'):
                original[key] = copied[key]
