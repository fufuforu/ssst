"""Preserve original superpoint identity as a small, isolated semantic seed residual."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from tokengs.models.serial_query_core import SerialQueryCore


class SerialResidualQueryCore(SerialQueryCore):
    def __init__(self,opt,config):
        super().__init__(opt,config)
        # New modules must not advance the existing model/frontend RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.identity_projection=nn.Linear(config.dim,config.dim)
            nn.init.eye_(self.identity_projection.weight);nn.init.zeros_(self.identity_projection.bias)
        self.identity_gate_logit=nn.Parameter(torch.tensor(math.log(.1/.9)))

    def make_semantic_seed(self,reconstruction_query,initial_query):
        seed=super().make_semantic_seed(reconstruction_query,initial_query)
        identity=F.layer_norm(initial_query.detach(),(self.config.dim,))
        return seed+self.identity_gate_logit.sigmoid()*self.identity_projection(identity)
