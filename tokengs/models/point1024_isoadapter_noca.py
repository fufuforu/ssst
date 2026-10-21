"""P1024iso ablation: geometry-only routing, no learned Object->Anchor CA."""
import torch
from tokengs.models.dynamic_panoptic_controller import DynamicObjectLayer
from tokengs.models.point1024_isoadapter_core import Point1024IsolatedAdapterCore
from tokengs.models.point1024_isoadapter_model import LocusGSPoint1024IsolatedAdapter

NO_CA_ARCH='OMEGA_POINT1024_FMQUERY_ISOLATED_NO_ANCHOR_CA_GEOMETRIC_ROUTING_V1'

class GeometricOnlyObjectLayer(DynamicObjectLayer):
    def _read_anchor(self,q,a,bias):
        # a and q content are deliberately absent from geometric routing.
        # Retain baseline LayerNorm(q + 0), preserving normalization placement.
        if not torch.isfinite(bias).all():raise FloatingPointError('Nonfinite geometric routing bias')
        weights=bias.softmax(-1)
        return torch.zeros_like(q),weights[:,None]  # [B,1,Nthing+2,Nanchor]

class Point1024NoCACore(Point1024IsolatedAdapterCore):
    def __init__(self,opt,config):
        super().__init__(opt,config)
        # Construct all baseline modules first to preserve initialization RNG.
        for layer in self.panoptic.layers.values():
            layer.__class__=GeometricOnlyObjectLayer
            del layer.anchor_attention

class LocusGSPoint1024IsolatedNoCA(LocusGSPoint1024IsolatedAdapter):
    architecture_name=NO_CA_ARCH
    core_type=Point1024NoCACore
    def step_loss(self,*args,**kwargs):
        output,metrics=super().step_loss(*args,**kwargs)
        metrics.update(object_anchor_ca_enabled=0,geometric_anchor_routing_enabled=1)
        return output,metrics
