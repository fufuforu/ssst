"""C/D factorial ablations: disable only the Query->FM cross-attention read."""
import torch
from tokengs.models.dynamic_panoptic_controller import DynamicObjectLayer
from tokengs.models.point1024_isoadapter_core import Point1024IsolatedAdapterCore
from tokengs.models.point1024_isoadapter_model import LocusGSPoint1024IsolatedAdapter
from tokengs.models.point1024_isoadapter_noca import GeometricOnlyObjectLayer,Point1024NoCACore

C_ARCH='OMEGA_POINT1024_FMQUERY_ISOLATED_ANCHOR_CA_NO_FM_CA_V1'
D_ARCH='OMEGA_POINT1024_FMQUERY_ISOLATED_NO_ANCHOR_CA_NO_FM_CA_GEOMETRIC_ROUTING_V1'

class NoFMRead:
    def _read_image(self,q,image):
        # Preserve norms[1](q1 + 0); memory LN is retained but never executed.
        return torch.zeros_like(q),None

class AnchorOnlyObjectLayer(NoFMRead,DynamicObjectLayer):pass
class GeometricOnlyNoFMObjectLayer(NoFMRead,GeometricOnlyObjectLayer):pass

class Point1024AnchorOnlyCore(Point1024IsolatedAdapterCore):
    def __init__(self,opt,config):
        super().__init__(opt,config)
        for layer in self.panoptic.layers.values():
            layer.__class__=AnchorOnlyObjectLayer
            del layer.image_attention

class Point1024NeitherCACore(Point1024NoCACore):
    def __init__(self,opt,config):
        super().__init__(opt,config)
        for layer in self.panoptic.layers.values():
            layer.__class__=GeometricOnlyNoFMObjectLayer
            del layer.image_attention

class NoFMMetrics:
    def step_loss(self,*args,**kwargs):
        output,metrics=super().step_loss(*args,**kwargs)
        metrics.update(object_anchor_ca_enabled=int(self.architecture_name==C_ARCH),object_fm_ca_enabled=0,
            geometric_anchor_routing_enabled=int(self.architecture_name==D_ARCH))
        return output,metrics

class LocusGSPoint1024AnchorOnly(NoFMMetrics,LocusGSPoint1024IsolatedAdapter):
    architecture_name=C_ARCH
    core_type=Point1024AnchorOnlyCore

class LocusGSPoint1024NeitherCA(NoFMMetrics,LocusGSPoint1024IsolatedAdapter):
    architecture_name=D_ARCH
    core_type=Point1024NeitherCACore
