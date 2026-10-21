"""Reconstruction-only Q+P control on the native 1024/256 dual architecture."""
import os
from tokengs.models.superpoint_dual_core import SuperpointDualCore
from tokengs.models.reconstruction_qp import LearnedReconstructionAnchorExpansion

class ReconstructionQPSuperpointDualCore(SuperpointDualCore):
    def __init__(self,opt,config):
        if os.environ.get('RG_IOUHEAD') not in (None,'','0'):
            raise ValueError('Reconstruction Q+P control uses the baseline understanding branch without an IoU-head change')
        super().__init__(opt,config)
        self.reconstruction_qp=LearnedReconstructionAnchorExpansion(num_slots=20,dim=1024)
    def configure_qp_slots(self,num_slots):
        if isinstance(num_slots,bool) or not isinstance(num_slots,int) or num_slots<1:raise ValueError('Positive integer Q bank size required')
        if num_slots!=self.reconstruction_qp.num_slots:
            reference=self.reconstruction_qp.learned_q
            self.reconstruction_qp=LearnedReconstructionAnchorExpansion(num_slots=num_slots,dim=1024).to(device=reference.device,dtype=reference.dtype)
    def initialize_reconstruction_anchors(self,reconstruction_seed):
        p=self.reconstruction_seed_projection(reconstruction_seed['query'])
        return self.reconstruction_qp(p,reconstruction_seed['centers'],reconstruction_seed['radii'])
