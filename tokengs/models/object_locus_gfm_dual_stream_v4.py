"""V3 architecture with geometric-aware SAM query pooling only."""
from tokengs.models.object_locus_gfm_dual_stream_v3 import LocusGSGFMDualStreamV3
from tokengs.models.geometric_sam_initialization import GeometricSAMQueryInitializer


class LocusGSGFMDualStreamV4(LocusGSGFMDualStreamV3):
    architecture_name='LOCUSGS_OMEGA_GFM_GEOMETRIC_SAM_POOL_V4'

    def __init__(self,opt,sam_residual=False,**kwargs):
        super().__init__(opt,**kwargs)
        if sam_residual:
            from tokengs.models.geometric_sam_initialization import ResidualGeometricSAMQueryInitializer
            self.query_initializer=ResidualGeometricSAMQueryInitializer()
        else:
            self.query_initializer=GeometricSAMQueryInitializer()
