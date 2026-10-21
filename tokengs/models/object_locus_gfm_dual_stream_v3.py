"""GFM dual adapters; SAM initializes queries by pooling semantic-adapter output."""
from tokengs.models.object_locus_gfm_dual_stream_v2 import LocusGSGFMDualStreamV2
from tokengs.models.gfm_dual_stream_components import SemanticSAMQueryInitializer


class LocusGSGFMDualStreamV3(LocusGSGFMDualStreamV2):
    architecture_name='LOCUSGS_OMEGA_GFM_SEMANTIC_POOL_V3'
    def __init__(self,opt,**kwargs):
        super().__init__(opt,**kwargs)
        self.query_initializer=SemanticSAMQueryInitializer()

    def query_seed_memory(self,raw_gfm,fm):
        # No detach: query initialization jointly trains the semantic adapter.
        return fm
