"""V3 architecture with geometric-aware SAM query pooling only."""
from tokengs.models.object_locus_gfm_dual_stream_v3 import LocusGSGFMDualStreamV3
from tokengs.models.geometric_sam_initialization import GeometricSAMQueryInitializer


class LocusGSGFMDualStreamV4(LocusGSGFMDualStreamV3):
    architecture_name='LOCUSGS_OMEGA_GFM_GEOMETRIC_SAM_POOL_V4'

    def __init__(self,opt,sam_residual=False,anchor_query_seed=False,**kwargs):
        super().__init__(opt,**kwargs)
        if anchor_query_seed and not sam_residual:raise ValueError('Anchor seed requires learned-base residual')
        self.anchor_query_seed=anchor_query_seed
        if anchor_query_seed:
            if self.write_strength!=0.:raise ValueError('Two-stage Anchor seed requires Write=0')
            from tokengs.models.anchor_seed_initialization import UpdatedAnchorSAMQueryInitializer
            self.query_initializer=UpdatedAnchorSAMQueryInitializer()
        elif sam_residual:
            from tokengs.models.geometric_sam_initialization import ResidualGeometricSAMQueryInitializer
            self.query_initializer=ResidualGeometricSAMQueryInitializer()
        else:
            self.query_initializer=GeometricSAMQueryInitializer()

    def decode_scene(self,tokens,latent,rays,fm,proposals,omega,mu,radius,raw_gfm):
        if not self.anchor_query_seed:
            return super().decode_scene(tokens,latent,rays,fm,proposals,omega,mu,radius,raw_gfm)
        from tokengs.models.object_locus_v3_set_controller import scene_normalization
        reconstruction,stats=self.anchor_decoder.forward_stateful(tokens,latent,rays,None,None,None,None,self.understanding_step,
            seed_mu=mu,seed_radius=radius,seed_c=None,seed_s=None,write_strength=0.,reconstruction_only=True)
        final=reconstruction[-1]
        _,ell=scene_normalization(reconstruction[5]['mu'].detach())
        # Reuse the existing understanding-side Anchor projection. Gradients may
        # train this projection, but cannot flow into reconstruction content.
        features=self.panoptic.encode_token(final['tokens'].detach(),final['mu'].detach(),final['radii'].detach(),ell.detach())
        context=dict(features=features,mu=final['mu'].detach(),records=[[] for _ in range(len(tokens))])
        q,c,s,proposal_log=self.query_initializer(self.query_seed_memory(raw_gfm,fm),proposals,omega['points'],mu.detach(),seed_context=context)
        states,_=self.anchor_decoder.forward_stateful(tokens.detach(),latent,rays,self.panoptic,fm,q,self.understanding.mask_embedder,self.understanding_step,
            seed_mu=mu.detach(),seed_radius=radius.detach(),seed_c=c,seed_s=s,write_strength=0.,reconstruction_states=reconstruction)
        # The existing multi-layer reconstruction objective consumes states,
        # not just the exported final Gaussians. Keep original reconstruction
        # tensors there while Object results were computed from detached reads.
        states=[dict(state,**{key:rec[key] for key in ('tokens','mu','rho','radii')})
                for state,rec in zip(states,reconstruction)]
        return states,stats,q,c,s,proposal_log,final
