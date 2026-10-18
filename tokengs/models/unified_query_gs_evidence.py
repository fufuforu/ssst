"""Attempt02 core with projected, depth-validated per-Gaussian GFM evidence."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.unified_query_core import UnifiedQueryCore
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.unified_superpoints import SuperpointQueryInitializer
from tokengs.models.locusgs_recon import LocusGSGaussianHead
from tokengs.models.input_types import ModelInputDecoder
from tokengs.models.object_locus_posefree_geometry import camera_vectors

EVIDENCE_DIM = 256


def sample_gs_evidence(memory, gaussians, *, return_zero_fraction=False):
    """Sample native GFM at GS projections; average only depth-consistent views.

    lift_features is a rasterizer adjoint: its grad_outputs must match the
    renderer's 256x256 output, whereas this memory is the native 32x32 GFM.
    Use explicit projection rather than resizing or changing the renderer.
    """
    assert gaussians.ndim == 3 and gaussians.shape[0] == 1 and gaussians.shape[2] == 14
    n = gaussians.shape[1]
    grid, c2w, intrinsics, depth = (memory[k] for k in ('grid', 'c2w', 'intrinsics', 'depth'))
    assert grid.ndim == 5 and grid.shape[:3] == (1, 2, EVIDENCE_DIM)
    assert grid.shape[3] == grid.shape[4] and grid.shape[3] > 0
    assert c2w.shape == (1, 2, 4, 4)
    assert intrinsics.shape == (1, 2, 3, 3)
    assert depth.shape == (1, 2, 1, 256, 256)
    assert torch.isfinite(gaussians).all()
    mu = gaussians[0, :, :3].detach()
    assert mu.shape == (n, 3)
    values, validities = [], []
    for view in (0, 1):
        w2c = torch.linalg.inv(c2w[0, view])
        assert w2c.shape == (4, 4)
        xyz = mu @ w2c[:3, :3].transpose(0, 1) + w2c[:3, 3]
        assert xyz.shape == (n, 3)
        z = xyz[:, 2]
        assert z.shape == (n,)
        safe_z = torch.where(z != 0, z, torch.ones_like(z))
        assert safe_z.shape == (n,)
        k = intrinsics[0, view]
        u = k[0, 0] * xyz[:, 0] / safe_z + k[0, 2]
        v = k[1, 1] * xyz[:, 1] / safe_z + k[1, 2]
        assert u.shape == v.shape == (n,)
        grid_xy = torch.stack(((u + .5) / 256 * 2 - 1, (v + .5) / 256 * 2 - 1), -1)[None, :, None]
        assert grid_xy.shape == (1, n, 1, 2)
        sampled = F.grid_sample(grid[0, view][None], grid_xy, mode='bilinear',
                                align_corners=False, padding_mode='border')
        assert sampled.shape == (1, EVIDENCE_DIM, n, 1)
        feat = sampled[0, :, :, 0].transpose(0, 1)
        assert feat.shape == (n, EVIDENCE_DIM)
        sampled_depth = F.grid_sample(depth[0, view][None], grid_xy, mode='bilinear',
                                      align_corners=False, padding_mode='border')
        assert sampled_depth.shape == (1, 1, n, 1)
        depth_sampled = sampled_depth[0, 0, :, 0]
        assert depth_sampled.shape == (n,)
        valid = ((z > 0) & (u >= 0) & (u < 256) & (v >= 0) & (v < 256)
                 & ((z - depth_sampled).abs() <= .05 * z))
        assert valid.shape == (n,)
        values.append(torch.where(valid[:, None], feat, torch.zeros_like(feat)))
        validities.append(valid.to(feat.dtype))
    total = values[0] + values[1]
    count = validities[0] + validities[1]
    assert total.shape == (n, EVIDENCE_DIM) and count.shape == (n,)
    evidence = (total / count.clamp_min(1)[:, None])[None]
    assert evidence.shape == (1, n, EVIDENCE_DIM)
    assert torch.isfinite(evidence).all()
    if return_zero_fraction:
        with torch.no_grad():
            zero_fraction = (evidence.abs().sum(-1) == 0).float().mean()
            assert zero_fraction.shape == ()
            zero_fraction = float(zero_fraction)
        return evidence, zero_fraction
    return evidence


class GSEvidenceUnifiedCore(UnifiedQueryCore):
    def __init__(self, opt, config):
        # Keep the base construction order and consume exactly its RNG stream.
        nn.Module.__init__(self)
        self.config = config
        d = config.dim
        assert d == EVIDENCE_DIM and opt.dec_patch_size == 8
        assert config.matching_mode == 'source_many_to_one'
        assert config.support_matching is True and config.support_purity == .5
        self.initializer = SuperpointQueryInitializer(config)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(d, config.heads, 4*d, dropout=0., activation='gelu', batch_first=True, norm_first=True) for _ in range(config.depth)])
        self.geometry = nn.ModuleList([nn.Linear(d, 4) for _ in self.blocks])
        for head in self.geometry:
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        self.reconstruction_adapter = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.activation_head = LocusGSGaussianHead(opt.evolve(enc_embed_dim=d))
        self.semantic_adapter = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.classifier = nn.Linear(d, 19)
        self.query_mask = nn.Linear(d, d)
        self.child_index = nn.Embedding(opt.dec_patch_size**2, 16)
        original_child = nn.Sequential(nn.Linear(d+19, d), nn.GELU(), nn.Linear(d, d))
        with torch.random.fork_rng(devices=[]):
            self.child_feature = nn.Sequential(nn.Linear(d+19+EVIDENCE_DIM, d), nn.GELU(), nn.Linear(d, d))
        # fork_rng alone preserves subsequent modules, not these common tensors.
        with torch.no_grad():
            self.child_feature[0].bias.copy_(original_child[0].bias)
            self.child_feature[2].weight.copy_(original_child[2].weight)
            self.child_feature[2].bias.copy_(original_child[2].bias)
        self.stuff_feature = nn.Linear(d, 2*d)

    def forward(self, cloud, memory, superpoint_ids=None):
        assert memory['tokens'].ndim == 3 and memory['tokens'].shape[0] == 1
        assert memory['tokens'].shape[2] == self.config.dim
        initial=self.initializer(cloud,superpoint_ids)
        q=initial['query'];centers=initial['centers'];radii=initial['radii'];states=[]
        for i,(block,head) in enumerate(zip(self.blocks,self.geometry)):
            q=self.decode_query(block,q,memory['tokens'],centers,radii,i)
            update=head(q)
            centers=centers+.1*radii[...,None]*update[...,:3].tanh()
            radii=radii*torch.exp(.1*update[...,3].tanh())
            states.append(dict(layer=i+1,tokens=self.reconstruction_adapter(q),mu=centers,radii=radii,unified_query=q))
        gaussians=self.activation_head(states[-1]['tokens'],centers,radii)
        semantic=self.semantic_adapter(q);logits=self.classifier(semantic)
        parent=semantic.repeat_interleave(self.child_index.num_embeddings,dim=1)
        index=self.child_index.weight.repeat(q.shape[1],1)[None].expand(len(q),-1,-1)
        local=(gaussians[...,:3]-centers.repeat_interleave(self.child_index.num_embeddings,1))/radii.repeat_interleave(self.child_index.num_embeddings,1)[...,None]
        evidence, evidence_zero_fraction = sample_gs_evidence(memory, gaussians, return_zero_fraction=True)
        assert gaussians.shape == (1, q.shape[1]*64, 14)
        assert parent.shape == (1, q.shape[1]*64, self.config.dim)
        assert index.shape == (1, q.shape[1]*64, 16)
        assert local.shape == (1, q.shape[1]*64, 3)
        assert evidence.shape == (1, q.shape[1]*64, EVIDENCE_DIM)
        assert torch.isfinite(evidence).all()
        children=self.child_feature(torch.cat((parent,index,local,evidence),-1))
        assert children.shape == (1, q.shape[1]*64, self.config.dim)
        masks=self.query_mask(semantic)
        stuff=self.stuff_feature(semantic.mean(1)).reshape(len(q),2,self.config.dim)
        membership_logits=children@torch.cat((masks,stuff),1).transpose(1,2)/self.config.dim**.5
        states[-1]['thing_logits19']=logits
        return dict(states=states,gaussians=gaussians,gaussian_membership=membership_logits.sigmoid(),gaussian_mask_logits=membership_logits,
            evidence_zero_fraction=evidence_zero_fraction,
            thing_logits19=logits,p_class=logits.softmax(-1),q_pre=initial['query'],unified_query=q,
            superpoints=initial,cloud=cloud,gaussian_parent_query=torch.arange(q.shape[1],device=q.device).repeat_interleave(self.child_index.num_embeddings),num_thing_queries=q.shape[1],
            **self.auxiliary_gaussian_readout(children))

    def parameter_groups(self):
        result = super().parameter_groups()
        shared = {id(p) for p in result['shared']}
        understanding = {id(p) for p in result['understanding']}
        assert all(id(p) not in shared and id(p) in understanding for p in self.child_feature.parameters())
        return result


class LocusGSGSEvidenceUnified(LocusGSUnifiedQueryV1):
    architecture_name = 'OMEGA_3D_SUPERPOINT_UNIFIED_QUERY_GS_EVIDENCE_V1'
    core_type = GSEvidenceUnifiedCore

    def step_loss(self, batch, *, step=0, phase='train', understanding_weight=1.):
        output, metrics = super().step_loss(batch, step=step, phase=phase,
                                           understanding_weight=understanding_weight)
        zero_fraction = output['prediction']['evidence_zero_fraction']
        assert isinstance(zero_fraction, float) and 0. <= zero_fraction <= 1.
        metrics['evidence_zero_fraction'] = zero_fraction
        return output, metrics

    def encode_context(self, rgb):
        omega, cloud, memory = super().encode_context(rgb)
        assert memory.ndim == 3 and memory.shape[0] == 1 and memory.shape[2] == self.config.dim
        v = 2
        n = memory.shape[1] // v
        side = int(round(n ** .5))
        assert side * side * v == memory.shape[1], 'unexpected token grid'
        grid = memory.reshape(1, v, side, side, self.config.dim).permute(0,1,4,2,3).contiguous()
        assert grid.shape == (1, 2, EVIDENCE_DIM, side, side)
        decoder = ModelInputDecoder(cam_view=torch.linalg.inv(omega['c2w']).transpose(-1,-2),
                                    intrinsics=camera_vectors(omega['k']))
        return omega, cloud, dict(tokens=memory, grid=grid, decoder=decoder, renderer=self.gs,
                                 intrinsics=omega['k'], c2w=omega['c2w'], depth=omega['depth'])


@dataclass(frozen=True)
class CoverageRecipe:
    loss_weight: float = .1
    erode_px: int = 1
    views: int = 2
    max_inside_fraction_guard: float = 1.


RECIPE_COVERAGE = CoverageRecipe()


def coverage_inside_mask(gt, valid):
    """Union of valid GT instances, eroded one pixel; no GT reconstruction."""
    assert gt.ndim == 4 and gt.shape[1:] == (2, 256, 256) and gt.dtype == torch.bool
    assert valid.shape == (2, 256, 256) and valid.dtype == torch.bool
    inside = gt.any(0) & valid
    assert inside.shape == (2, 256, 256)
    inside = (-F.max_pool2d(-inside.float()[:, None], 3, 1, 1)[:, 0]) > .5
    assert inside.shape == (2, 256, 256)
    assert float(inside.float().mean()) <= RECIPE_COVERAGE.max_inside_fraction_guard
    return inside


class LocusGSGSEvidenceCoverage(LocusGSGSEvidenceUnified):
    architecture_name = 'OMEGA_3D_SUPERPOINT_UNIFIED_QUERY_GS_EVIDENCE_COVERAGE_V1'

    @property
    def extra_checkpoint_metadata(self):
        return dict(coverage_recipe=asdict(RECIPE_COVERAGE))

    def step_loss(self, batch, *, step=0, phase='train', understanding_weight=1.):
        output, metrics = super().step_loss(batch, step=step, phase=phase,
                                           understanding_weight=understanding_weight)
        pred = output['prediction']
        targets = pred['final_targets']
        alpha = pred['alpha'][0, :RECIPE_COVERAGE.views, 0]
        assert alpha.shape == (2, 256, 256)
        gt = targets['gt_pixel_masks'][0]
        valid = targets['valid_pixels'][0]
        inside = coverage_inside_mask(gt, valid)
        assert inside.shape == alpha.shape
        applied = bool(inside.any())
        if applied:
            inside_alpha = alpha[inside]
            assert inside_alpha.ndim == 1 and inside_alpha.shape[0] == int(inside.sum())
            cov = F.binary_cross_entropy(inside_alpha.clamp(1e-6, 1-1e-6), torch.ones_like(inside_alpha))
            alpha_inside_mean = inside_alpha.mean().detach()
        else:
            cov = alpha.sum() * 0.
            alpha_inside_mean = 0.
        assert cov.shape == () and torch.isfinite(cov)
        metrics['loss_coverage_applied'] = applied
        metrics['loss_coverage'] = cov.detach()
        metrics['loss_coverage_weighted'] = (RECIPE_COVERAGE.loss_weight * cov).detach()
        metrics['alpha_inside_mean'] = alpha_inside_mean
        metrics['coverage_inside_pixels'] = float(inside.sum())
        # Geometry regularizer: intentionally independent of understanding_weight.
        metrics['loss'] = (metrics['loss_recon'] + understanding_weight * metrics['loss_understanding']
                           + RECIPE_COVERAGE.loss_weight * cov)
        return output, metrics
