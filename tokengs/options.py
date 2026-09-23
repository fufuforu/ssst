# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tyro CLI options and named presets (`AllConfigs` subcommands)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import tyro


@dataclass
class Options:
    # --- general
    evaluating: bool = False
    workspace: str = "./workspace"
    resume: str | None = None
    model_type: str = "tokengs"
    seed: int = 42

    # --- wandb / logging
    use_wandb: bool = False
    experiment_name: str = "tokengs"
    out_dir: str = "outputs"
    project_name: str = "TokenGS"

    # --- model architecture
    img_size: tuple[int, int] = (256, 256)
    patch_size: int = 8
    dec_patch_size: int | None = None
    enc_depth: int = 3
    dec_depth: int = 12
    enc_embed_dim: int = 1024
    enc_num_heads: int = 16
    mlp_ratio: int = 4
    clip_head_readout_std: float = 0.002
    clip_head_z_init: float | None = None
    dec_init_values: float | None = None

    # --- gaussian splatting
    bg_color: Literal["white", "black", "grey"] = "grey"
    gaussian_scale_cap: float = 0.075
    opacity_bias: float = 2.0
    gaussian_z_offset: float = 1.0
    num_gs_tokens: int = 1024
    token_dim: int = 1024
    gs_token_std: float = 1e-2
    num_dynamic_gs_tokens: int = 0
    init_dynamic_tokens_from_static: bool = False
    init_tokens_from_existing: bool = False
    init_latents_from_existing: bool = False

    # --- dataset
    data_mode: tuple[tuple[str, int], ...] = (("dl3dv_scaled_0.15", 6),)
    num_views: int = 8
    num_input_views: int = 4
    znear: float = 0.025
    zfar: float = 125.0
    camera_normalization_method: Literal["mean_cam", "first_cam"] = "first_cam"
    camera_scale_method: Literal["constant", "distance", "bound", "pointmap"] = "constant"
    pointmap_trim_lo: float = 0.0
    pointmap_trim_hi: float = 1.0
    num_workers: int = 16
    dataset_kwargs: dict[str, str] | None = None

    # --- training
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 30
    max_iters_per_epoch: int = 1_000_000
    lr: float = 4e-4
    pct_start_steps: int = 1000
    final_div_factor: float = 1000.0
    gradient_clip: float = 1.0
    mixed_precision: str = "bf16"
    deferred_bp: bool = False
    use_input_supervision: bool = False
    mean_of_grads: Literal["none", "per-scene", "per-view"] = "none"
    mean_of_grads_scene_chunk_size: int = 1
    mean_of_grads_view_chunk_size: int | None = None

    # --- loss weights
    rgb_loss_type: Literal["l1", "l2"] = "l2"
    lambda_rgb: float = 1.0
    lambda_lpips: float = 0.0
    lambda_mask: float = 0.0
    lambda_ssim: float = 0.2
    lambda_visibility: float = 1.0
    visibility_distance_threshold: float = 1.0
    lambda_opacity: float = 0.0
    lambda_dyn_aux: float = 0.0
    lambda_dyn_aux_warmup_steps: int = 0
    lambda_dyn_aux_decay_steps: int = 0
    lambda_dyn_aux_min: float = 0.0

    # --- logging frequency
    print_freq: int = 10
    log_image_freq: int = 100

    # --- evaluation
    eval_n_media_dumps: int = 0
    strict_checkpoint_loading: bool = True

    # --- test-time training (eval)
    use_ttt_for_eval: bool = False
    ttt_mode: Literal["token-tuning", "scene-latent-tuning", "tokens", "latents"] = "token-tuning"
    ttt_n_steps: int = 50
    ttt_lr: float = 1e-4

    # --- dynamic scenes
    time_embedding: bool = False
    time_embedding_dim: int = 2
    use_interp_target: bool = False

    # --- latent bottleneck release architecture
    use_multiscale_encoder: bool = False
    multiscale_encoder_layers: tuple[int, ...] = (5, 7, 9, 11)
    use_latent_bottleneck: bool = False
    num_latents: int = 4096
    latent_cross_attn_depth: int = 12

    # --- augmentation
    random_reflect: bool = True

    # --- SSST: spatially grounded shared tokens + unified object queries ---
    # Anchor prior in the TokenGS normalized world frame (first context camera
    # at the origin, dataset scene scale applied to camera translations). The
    # defaults follow the measured SIU3R processed-ScanNet scale: the visible
    # surface sits around z ~ 0.1..0.5 with p95 |x| ~ 0.2.
    anchor_center_z: float = 0.25
    anchor_extent: float = 0.3
    anchor_init_radius: float = 0.05
    anchor_num_freqs: int = 4
    anchor_refine_step: float = 0.25
    anchor_radius_min: float = 0.005
    anchor_radius_max: float = 1.0
    anchor_radius_soft_min: float = 0.01
    anchor_radius_soft_max: float = 0.25
    anchor_local_offset_bound: float = 1.0
    # LocusGS-style anchor-to-ray geometric attention bias (SSST decoder only).
    anchor_ray_bias: bool = True
    anchor_ray_sigma0: float = 0.1
    anchor_ray_bias_clamp: float = -20.0
    # Raw parameter of softplus(raw) = gamma; -6 keeps the bias a small
    # perturbation at initialization so a warm start stays meaningful.
    anchor_ray_bias_init: float = -6.0
    num_object_queries: int = 100
    semantic_class_count: int = 20
    num_object_query_layers: int = 2
    query_seed_std: float = 0.02
    query_block_init_values: float = 0.01
    query_spatial_pe_std: float = 0.02
    assignment_temperature_init: float = 5.0
    use_instance_labels: bool = False
    # Experiment 1: train only the spatial reconstruction path.  The unified
    # object-query branch is frozen and never executed (no query forward, no
    # mask rendering, no Hungarian matching, no understanding loss).
    reconstruction_only: bool = False
    # --- LocusGS-faithful reconstruction (arXiv:2608.12825), ScanNet adaptation ---
    # Values the paper does not specify are marked "unspecified" with our choice.
    locusgs_sigma0: float = 0.1                      # paper App. A.3: sigma_0 = 0.1
    locusgs_bandwidth_floor: float = 1e-6            # paper: "squared bandwidth is lower-bounded"
    locusgs_bias_clamp: float = -20.0                # paper App. A.5: clamp [-20, 0]
    locusgs_ray_bias_scale: float = 1.0              # 0 disables the geometric bias (ablation)
    locusgs_pe_num_freqs: int = 4                    # sinusoidal PE bands (unspecified)
    locusgs_pe_hidden_dim: int = 1024                # PE MLP hidden dim (unspecified)
    # Both are unspecified by the paper; they are the two switches audited in the
    # stability matrix (gamma init and how the anchor embedding enters the layer).
    locusgs_gamma_raw_init: float = 0.0               # 0 -> gamma = softplus(0) = 0.693
    # Eq. 2 is written as a single `q_tilde = q + p` before self-attention.  The
    # stability audit showed the "persistent" reading (adding the anchor PE to the
    # residual stream every layer) inflates ||tokens|| 4.2 -> 50 and destabilizes
    # anchors, so the formal default is the non-persistent "injected" reading:
    # `tokens = tokens + self_attn(tokens + p)`.
    locusgs_pe_mode: Literal["persistent", "injected"] = "injected"
    # "Predefined initial support radius" (unspecified by the paper).  Chosen so
    # sigma_0 * r0 is commensurate with the measured ScanNet anchor-to-ray
    # distances (median 0.030, p95 0.050): r0 = 0.15 keeps the geometric bias
    # informative instead of fully clamped.
    locusgs_radius_init: float = 0.15
    locusgs_radius_epsilon: float = 1e-4
    # Random anchor init box (unspecified by the paper); we use the measured
    # ScanNet normalized surface extent (|x|,|y| p99 ~ 0.2).
    locusgs_anchor_init_extent: float = 0.2
    locusgs_anchor_init_center_z: float = 0.25
    locusgs_supervised_layers: tuple[int, ...] = (6, 12)   # paper Table 5: {6, 12} is best
    canonical_gaussian_visibility_weight: float = 1.0      # paper: lambda_G = 1.0
    canonical_anchor_visibility_weight: float = 0.1        # paper: lambda_A = 0.1
    # --- joint one-stage loss curriculum and spatial regularization ---
    understanding_warmup_steps: int = 2000
    understanding_start_weight: float = 0.1
    understanding_final_weight: float = 1.0
    spatial_compactness_weight: float = 1e-3
    spatial_radius_weight: float = 1e-3
    gradient_diagnostic_freq: int = 200
    init_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.dec_patch_size is None:
            self.dec_patch_size = self.patch_size
        self.validate()

    def validate(self) -> None:
        if self.evaluating:
            assert not self.use_input_supervision, "use_input_supervision must be False when evaluating"
        if self.mean_of_grads not in ("none", "per-scene", "per-view"):
            raise ValueError("mean_of_grads must be one of: none, per-scene, per-view")
        if self.deferred_bp and self.mean_of_grads != "none":
            raise ValueError("deferred_bp and mean_of_grads are mutually exclusive backprop strategies")
        if self.mean_of_grads_scene_chunk_size <= 0:
            raise ValueError("mean_of_grads_scene_chunk_size must be positive")
        if self.mean_of_grads_view_chunk_size is not None and self.mean_of_grads_view_chunk_size <= 0:
            raise ValueError("mean_of_grads_view_chunk_size must be positive")

    def evolve(self, **changes: Any) -> Options:
        """Return a deep copy with the given fields replaced."""
        new_instance = copy.deepcopy(self)
        for key, value in changes.items():
            if not hasattr(new_instance, key):
                raise AttributeError(f"Options has no attribute '{key}'")
            setattr(new_instance, key, value)
        new_instance.validate()
        return new_instance


config_defaults: dict[str, Options] = {}
config_doc: dict[str, str] = {}

config_doc["train_dl3dv_base"] = "DL3DV training defaults (long schedule, capped iters/epoch)."
config_defaults["train_dl3dv_base"] = Options(
    num_epochs=300,
    max_iters_per_epoch=500,
    pct_start_steps=2000,
)

config_doc["finetune_dl3dv_2view"] = "Short finetune from existing tokens, 2 input views, wide images."
config_defaults["finetune_dl3dv_2view"] = config_defaults["train_dl3dv_base"].evolve(
    num_epochs=20,
    pct_start_steps=400,
    lr=4e-5,
    num_gs_tokens=4096,
    init_tokens_from_existing=True,
    num_input_views=2,
    img_size=(256, 448),
)

config_doc["finetune_dl3dv_4view"] = "Like finetune_dl3dv_2view with 4 input views."
config_defaults["finetune_dl3dv_4view"] = config_defaults["finetune_dl3dv_2view"].evolve(
    num_input_views=4,
)

config_doc["finetune_dl3dv_6view"] = "Like finetune_dl3dv_2view with 6 input views and 10 total views."
config_defaults["finetune_dl3dv_6view"] = config_defaults["finetune_dl3dv_2view"].evolve(
    num_input_views=6,
    num_views=10,
)

config_doc["eval_dl3dv_2view"] = "DL3DV eval preset: 2 views, eval JSON, single batch."
config_defaults["eval_dl3dv_2view"] = Options(
    data_mode=(("dl3dv_eval_scaled_0.15", 1),),
    dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_2v.json"},
    num_input_views=2,
    img_size=(256, 448),
    evaluating=True,
    num_gs_tokens=4096,
    use_input_supervision=False,
    batch_size=1,
)

config_doc["eval_dl3dv_4view"] = "DL3DV eval preset: 4 input views."
config_defaults["eval_dl3dv_4view"] = config_defaults["eval_dl3dv_2view"].evolve(
    num_input_views=4,
    dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_4v.json"},
)
config_doc["eval_dl3dv_6view"] = "DL3DV eval preset: 6 input views."
config_defaults["eval_dl3dv_6view"] = config_defaults["eval_dl3dv_2view"].evolve(
    num_input_views=6,
    dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_6v.json"},
)


# ----- DL3DV latent-bottleneck release presets -----
_LATENT_D12_ARCH = {
    "enc_depth": 12,
    "dec_depth": 1,
    "dec_patch_size": 8,
    "clip_head_readout_std": 0.002,
    "clip_head_z_init": 0.1,
    "dec_init_values": 0.01,
    "gaussian_z_offset": 0.0,
    "opacity_bias": 2.0,
    "gs_token_std": 0.02,
    "use_multiscale_encoder": True,
    "multiscale_encoder_layers": (5, 7, 9, 11),
    "use_latent_bottleneck": True,
    "num_latents": 4096,
    "latent_cross_attn_depth": 12,
    "camera_normalization_method": "mean_cam",
    "camera_scale_method": "constant",
}

_SSIM_LOSS = {
    "rgb_loss_type": "l1",
    "lambda_rgb": 0.8,
    "lambda_ssim": 0.2,
    "lambda_lpips": 0.0,
}

_LPIPS_LOSS = {
    "rgb_loss_type": "l2",
    "lambda_rgb": 1.0,
    "lambda_ssim": 0.0,
    "lambda_lpips": 0.5,
}


def _latent_dl3dv_train_preset(num_input_views: int, num_views: int) -> Options:
    return config_defaults["train_dl3dv_base"].evolve(
        data_mode=(("dl3dv_scaled_0.15", 6),),
        num_epochs=20,
        pct_start_steps=400,
        lr=4e-5,
        num_gs_tokens=4096,
        init_tokens_from_existing=True,
        init_latents_from_existing=True,
        num_input_views=num_input_views,
        num_views=num_views,
        img_size=(256, 448),
        **_LATENT_D12_ARCH,
    )


config_doc["train_dl3dv_latent_base"] = (
    "Scratch DL3DV latent-bottleneck training base. Uses pointmap scene "
    "rescaling, the 12-layer encoder latent architecture, and no checkpoint "
    "initialization."
)
config_defaults["train_dl3dv_latent_base"] = Options(
    data_mode=(("dl3dv_scaled_1.0", 6),),
    num_epochs=178,
    pct_start_steps=2000,
    use_input_supervision=True,
    rgb_loss_type="l1",
    lambda_rgb=0.8,
    **{**_LATENT_D12_ARCH, "camera_scale_method": "pointmap"},
)


def _latent_dl3dv_eval_preset(num_input_views: int, evaluation_json: str) -> Options:
    return Options(
        data_mode=(("dl3dv_eval_scaled_0.15", 1),),
        dataset_kwargs={"evaluation_json": evaluation_json},
        num_input_views=num_input_views,
        img_size=(256, 448),
        evaluating=True,
        num_gs_tokens=4096,
        use_input_supervision=False,
        ttt_mode="scene-latent-tuning",
        ttt_lr=1e-2,
        batch_size=1,
        **_LATENT_D12_ARCH,
    )


for _num_input_views, _num_views in ((2, 8), (4, 8), (6, 10)):
    _name = f"finetune_dl3dv_latent_{_num_input_views}view_ssim"
    config_doc[_name] = (
        f"Finetune the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view SSIM model."
    )
    config_defaults[_name] = _latent_dl3dv_train_preset(
        _num_input_views, _num_views
    ).evolve(**_SSIM_LOSS)

    _name = f"finetune_dl3dv_latent_{_num_input_views}view_lpips"
    config_doc[_name] = (
        f"Finetune the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view LPIPS model."
    )
    config_defaults[_name] = _latent_dl3dv_train_preset(
        _num_input_views, _num_views
    ).evolve(**_LPIPS_LOSS)

for _num_input_views in (2, 4, 6):
    _eval_json = f"assets/evaluation_idx_dl3dv_depthsplat_{_num_input_views}v.json"

    _name = f"eval_dl3dv_latent_{_num_input_views}view_ssim"
    config_doc[_name] = (
        f"Evaluate the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view SSIM checkpoint."
    )
    config_defaults[_name] = _latent_dl3dv_eval_preset(
        _num_input_views, _eval_json
    ).evolve(**_SSIM_LOSS)

    _name = f"eval_dl3dv_latent_{_num_input_views}view_lpips"
    config_doc[_name] = (
        f"Evaluate the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view LPIPS checkpoint."
    )
    config_defaults[_name] = _latent_dl3dv_eval_preset(
        _num_input_views, _eval_json
    ).evolve(**_LPIPS_LOSS)


# ----- Kubric 4D dynamic finetune -----
config_doc["finetune_dl3dv_kubric_dyn"] = (
    "Dynamic finetune of the DL3DV base on Kubric4D. Adds dynamic GS tokens "
    "warm-started from the static tokens, sinusoidal time embeddings, "
    "interpolated target-frame sampling, and pointmap camera scaling."
)
config_defaults["finetune_dl3dv_kubric_dyn"] = config_defaults["train_dl3dv_base"].evolve(
    use_input_supervision=False,
    data_mode=(("kubric_scaled_1.0", 100),),
    dataset_kwargs={},
    num_input_views=4,
    num_views=8,
    img_size=(256, 256),
    num_epochs=100,
    max_iters_per_epoch=500,
    pct_start_steps=2500,
    lr=4e-5,
    init_tokens_from_existing=True,
    num_gs_tokens=1024,
    num_dynamic_gs_tokens=256,
    init_dynamic_tokens_from_static=True,
    time_embedding=True,
    time_embedding_dim=2,
    use_interp_target=True,
    camera_normalization_method="mean_cam",
    camera_scale_method="pointmap",
    enc_depth=12,
    dec_depth=1,
    dec_patch_size=8,
    clip_head_readout_std=0.002,
    clip_head_z_init=0.1,
    dec_init_values=0.01,
    gaussian_z_offset=0.0,
    opacity_bias=2.0,
    gs_token_std=0.02,
    use_multiscale_encoder=True,
    multiscale_encoder_layers=(5, 7, 9, 11),
    use_latent_bottleneck=True,
    num_latents=4096,
    latent_cross_attn_depth=12,
    rgb_loss_type="l1",
    lambda_rgb=0.8,
    lambda_ssim=0.2,
    lambda_visibility=1.0,
    lambda_opacity=0.0,
    project_name="TokenGS-Kubric",
)

config_doc["finetune_dl3dv_kubric_static"] = (
    "Stage-1 Kubric domain finetune without dynamic tokens or time embeddings. "
    "Use this when reproducing the two-stage warm-start variant."
)
config_defaults["finetune_dl3dv_kubric_static"] = config_defaults["finetune_dl3dv_kubric_dyn"].evolve(
    num_dynamic_gs_tokens=0,
    init_dynamic_tokens_from_static=False,
    time_embedding=False,
)

config_doc["finetune_dl3dv_kubric_dyn_v2"] = (
    "Kubric dynamic finetune with an auxiliary dynamic-only render loss."
)
config_defaults["finetune_dl3dv_kubric_dyn_v2"] = config_defaults["finetune_dl3dv_kubric_dyn"].evolve(
    lambda_dyn_aux=0.3,
)

_kubric_dyn_release = config_defaults["finetune_dl3dv_kubric_dyn_v2"].evolve(
    lambda_dyn_aux=0.3,
    lambda_dyn_aux_warmup_steps=5000,
    lambda_dyn_aux_decay_steps=10000,
    lambda_dyn_aux_min=0.0,
)
config_doc["finetune_dl3dv_kubric_dyn_release"] = (
    "Released Kubric dynamic finetune schedule: hold lambda_dyn_aux=0.3 for "
    "5K steps, then linearly decay it to 0 over 10K steps."
)
config_defaults["finetune_dl3dv_kubric_dyn_release"] = _kubric_dyn_release
config_doc["finetune_dl3dv_kubric_dyn_v3"] = (
    "Backward-compatible alias for finetune_dl3dv_kubric_dyn_release."
)
config_defaults["finetune_dl3dv_kubric_dyn_v3"] = _kubric_dyn_release

# ----- SSST: spatially grounded shared tokens (local reconstruction unit +
# local understanding unit) with unified object queries -----
_SSST_ARCH = {
    "img_size": (256, 256),
    "patch_size": 8,
    "dec_patch_size": 8,
    "enc_depth": 3,
    "dec_depth": 12,
    "enc_embed_dim": 1024,
    "enc_num_heads": 16,
    "num_gs_tokens": 1024,
    "token_dim": 1024,
    "gs_token_std": 0.01,
    # The anchors already sit where the normalized SIU3R scene is, so the
    # legacy random-initialization z offset is not applied.
    "gaussian_z_offset": 0.0,
    "random_reflect": False,
    "camera_normalization_method": "first_cam",
    "camera_scale_method": "constant",
    "use_instance_labels": True,
    "num_input_views": 2,
    "num_views": 4,
}

_SSST_LOSS = {
    "rgb_loss_type": "l2",
    "lambda_rgb": 1.0,
    "lambda_ssim": 0.0,
    "lambda_lpips": 0.5,
    "lambda_visibility": 0.0,
    "lambda_mask": 0.0,
    "lambda_opacity": 0.0,
}

_SSST_DATA = {
    "data_mode": (("siu3r_processed_scannet", 1),),
    "dataset_kwargs": {"data_root": "/space/mawb/SIU3R/data/scannet"},
}

config_doc["train_siu3r_ssst"] = (
    "One-stage SIU3R joint training with spatially grounded shared tokens "
    "(2 context views, 4 supervised records) and 100 unified object queries. "
    "Curriculum: the understanding loss ramps from 0.1 to 1.0 over 2000 steps."
)
config_defaults["train_siu3r_ssst"] = Options(
    model_type="siu3r_joint_ssst",
    workspace="/space/mawb/ssst/workspace/siu3r_ssst_joint_v1",
    experiment_name="siu3r_ssst_joint_v1",
    project_name="TokenGS-SSST",
    num_epochs=20,
    max_iters_per_epoch=1_000_000,
    batch_size=1,
    gradient_accumulation_steps=3,
    lr=1e-4,
    pct_start_steps=1000,
    gradient_clip=1.0,
    mixed_precision="bf16",
    # In-process loading keeps the pair-order audit active and avoids many
    # readers on the shared filesystem; raise it for throughput if needed.
    num_workers=0,
    seed=42,
    **_SSST_ARCH,
    **_SSST_LOSS,
    **_SSST_DATA,
)

config_doc["train_siu3r_ssst_re10k"] = (
    "Same as train_siu3r_ssst but warm-started from the released RE10K "
    "reconstruction checkpoint. Only name/shape-compatible tokens are loaded; "
    "anchors, refinement, queries and the understanding head stay fresh."
)
config_defaults["train_siu3r_ssst_re10k"] = config_defaults["train_siu3r_ssst"].evolve(
    workspace="/space/mawb/ssst/workspace/siu3r_ssst_joint_re10k_init_v1",
    experiment_name="siu3r_ssst_joint_re10k_init_v1",
    init_checkpoint=(
        "/space/mawb/tokengs_siu3r_joint_v1/workspace/"
        "siu3r_joint_re10k_init_context_order_fixed_v1_v3/checkpoints/step_00001000"
    ),
)

config_doc["eval_siu3r_ssst"] = (
    "SSST validation configuration (2 context + 4 novel records) used when a "
    "checkpoint directory does not ship its own config.yaml."
)
config_defaults["eval_siu3r_ssst"] = config_defaults["train_siu3r_ssst"].evolve(
    evaluating=True,
    num_views=6,
    batch_size=1,
    mixed_precision="no",
    random_reflect=False,
)

# ----- LocusGS-faithful ScanNet reconstruction + plain TokenGS control -----
# Canonical reconstruction objective (arXiv:2608.12825 Eq. 26-30): MSE +
# 0.2*(1-SSIM)/2 + 1.0*L_vis(Gaussians) + 0.1*L_vis(anchors), no LPIPS, no GT
# depth, no spatial hinge/compactness terms.  Optimizer kept identical to the
# previous ScanNet runs (lr 1e-4, warmup 1000, wd 0.05) so the only change is
# architecture + objective (Variant B of the audit).
_CANONICAL_RECON = {
    "rgb_loss_type": "l2",
    "lambda_rgb": 1.0,
    "lambda_ssim": 0.2,
    "lambda_lpips": 0.0,
    "lambda_visibility": 0.0,   # visibility is added explicitly (Gaussians + anchors)
    "lambda_mask": 0.0,
    "lambda_opacity": 0.0,
    # The paper adopts "the visibility regularization used in TokenGS", whose
    # implementation clips the per-point penalty at this threshold.  Without the
    # clip, anchors that project behind the camera give an unbounded phi.
    "visibility_distance_threshold": 1.0,
    "spatial_compactness_weight": 0.0,      # our own terms are removed for faithfulness
    "spatial_radius_weight": 0.0,
    "reconstruction_only": True,
    # TokenGS's canonical z offset: the free-XYZ head predicts positions near the
    # origin at initialization, so without it nothing renders (alpha = 0) and the
    # RGB objective has no gradient.  LocusGS keeps 0.0 because its randomly
    # initialized anchors already lie in the normalized scene box.
    "gaussian_z_offset": 1.0,
}

config_doc["train_siu3r_locusgs_recon"] = (
    "LocusGS-faithful ScanNet reconstruction (2 context + 2 novel): learnable "
    "anchor centers + softplus radii, anchor PE injected as the self-attention "
    "input only (pe_mode=injected, not accumulated into the residual stream), "
    "anchor-to-ray bias (sigma0=0.1, clamp [-20,0], learnable gamma), raw "
    "residual anchor refinement, anchor-centered Gaussian decoding, and "
    "multi-layer supervision at decoder layers {6,12} with weights {1/3, 2/3}."
)
config_defaults["train_siu3r_locusgs_recon"] = config_defaults["train_siu3r_ssst"].evolve(
    model_type="siu3r_locusgs_recon",
    workspace="/space/mawb/ssst/workspace/siu3r_locusgs_faithful_recon_v1",
    experiment_name="siu3r_locusgs_faithful_recon_v1",
    project_name="TokenGS-LocusGS",
    # Paper-reported optimisation scale (Sec. 4 / App.: AdamW, base lr 4e-4, 2000-step
    # warmup then cosine).  The ScanNet adaptation keeps the same scale; the
    # effective batch differs from the paper, so this is a recipe adaptation.
    lr=4e-4,
    pct_start_steps=2000,
    **{**_CANONICAL_RECON, "gaussian_z_offset": 0.0, "locusgs_pe_mode": "injected"},
)

config_doc["train_siu3r_plain_tokengs_canonical_recon"] = (
    "Plain TokenGS (free-XYZ ClipActivationHead, no anchors/radius/refinement/"
    "ray-bias, final layer only) under the same canonical reconstruction "
    "objective and the same ScanNet 2+2 protocol: the controlled baseline."
)
config_defaults["train_siu3r_plain_tokengs_canonical_recon"] = config_defaults[
    "train_siu3r_ssst"
].evolve(
    model_type="siu3r_plain_tokengs_canonical_recon",
    workspace="/space/mawb/ssst/workspace/siu3r_plain_tokengs_canonical_recon_v1",
    experiment_name="siu3r_plain_tokengs_canonical_recon_v1",
    project_name="TokenGS-LocusGS",
    **_CANONICAL_RECON,
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
