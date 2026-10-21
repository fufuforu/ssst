# Paired reconstruction diagnostics during the A/B continuation

User request, 2026-10-12: examine causes of the approximately 5 dB reconstruction drop beyond insufficient training and feature adaptation. This separately authorized diagnostic work leaves both ongoing training jobs and their official evaluations unchanged.

Inference only; zero optimizer updates. Fixed legacy epoch-6 checkpoint and adapted-2/frozen-joint-4 checkpoint, with SHA checks. Seed-42 selection of one window in each of 16 training and 16 held-out scenes; no selection by performance. Identical provider RGB hashes and frame IDs required. Training windows have four views; validation windows have six. Report each split separately and average all requested views, without separate context/target/novel tables. Small diagnostic samples are not official full-validation scores.

The legacy 24.4536 number used novel frames; its all-view value was 24.9172. The adapted pose-free checkpoint has all-view PSNR 19.3692. The difference remains approximately 5.55 dB under matching aggregation. Legacy used GT geometry; the current version predicts it, so the architecture substitution changed more than appearance features.

| Intervention | What it probes |
| --- | --- |
| old_locus_gt | Paired old baseline; existing GT-posed protocol |
| current | Exact native pose-free source model |
| same_gs_gt_k | Intrinsic errors, same Gaussians and predicted poses |
| same_gs_gt_pose_baseline | Rotation/translation-direction errors, same predicted intrinsics |
| same_gs_gt_camera_baseline | Combined camera oracle, context-baseline scale matched |
| same_gs_gt_camera_depth | Combined camera oracle, GT context depth median normalized to 0.25 |
| same_features_gt_rays_baseline | GT geometry in decoder rays and renderer, cached actual VGGT features |
| same_features_gt_rays_legacy_units | As above with original 0.15-metric camera units; probes decoder spatial priors |
| feedback_off | Disable understanding-to-GS decoder feedback during inference |
| predicted_ray_scale_0.5 / 2.0 | Scale decoder conditioning and render camera translations, with unchanged learned anchor priors |

All GT variants are **DIAGNOSTIC ONLY** and excluded from formal scores. Never substitute raw world poses into a first-camera scene. Explicit GT gauges are first-camera alignment followed by either positive baseline-norm ratio or GT context median depth normalization. These are isolated diagnostic interventions, not changes to the production shared_context_depth_sim3_v2 calibrator. No source features or checkpoints are overwritten. Feature tensors remain transient in GPU memory.

Identity gates: direct RGB rendering equals the native render_generated_at output; cached-feature replay reproduces native RGB; unchanged Gaussians under camera interventions; old/new model parameter versions remain fixed; paired provider RGB hashes match. GT oracles can worsen PSNR because the learned representation compensates for predicted geometry or because the gauge remains imperfect. A negative oracle result does not prove geometry is correct. Feedback-off and scale sweeps are distribution shifts, so they identify sensitivity rather than prove a trained alternative will improve.

Each window stores per-image floating RGB PSNR using SIU3R's default dynamic GT range, MSE, alpha coverage, and display PNGs. PNGs are not the source of the floating score. Aggregation includes paired per-window deltas and standard errors; 16 scenes are exploratory evidence, not a precision benchmark. The script does not optimize cameras, reset weights, change losses, or run training.

CPU contracts cover common-frame invariance, distinct unit gauges, degenerate inputs, and K resizing. Real GPU gates must pass before interpreting the intervention scores. Launch uses an immutable committed snapshot on one available GPU of node 17 or 18, without competing with A/B's eight-GPU nodes 11/13. Keep failed attempt evidence and fix ordinary engineering errors before retrying a fresh diagnostic directory.
