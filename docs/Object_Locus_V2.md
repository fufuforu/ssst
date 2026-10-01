# Object-Locus V2

Versioned feed-forward architecture: `LOCUSGS_OBJECT_LOCUS_V2`, registered as
`siu3r_object_locus_v2` with preset `train_siu3r_object_locus_v2`.

## State and anchor masks

V2 retains the V1.1 state path: 1024 anchors, registered decoder layers 6/8/10/12,
256-dimensional states, 8-head evidence reading, 100 thing states and two stuff
states. Layer 6 initializes `(q,c,s)`. Each registered layer reads evidence over all
1024 anchors with the existing soft geometry bias, applies the residual object
decoder, updates `(c,s)`, then predicts independent anchor masks. The V2 mask head
uses 256-dimensional anchor/query features:

```text
f_anchor = LN_mask_a(W_mask_a(a))
m_query  = Linear256(GELU(Linear256(LN_mask_q(q))))
b_query = mask_bias(LN_mask_q(q))
L_A = einsum(f_anchor, m_query) / sqrt(256) - b_query
anchor_membership = sigmoid(L_A)  # [B,1024,102]
```

The 102 masks are 100 thing hypotheses, wall, and floor. They are independent
sigmoids; there is no void mask channel, channel softmax, or c/s mask bias. Existing
16-dimensional identity features remain a separate reconstruction feature.

## Gaussian children and rendered masks

Canonical reconstruction still emits 1024×64 Gaussians with unchanged geometry,
opacity, and RGB. Each child forms the fixed 14D geometry/color descriptor from
normalized xyz offset, normalized log scale, rotation, opacity, and RGB. A learned
16D child-index embedding and shared 286→256→256 MLP produce a residual added to
the parent anchor mask feature. Its last linear is initialized to zero, so initial
child logits equal parent logits; training may separate siblings without changing
Gaussian reconstruction values.

```text
L_G = dot(f_gaussian, m_query) / sqrt(256) - b_query
M_G = sigmoid(L_G)
membership_mass, identity_render, alpha = render_feature_channels(G, [M_G,e_id])
M_pixel = clamp(membership_mass / clamp_min(alpha,1e-6), 0,1), alpha==0 -> 0
```

Membership channels are not normalized across objects. Semantic scores combine
wall/floor masks and thing masks weighted by the 18 foreground class probabilities,
then normalize over the 20 semantic classes with the registered `1e-6` denominator.

## Matching, losses, and inference

One L12 Hungarian assignment uses class, deterministic valid context pixels (up to
4096), and trusted visible-anchor costs with weights `1/5/5/2/2`. L6/L8/L10 reuse
the same scene-global pairs. Anchor masks use BCE-with-logits and probability Dice;
unsupported thing GTs are excluded from thing-anchor averages, and wall/floor are
supervised only on trusted anchors. Pixel BCE and Dice act directly on rendered
probabilities. The retained understanding weighting is `0.1` thing, `0.1` stuff,
`0.1` semantic, `0.01` identity, and `0.1` anchor-group, plus the fixed auxiliary
term. GC applies 0.01 to understanding gradients entering shared reconstruction
parameters and 1.0 to object-branch understanding gradients.

Panoptic inference is feed-forward and GT-free. Candidate things require a
foreground class argmax and max foreground probability ≥0.05. Raw binary masks use
membership ≥0.5 and alpha >0.05. Things compete by class confidence×membership;
stuff competes by membership. Lowest channel wins exact ties. A thing losing at
least half its precompetition area is removed and its pixels remain void. Export
scores use the scope's final assigned pixels and are generated separately for
context and target scopes.

## Training stages

Stage A starts from the locked pretrained reconstruction and fresh V2 branch. It
cycles the fixed monitor_train16 windows in file order for 1500 updates, with
understanding weight `min(step/100,1)` and a 50-step linear LR warmup to object
`1e-4` / reconstruction `1e-5`. Stage A gates use raw independent-mask IoU≥0.5
fraction ≥0.40, matched classification accuracy ≥0.60, and context and novel PSNR
drops ≤0.5dB from this version's step 0. Stage B runs only if all gates pass; it
continues optimizer state for 5000 updates over the locked 128-scene plan. B-stage
uses the registered 100-step LR warmup followed by cosine schedule, with no
automatic validation-based stopping or tuning.

Seed is 42 and V2 branch initialization seed is 31415. Training is joint FP32,
batch size one, AdamW `(0.9,0.95)`, weight decay 0.05 with registered no-decay
parameters, global clip 1.0, and GC alpha 0.01. No V1/V1.1 object weights transfer.

The historical Object-Locus V1 failure at step 4090 remains `UNKNOWN`; V2 does not
claim to repair or reproduce that failure.
