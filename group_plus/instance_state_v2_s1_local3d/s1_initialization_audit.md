# s1_initialization_audit

Baseline HEAD: `29d0c80f74279521d1f131f2d4f59bf92ca347fa` (S0 artifacts are read-only).

## What S0 actually does at layer 6

```python
sel = deterministic_fps(mu.detach(), NUM_THING)   # 1024 layer-6 anchors -> 100 seed indices
x6  = controller.encode_token(tokens, mu, radii, ell)   # [B,1024,D] token features
q_thing = query_init[:100] + gather_tokens(x6, sel)     # each q_j reads ONLY x6[fps_j]
c = gather_tokens(mu, sel)                               # seed centre per state
s = ell                                                # isotropic support
```

So the 100 thing states are seeded by **one anchor feature each**, chosen purely by
deterministic farthest-point sampling of the 1024 layer-6 centres.  Nothing else in the
state formation is local: the only spatial knowledge in `q` is the identity of that single
token.

## The single S1 change

`q_thing = query_init[:100] + local_3d_evidence_pool(x6, mu, sel, k=8).pooled`

* FPS call, `c`, `s`, the stuff initialisation and every downstream module are untouched;
* distance, neighbour selection and weights are computed from `mu.detach()`, so the pooling
  adds no geometry gradient path;
* the pooled feature keeps the normal gradient to `x6`;
* **zero new learnable parameters**.

Manifest used for both arms: `train128_windows1024.json` (128 scenes /
1024 windows), sha256 `1f37d08c...`.
