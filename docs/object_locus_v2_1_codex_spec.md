# Object-Locus V2.1

Object-Locus V2.1 retains the V2 scene-conditioned states, evidence reading,
dynamic geometry, independent anchor and Gaussian masks, Gaussian child feature
residual, identity path, rendering, visibility, and reconstruction objective.
The sole model-side change is the mask-conditioned category classifier with a
separate objectness head, plus the registered small-stage / expansion data plan.
This is a fixed candidate design; implementation and training completion do not
establish cross-scene understanding.

## Classification readout

At each registered layer, thing membership weights pool the 1024 anchor
features. L6/L8/L10 use that pooled feature for the shared classifier. L12 first
creates child-specific Gaussian features and independent Gaussian memberships,
then pools Gaussian features with membership times detached Gaussian opacity;
mass below `1e-6` falls back to L12 anchor pooling. The shared classifier fuses
normalized thing state and pooled scene feature through `Linear(512,256)`, GELU,
LayerNorm, then separate 18-way category and scalar objectness heads.

`P19(k)=sigmoid(objectness)*softmax(category)[k]` for 18 thing categories and
`P19(18)=1-sigmoid(objectness)`. A stable 19-channel log-probability preserves
the prior matching/export interface. It is not used for the old unmatched-slot
category CE. Category CE trains matched slots only; objectness BCE separately
averages matched positive and unmatched negative slots and combines the two
means with equal weight when both sets are present.

The final thing objective is `2*category_CE + 2*objectness_BCE +
5*pixel_BCE + 5*pixel_Dice`; final understanding preserves the V2 weights
`0.1*thing + 0.1*stuff + 0.1*semantic + 0.01*identity + 0.1*anchor`. Each
auxiliary layer uses `0.2*category_CE + 0.2*objectness_BCE + 0.1*anchor`, and
the auxiliary mean is weighted by `0.25`. Legacy unmatched-class CE is unused.

Final Hungarian matching remains single-pass at L12, using the Gaussian-pooled
classifier and the existing V2 mask/anchor matching costs. Its pairs are reused
for L6/L8/L10 auxiliary category, objectness, and anchor-mask terms. Semantic
readout uses joint `P19`; panoptic filtering, confidence, thresholds, label
mapping, and official evaluator remain fixed to the specification.

Official inference keeps the fixed rule: a query must have foreground joint
probability at least `0.05` and the joint argmax must not be no-object. Binary
masks use membership `>=0.5` and alpha `>0.05`; thing/stuff candidates compete
per pixel, and a thing is removed if its winning area is less than half its raw
area. Removed pixels remain void. AP confidence is joint class probability
times mean membership over assigned pixels.

## Fresh initialization and training plan

Reconstruction starts from the SHA-locked pretrained step 47500 checkpoint;
the object branch starts fresh with global seed 42 and object seed 31415. All
reconstruction and object parameters remain trainable. FP32, batch one, AdamW
(`betas=(0.9,0.95)`, `eps=1e-8`), object/reconstruction peak LR `1e-4/1e-5`,
weight decay `0.05`, global clip `1.0`, and shared understanding gradient scale
`0.01` are fixed. There is no object-to-anchor feedback.

The deterministic frame-disjoint split has 16 small-stage scenes, 112 small
training windows (7 per scene), and 16 same-scene held-out windows. The selected
dev8 scenes are checked against the full 128-scene training pool. Expanded
training uses 1008 remaining windows across 128 scenes after excluding 16
holdout-overlapping windows. Stage S has 16 epochs / 1792 updates, each window
seen 16 times. Stage E, only if every registered task gate passes, has 16 epochs
/ 16128 updates, continuing the same optimizer and model. The full maximum is
17920 updates. It is not a fresh 5000-step run.

Stage S expansion gates use raw mask IoU, conditional and joint classification,
matched objectness recall, class-agnostic/aware output, official AP50, and
context/true-novel PSNR against S0. Failure to pass ends training normally; no
automatic adjustment or extra epochs are allowed. Passing only authorizes the
pre-registered Stage E and is not a claim of successful generalization.

The experiment uses the registered GT camera poses and therefore is not the
full unposed SIU3R benchmark. Official evaluation thresholds and evaluator code
are not changed.
