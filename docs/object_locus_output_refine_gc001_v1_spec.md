# Object-Locus output-side 3D evidence refinement, R3D

R3D is the preregistered output-side refinement experiment on the complete
Full1201 epoch-6 model. It adds one pre-LN 8-head cross-attention over every
decoded Gaussian and one pre-LN 256→512→256 FFN after the existing lifting and
child-feature fusion, immediately before the final mask embedder and class head.

The existing decoder, L6/L8/L10/L12 feedback and routing, reconstruction and
V3-Set objectives, renderer, candidate selection, and packed-panoptic readout
remain inherited unchanged. The refiner uses detached current XYZ/C/S only to
form the thing-query geometry bias; query and fused Gaussian feature gradients
remain live. Stuff queries receive zero geometry bias. All 65,536 Gaussians
participate in each query's softmax denominator. Queries are chunked by eight
with non-reentrant activation checkpointing during training.

The refiner is zero-output initialized: W_O and W_2 are zero, so construction
preserves q exactly. The final mask and class heads both read q_refined. The
decoder's q remains available as q_base in state and top-level output; its
anchor masks and c/s remain the decoder outputs. The existing matcher and
classification loss consume the refined `states[-1].thing_logits19` through
the unchanged V3-Set loss contract.

Training resumes only from the strictly loaded Full1201 epoch-6 source. Four
physical RTX 3090 ranks each process slots r and r+4 as separate batch-1
microbatches, accumulate one-half of each slot gradient, all-reduce once per
update, clip once, and step once. This preserves the fixed eight-window plan
and global batch eight without claiming bitwise reduction equivalence to the
prior eight-rank execution. The registered coefficient remains alpha=0.01.

The one Slurm allocation runs single-card smoke, four-card accumulation smoke,
then fresh-process formal training. It does not run evaluation. Formal
`startup_confirmation.json` certifies only the first twenty optimizer updates;
training continues to update 1008 and then waits for user notification before
any registered evaluation.
