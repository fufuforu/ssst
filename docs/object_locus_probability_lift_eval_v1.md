# Object Locus probability lifting paired evaluation

This branch adds a readout-only P arm for the locked epoch 6 Full1201 checkpoint. The C arm calls the inherited feature-domain readout directly. P computes the final-query pixel mask logits after 256×256 feature upsampling, applies sigmoid before renderer-transpose lifting, blends with child probabilities using the original visibility gate, then adds the original `W_res` logit residual. No state is added to the model.

The fixed runner is `scripts/eval_object_locus_probability_lift_v1.py`. It requires one allocated RTX 4090 on 3dimage-17, FP32, disabled TF32, exposure 50064 and the pinned SIU3R checkout. The paired official export roots and reports are written under `/space/mawb/ssst/group_plus/object_locus_probability_lift_eval_v1`.

Run `scripts/check_object_locus_probability_lift_v1.py` for the small CPU arithmetic contract. Submit `scripts/submit_object_locus_probability_lift_eval_v1.sh` only after pushing the task branch, as required by the task prompt. The batch script runs the two-window smoke before the full registered Val32 paired evaluation. It does not train or update model parameters.

The final classification must use the prompt's registered SUCCESS, FAILURE, INCONCLUSIVE, and INVALID thresholds. This is a same-checkpoint inference comparison and does not establish the efficacy of a retrained model.
