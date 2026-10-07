"""Method-only competition-supervised child of the registered Panoptic V1."""
from __future__ import annotations

from tokengs.models.object_locus_panoptic_v1 import LocusGSObjectLocusPanopticV1Recon
from tokengs.models.object_locus_competition_v1_loss import (
    COMPETITION_LAMBDA, EXTERNAL_WEIGHT, context_competition_loss,
)


class LocusGSObjectLocusCompetitionV1Recon(LocusGSObjectLocusPanopticV1Recon):
    """Adds a loss method only; architecture and forward remain inherited."""

    def step_loss(self, batch, *, step, phase="train", coupled=None,
                  understanding_weight=1.0):
        result, metrics = super().step_loss(
            batch, step=step, phase=phase, coupled=coupled,
            understanding_weight=understanding_weight,
        )
        lam = float(getattr(self, "competition_lambda", COMPETITION_LAMBDA))
        if lam == 0.0:
            return result, metrics
        if lam != COMPETITION_LAMBDA:
            raise ValueError(f"registered competition lambda is fixed at {COMPETITION_LAMBDA}")
        prediction = result["prediction"]
        old = metrics["loss_understanding"]
        comp = context_competition_loss(prediction, batch)
        added = EXTERNAL_WEIGHT * lam * comp
        combined = old + added
        metrics.update(
            loss_under_old=old,
            loss_competition=comp,
            weighted_loss_competition=added,
            loss_under_new=combined,
            loss_understanding=combined,
            loss=metrics["loss_recon"] + float(understanding_weight) * combined,
            loss_total=metrics["loss_recon"] + float(understanding_weight) * combined,
        )
        return result, metrics
