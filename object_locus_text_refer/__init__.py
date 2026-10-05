"""Text referring segmentation over frozen Object-Locus instances."""

from .head import ObjectLocusTextReferHead, soft_gaussian_membership, hard_gaussian_membership

__all__ = ["ObjectLocusTextReferHead", "soft_gaussian_membership", "hard_gaussian_membership"]
