"""The validated B architecture, trained jointly from Baseline4's zero state."""
from tokengs.models.reconstruction_abcd import variants
from scripts.reconstruction4096_scratch_policy import ARCH


class Reconstruction4096Scratch4(variants()['reconstruction_b']):
    architecture_name = ARCH
