"""1024 FPS reconstruction anchors; native 256D understanding is retained."""
import os
from tokengs.models.superpoint_dual_core import SuperpointDualCore
from tokengs.models.reconstruction_fps_superpoint import FPSWithSuperpointFeatures


class FPSSuperpointDualCore(SuperpointDualCore):
    def __init__(self, opt, config):
        if os.environ.get('RG_IOUHEAD') not in (None, '', '0'):
            raise ValueError('FPS-superpoint control uses baseline understanding without IoU-head changes')
        super().__init__(opt, config)
        self.reconstruction_fps = FPSWithSuperpointFeatures()

    def initialize_reconstruction_anchors(self, reconstruction_seed, cloud=None):
        return self.reconstruction_fps(cloud, reconstruction_seed,
            self.reconstruction_seed_projection(reconstruction_seed['query']))
