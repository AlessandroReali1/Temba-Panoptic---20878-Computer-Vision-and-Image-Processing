import torch.nn as nn
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY

from .video_maskformer_temba import VideoMaskFormerTembaFrozen

@META_ARCH_REGISTRY.register()
class VideoMaskFormerTembaE2E(VideoMaskFormerTembaFrozen):
    def train(self, mode: bool = True):
        """
        End-to-end TEMBA model: use the standard PyTorch training behavior.

        This intentionally bypasses the frozen variant train() method, which
        keeps Mask2Former submodules in eval mode.
        """
        nn.Module.train(self, mode)
        return self

    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Undo all freezing done by the frozen variant.
        for p in self.parameters():
            p.requires_grad = True
