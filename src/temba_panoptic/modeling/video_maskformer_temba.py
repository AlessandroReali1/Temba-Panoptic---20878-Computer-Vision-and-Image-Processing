
from __future__ import annotations

from detectron2.modeling import META_ARCH_REGISTRY

from mask2former_video.video_maskformer_model import VideoMaskFormer

@META_ARCH_REGISTRY.register()
class VideoMaskFormerTembaFrozen(VideoMaskFormer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._freeze_everything_except_temba_and_class_embed()

    @classmethod
    def from_config(cls, cfg):
        return super().from_config(cfg)

    def _freeze_everything_except_temba_and_class_embed(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if hasattr(self.sem_seg_head, "temba_adapters"):
            self.sem_seg_head.temba_adapters.requires_grad_(True)

        predictor = getattr(self.sem_seg_head, "predictor", None)
        if predictor is not None and hasattr(predictor, "class_embed"):
            predictor.class_embed.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if hasattr(self, "backbone"):
                self.backbone.eval()
            if hasattr(self, "sem_seg_head") and hasattr(self.sem_seg_head, "pixel_decoder"):
                self.sem_seg_head.pixel_decoder.eval()
            predictor = getattr(self.sem_seg_head, "predictor", None)
            if predictor is not None:
                predictor.eval()
                if hasattr(predictor, "class_embed"):
                    predictor.class_embed.train()
            if hasattr(self.sem_seg_head, "temba_adapters"):
                self.sem_seg_head.temba_adapters.train()
        return self
