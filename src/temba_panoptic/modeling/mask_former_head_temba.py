
from __future__ import annotations

from typing import Dict

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead

from .temba_adapter import MultiScaleTembaAdapter

@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHeadTemba(MaskFormerHead):
    @configurable
    def __init__(
        self,
        *,
        num_frames: int,
        multi_scale_in_channels: int,
        adapter_dim: int,
        adapter_depth: int,
        local_kernel_size: int,
        local_dilations,
        dts_dilations,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.temba_adapters = MultiScaleTembaAdapter(
            in_channels=multi_scale_in_channels,
            adapter_dim=adapter_dim,
            adapter_depth=adapter_depth,
            local_kernel_size=local_kernel_size,
            local_dilations=tuple(local_dilations),
            dts_dilations=tuple(dts_dilations),
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret.update(
            {
                "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
                "multi_scale_in_channels": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
                "adapter_dim": cfg.MODEL.TEMBA.ADAPTER_DIM,
                "adapter_depth": cfg.MODEL.TEMBA.ADAPTER_DEPTH,
                "local_kernel_size": cfg.MODEL.TEMBA.LOCAL_KERNEL_SIZE,
                "local_dilations": list(cfg.MODEL.TEMBA.LOCAL_DILATIONS),
                "dts_dilations": list(cfg.MODEL.TEMBA.DTS_DILATIONS),
                "dropout": float(cfg.MODEL.TEMBA.DROPOUT),
            }
        )
        return ret

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        del transformer_encoder_features
        multi_scale_features = self.temba_adapters(multi_scale_features, num_frames=self.num_frames)
        return self.predictor(multi_scale_features, mask_features, mask)
