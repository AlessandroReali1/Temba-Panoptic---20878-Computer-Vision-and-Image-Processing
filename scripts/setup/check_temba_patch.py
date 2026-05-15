#!/usr/bin/env python3
import os
import sys
from pathlib import Path

detectron2_root = os.environ.get("DETECTRON2_ROOT")
mask2former_root = os.environ.get("MASK2FORMER_ROOT")

if detectron2_root:
    sys.path.insert(0, detectron2_root)
if mask2former_root:
    sys.path.insert(0, mask2former_root)

print("DETECTRON2_ROOT:", detectron2_root)
print("MASK2FORMER_ROOT:", mask2former_root)

import mask2former_video.temba_config
import mask2former_video.data_video.datasets.register_jrdb_vis_from_panoptic
import mask2former_video.modeling.temba_adapter
import mask2former_video.modeling.mask_former_head_temba
import mask2former_video.video_maskformer_temba
import mask2former_video.video_maskformer_temba_e2e

from detectron2.modeling import META_ARCH_REGISTRY, SEM_SEG_HEADS_REGISTRY

print("TEMBA imports: OK")

META_ARCH_REGISTRY.get("VideoMaskFormerTembaE2E")
SEM_SEG_HEADS_REGISTRY.get("MaskFormerHeadTemba")

print("Detectron2 registries: OK")
print("VideoMaskFormerTembaE2E and MaskFormerHeadTemba are registered.")
