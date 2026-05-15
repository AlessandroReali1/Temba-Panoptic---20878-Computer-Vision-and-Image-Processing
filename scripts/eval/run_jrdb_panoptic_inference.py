#!/usr/bin/env python3
from __future__ import annotations

from detectron2.data.detection_utils import read_image

import argparse
import gc
import csv
import json
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

HOME = Path.home()
MASK2FORMER_DIR = Path(os.environ.get("MASK2FORMER_ROOT", "/path/to/Mask2Former"))
TEMBA_PROJECT_DIR = Path(os.environ.get("TEMBA_PROJECT_ROOT", "/path/to/temba_project"))
JRDB_ROOT = Path(os.environ.get("JRDB_PREPROC_ROOT", "/path/to/jrdb_preproc/out"))

sys.path.insert(0, str(MASK2FORMER_DIR))
sys.path.insert(0, str(TEMBA_PROJECT_DIR))
sys.path.insert(0, str(MASK2FORMER_DIR / "demo_video"))

os.environ["JRDB_PREPROC_ROOT"] = str(JRDB_ROOT)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import mask2former_video  # noqa: E402
import mask2former_video.data_video.datasets.register_jrdb_vis_from_panoptic  # noqa: E402

from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import add_maskformer2_config  # noqa: E402
from mask2former_video import add_maskformer2_video_config  # noqa: E402
from predictor import VisualizationDemo  # noqa: E402

MaskLike = Any
Annotation = Dict[str, Any]

def rgb2id(color: np.ndarray) -> np.ndarray:
    color = np.asarray(color, dtype=np.int64)
    if color.ndim == 2:
        return color
    return color[:, :, 0] + 256 * color[:, :, 1] + 65536 * color[:, :, 2]

def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)

def unwrap_predictions(result):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
        return result[0]
    if isinstance(result, list) and len(result) == 2 and isinstance(result[0], dict):
        return result[0]
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unexpected run_on_video return type for predictions: {type(result)}")

def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def decode_mask(mask: MaskLike) -> Optional[np.ndarray]:
    if mask is None:
        return None

    if isinstance(mask, np.ndarray):
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
        return mask.astype(bool)

    if isinstance(mask, list):
        arr = np.asarray(mask)
        if arr.ndim != 2:
            raise ValueError(f"Expected nested list -> 2D mask, got shape={arr.shape}")
        return arr.astype(bool)

    if isinstance(mask, dict):
        decoded = mask_utils.decode(mask)
        if decoded.ndim == 3:
            if decoded.shape[2] != 1:
                raise ValueError(f"Expected single RLE mask, got shape={decoded.shape}")
            decoded = decoded[:, :, 0]
        return decoded.astype(bool)

    raise TypeError(f"Unsupported mask type: {type(mask)}")

def predictions_to_panoptic_tracks(
    predictions: dict,
    video_id: int,
    num_frames: int,
    height: int,
    width: int,
    thing_idx_to_orig_id: Dict[int, int],
) -> List[dict[str, Any]]:
    if "pred_masks" not in predictions or "pred_labels" not in predictions or "pred_scores" not in predictions:
        return []

    pred_masks = to_numpy(predictions["pred_masks"])
    pred_labels = to_numpy(predictions["pred_labels"]).astype(np.int32)
    pred_scores = to_numpy(predictions["pred_scores"]).astype(np.float32)

    if pred_masks.ndim != 4:
        raise RuntimeError(f"Expected pred_masks ndim=4, got shape {pred_masks.shape}")

    # expected output: [N, T, H, W]
    if pred_masks.shape[0] == num_frames:
        pred_masks = np.transpose(pred_masks, (1, 0, 2, 3))
    elif pred_masks.shape[1] != num_frames:
        raise RuntimeError(
            f"Could not align pred_masks with video length. pred_masks shape={pred_masks.shape}, num_frames={num_frames}"
        )

    num_tracks = pred_masks.shape[0]
    output = []

    for idx in range(num_tracks):
        pred_label_contig = int(pred_labels[idx])
        # Defensive: drop predictions with labels outside the thing taxonomy
        # (some Mask2Former configs include an extra "no-object" class).
        if pred_label_contig not in thing_idx_to_orig_id:
            continue
        cat_id_orig = int(thing_idx_to_orig_id[pred_label_contig])
        
        segms = []
        for t in range(num_frames):
            mask = pred_masks[idx, t]
            if mask.dtype != np.bool_:
                mask = mask > 0.5

            if not np.any(mask):
                segms.append(None)
            else:
                segms.append(encode_binary_mask(mask))
        
        # Drop tracks that are empty across the entire video
        if all(s is None for s in segms):
            continue

        output.append(
            {
                "video_id": str(video_id),
                "id": int(idx + 1),
                "category_id": cat_id_orig,
                "score": float(pred_scores[idx]),
                "segmentations": segms,
            }
        )

    return output

def build_gt_panoptic_tracks_for_video(
    video_rec: dict,
    thing_id_map: Dict[int, int],
) -> List[dict[str, Any]]:
    frames = sorted(video_rec["frames"], key=lambda x: int(x["frame_id"]))
    t = len(frames)
    instances: Dict[int, dict[str, Any]] = {}

    for frame_idx, fr in enumerate(frames):
        pan_png = np.asarray(Image.open(fr["panoptic_file_name"]).convert("RGB"), dtype=np.uint8)
        pan_ids = rgb2id(pan_png)

        for seg in fr["segments_info"]:
            if int(seg.get("isthing", 1)) != 1:
                continue
            if int(seg.get("iscrowd", 0)) != 0:
                continue
            if seg.get("track_id", None) is None:
                continue

            orig_cat_id = int(seg["category_id"])
            if orig_cat_id not in thing_id_map:
                continue

            track_id = int(seg["track_id"])
            #contig_cat_id = int(thing_id_map[orig_cat_id])
            seg_id = int(seg["id"])

            mask = (pan_ids == seg_id)
            if not np.any(mask):
                continue

            if track_id not in instances:
                instances[track_id] = {
                    "video_id": str(video_rec["video_id"]),
                    "id": int(track_id),
                    "category_id": orig_cat_id, #contig_cat_id,
                    "segmentations": [None] * t,
                }

            instances[track_id]["segmentations"][frame_idx] = encode_binary_mask(mask)

    return list(instances.values())

def ensure_metadata(dataset_name: str, thing_classes: List[str]):
    meta = MetadataCatalog.get(dataset_name)
    meta.thing_classes = thing_classes

    if not hasattr(meta, "thing_colors"):
        colors = []
        for i in range(len(thing_classes)):
            colors.append([
                (37 * i) % 255,
                (17 * i + 73) % 255,
                (29 * i + 151) % 255,
            ])
        meta.thing_colors = colors

def build_cfg(args, num_classes: int):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)

    # Always register TEMBA config keys so config.yaml files that contain
    # MODEL.TEMBA can still be merged even for baseline inference.
    import mask2former_video.temba_config  # noqa: F401
    from mask2former_video.temba_config import add_temba_adapter_config
    add_temba_adapter_config(cfg)

    if args.model_type == "temba":
        import mask2former_video.modeling.mask_former_head_temba  # noqa: F401
        import mask2former_video.video_maskformer_temba  # noqa: F401

    cfg.merge_from_file(str(Path(args.config_file).expanduser().resolve()))

    cfg.MODEL.WEIGHTS = str(Path(args.ckpt).expanduser().resolve())
    cfg.MODEL.DEVICE = args.device

    if args.model_type == "temba":
        cfg.MODEL.META_ARCHITECTURE = "VideoMaskFormerTembaE2E"
        cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHeadTemba"
        cfg.MODEL.TEMBA.ADAPTER_DIM = args.adapter_dim
        cfg.MODEL.TEMBA.ADAPTER_DEPTH = args.adapter_depth
        cfg.MODEL.TEMBA.LOCAL_KERNEL_SIZE = args.local_kernel_size
        cfg.MODEL.TEMBA.LOCAL_DILATIONS = args.local_dilations
        cfg.MODEL.TEMBA.DTS_DILATIONS = args.dts_dilations
        cfg.MODEL.TEMBA.DROPOUT = args.dropout
    else:
        cfg.MODEL.META_ARCHITECTURE = "VideoMaskFormer"
        cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.DATASETS.TEST = ("jrdb_panovideo_val",)

    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.0
    if hasattr(cfg.MODEL, "MASK_FORMER") and hasattr(cfg.MODEL.MASK_FORMER, "TEST"):
        cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
        cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 1.0

    cfg.freeze()
    return cfg

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument(
        "--val-json",
        default=str(JRDB_ROOT / "annotations" / "val_video_panoptic.json"),
    )
    ap.add_argument(
        "--config-file",
        default=str(MASK2FORMER_DIR / "configs/jrdb/video_maskformer2_R50_jrdb_baseline.yaml"),
    )
    ap.add_argument("--confidence-threshold", type=float, default=0.05)
    ap.add_argument("--max-videos", type=int, default=0, help="0 means all val videos")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--model-type", default="baseline", choices=["baseline", "temba"])

    # TEMBA args
    ap.add_argument("--adapter-dim", type=int, default=256)
    ap.add_argument("--adapter-depth", type=int, default=1)
    ap.add_argument("--local-kernel-size", type=int, default=3)
    ap.add_argument("--local-dilations", nargs="+", type=int, default=[1, 1, 2])
    ap.add_argument("--dts-dilations", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--dropout", type=float, default=0.0)

    return ap.parse_args()


def _resolve_frame_image_path(frame: dict) -> Path:
    """
    Resolve the image path stored in a preprocessed JRDB frame dictionary.

    The preprocessing script usually stores absolute paths in `file_name`.
    For portability, this function also tries paths relative to JRDB_ROOT.
    """
    candidates = []

    for key in ("file_name", "image_file_name", "image_rel_path"):
        value = frame.get(key)
        if not value:
            continue

        raw = Path(value)

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(raw)

            if "JRDB_ROOT" in globals():
                root = Path(JRDB_ROOT)
                candidates.append(root / raw)
                candidates.append(root.parent / raw)

    for cand in candidates:
        if cand.exists():
            return cand

    tried = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not resolve image path for frame. Tried:\n" + tried
    )


def load_frames_for_video(video: dict) -> list[dict]:
    """
    Load all frames of one video into the Detectron2/Mask2Former input format.

    Returns a list of dictionaries containing:
      - image: torch.Tensor with shape [C, H, W]
      - height, width
      - file_name
      - video_id
      - frame_id
      - segments_info, if available
    """
    if "frames" not in video:
        raise KeyError("Expected video dictionary to contain a 'frames' field.")

    frames = sorted(
        video["frames"],
        key=lambda f: int(f.get("frame_id", f.get("image_id", 0))),
    )

    loaded = []
    for idx, frame in enumerate(frames):
        image_path = _resolve_frame_image_path(frame)
        image = read_image(str(image_path), format="BGR")
        h, w = image.shape[:2]

        loaded.append(
            {
                "image": torch.as_tensor(
                    image.astype("float32").transpose(2, 0, 1)
                ),
                "height": int(frame.get("height", h)),
                "width": int(frame.get("width", w)),
                "file_name": str(image_path),
                "video_id": str(frame.get("video_id", video.get("video_id", ""))),
                "frame_id": int(frame.get("frame_id", idx)),
                "segments_info": frame.get("segments_info", []),
            }
        )

    return loaded


def main():
    args = parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.val_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    all_categories = val_data["categories"]
    thing_cats = [c for c in all_categories if int(c.get("isthing", 1)) == 1]
    # Two complementary mappings between original taxonomy ids and the
    # contiguous indices used by the model's classifier head:
    #   thing_id_map:        original_id  -> contiguous_idx   (defines NUM_CLASSES at training)
    #   thing_idx_to_orig_id: contiguous_idx -> original_id   (used to remap model outputs)
    
    
    thing_id_map = {int(c["id"]): i for i, c in enumerate(thing_cats)}
    thing_idx_to_orig_id = {i: int(c["id"]) for i, c in enumerate(thing_cats)}
    #thing_orig_ids = set(thing_id_map.keys())
    thing_classes = [c["name"] for c in thing_cats]
    #eval_categories = [{"id": i, "name": name} for i, name in enumerate(thing_classes)]
    
    # IMPORTANT: eval_categories must use ORIGINAL ids, matching everything
    # we will write to disk (predictions JSON) and load from disk (GT JSON).
    eval_categories = [{"id": int(c["id"]), "name": c["name"]} for c in thing_cats]

    num_classes = len(thing_classes)
    print(f"Using JRDB thing NUM_CLASSES = {num_classes}")

    cfg = build_cfg(args, num_classes=num_classes)
    ensure_metadata("jrdb_panovideo_val", thing_classes)
    demo = VisualizationDemo(cfg)

    videos = val_data["videos"]
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    all_gt_annotations: List[dict[str, Any]] = []
    all_pred_annotations: List[dict[str, Any]] = []

    for video_idx, video in enumerate(videos, start=1):
        frames_meta = sorted(video["frames"], key=lambda x: int(x["frame_id"]))
        num_frames = len(frames_meta)
        if num_frames == 0:
            continue

        height = int(frames_meta[0]["height"])
        width = int(frames_meta[0]["width"])

        print(f"[{video_idx}/{len(videos)}] video_id={video['video_id']}  frames={num_frames}")

        frames = load_frames_for_video(video)
        predictions = unwrap_predictions(demo.run_on_video(frames))     ## this is where predictions are made

        gt_tracks = build_gt_panoptic_tracks_for_video(video, thing_id_map)
        pred_tracks = predictions_to_panoptic_tracks(
            predictions,
            video_id=str(video["video_id"]),
            num_frames=num_frames,
            height=height,
            width=width,
            thing_idx_to_orig_id=thing_idx_to_orig_id,
        )

        all_gt_annotations.extend(gt_tracks)
        all_pred_annotations.extend(pred_tracks)

        del frames
        del predictions
        del gt_tracks
        del pred_tracks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_json_path = out_root / "jrdb_panoptic_track_predictions.json"
    with pred_json_path.open("w", encoding="utf-8") as f:
        json.dump(all_pred_annotations, f)

    print("=" * 80)
    print(f"Saved prediction tracks to: {pred_json_path}")
    print("Use eval_jrdb_panoptic_metrics_from_track_json.py to compute PQ/STQ/VPQ.")
    return

if __name__ == "__main__":
    main()
