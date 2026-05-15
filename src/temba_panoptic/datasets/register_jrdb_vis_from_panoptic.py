import json
import os
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

def rgb2id(color):
    color = np.asarray(color, dtype=np.int32)
    if color.ndim == 2:
        return color
    return color[:, :, 0] + 256 * color[:, :, 1] + 65536 * color[:, :, 2]

def load_jrdb_vis_from_panoptic(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data["categories"]
    thing_cats = [c for c in categories if int(c["isthing"]) == 1]
    thing_id_map = {int(c["id"]): i for i, c in enumerate(thing_cats)}

    dataset_dicts = []

    for video in data["videos"]:
        frames = sorted(video["frames"], key=lambda x: int(x["frame_id"]))
        if not frames:
            continue

        record = {
            "video_id": video["video_id"],
            "file_names": [fr["file_name"] for fr in frames],
            "height": int(frames[0]["height"]),
            "width": int(frames[0]["width"]),
            "length": len(frames),
            "annotations": [],
        }

        for fr in frames:
            pan_png = np.asarray(Image.open(fr["panoptic_file_name"]).convert("RGB"), dtype=np.uint8)
            pan_ids = rgb2id(pan_png)

            frame_annos = []
            for seg in fr["segments_info"]:
                if int(seg["isthing"]) != 1:
                    continue
                if int(seg.get("iscrowd", 0)) != 0:
                    continue
                if seg.get("track_id", None) is None:
                    continue

                seg_id = int(seg["id"])
                cat_id = int(seg["category_id"])
                if cat_id not in thing_id_map:
                    continue

                mask = (pan_ids == seg_id).astype(np.uint8)
                if mask.sum() == 0:
                    continue

                rle = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("ascii")

                frame_annos.append(
                    {
                        "id": int(seg["track_id"]),
                        "category_id": int(thing_id_map[cat_id]),
                        "iscrowd": 0,
                        "bbox": [float(x) for x in seg["bbox"]],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": rle,
                    }
                )

            record["annotations"].append(frame_annos)

        dataset_dicts.append(record)

    return dataset_dicts

def register_jrdb_vis_dataset(name, json_file, thing_classes, thing_dataset_id_to_contiguous_id):
    DatasetCatalog.register(name, lambda jf=json_file: load_jrdb_vis_from_panoptic(jf))
    MetadataCatalog.get(name).set(
        json_file=str(json_file),
        evaluator_type="ytvis",
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
    )

def register_all_jrdb_vis_from_panoptic():
    preproc_root = os.environ.get("JRDB_PREPROC_ROOT", "")
    if not preproc_root:
        warnings.warn("JRDB-PanoTrack dataset was not registered because a required path or file is missing.")
        return

    preproc_root = Path(preproc_root)
    train_json = preproc_root / "annotations" / "train_video_panoptic.json"
    val_json = preproc_root / "annotations" / "val_video_panoptic.json"
    cats_json = preproc_root / "metadata" / "categories.json"

    missing = [p for p in [train_json, val_json, cats_json] if not p.exists()]
    if missing:
        warnings.warn(
            "JRDB-PanoTrack dataset was not registered because required files are missing: "
            + ", ".join(str(p) for p in missing),
            RuntimeWarning,
        )
        return

    with open(cats_json, "r", encoding="utf-8") as f:
        categories = json.load(f)

    thing_cats = [c for c in categories if int(c["isthing"]) == 1]
    thing_classes = [c["name"] for c in thing_cats]
    thing_dataset_id_to_contiguous_id = {int(c["id"]): i for i, c in enumerate(thing_cats)}

    register_jrdb_vis_dataset(
        "jrdb_panovideo_train",
        train_json,
        thing_classes,
        thing_dataset_id_to_contiguous_id,
    )
    register_jrdb_vis_dataset(
        "jrdb_panovideo_val",
        val_json,
        thing_classes,
        thing_dataset_id_to_contiguous_id,
    )

register_all_jrdb_vis_from_panoptic()
