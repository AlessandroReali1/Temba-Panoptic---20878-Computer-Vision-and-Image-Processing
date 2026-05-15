import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def id2rgb(id_map: np.ndarray) -> np.ndarray:
    rgb = np.zeros(id_map.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = id_map % 256
    rgb[..., 1] = (id_map // 256) % 256
    rgb[..., 2] = (id_map // 65536) % 256
    return rgb


def resolve_img_path(img_root: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    parts = rel.parts

    candidates = []

    candidates.append(img_root / rel)

    if len(parts) >= 5 and parts[0] == "image_stitched" and parts[2] == "images":
        candidates.append(img_root / parts[0] / parts[1] / parts[-1])

    if len(parts) >= 4 and parts[0] == "image_stitched" and parts[1] == parts[2]:
        candidates.append(img_root / parts[0] / parts[1] / parts[-1])

    if len(parts) >= 2:
        candidates.append(img_root / Path(*parts[1:]))

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not resolve image path:\n"
        f"  img_root={img_root}\n"
        f"  rel_path={rel_path}\n"
        f"  tried={candidates}"
    )


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def build_thing_categories(categories):
    thing_categories = [c for c in categories if int(c["isthing"]) == 1]
    thing_categories = sorted(thing_categories, key=lambda c: int(c["id"]))
    remapped = []
    thing_id_map = {}
    for new_id, c in enumerate(thing_categories):
        old_id = int(c["id"])
        thing_id_map[old_id] = new_id
        remapped.append(
            {
                "id": new_id,
                "name": c["name"],
                "isthing": 1,
                "original_id": old_id,
            }
        )
    return remapped, thing_id_map


def process_sequence(seq_json: Path, img_root: Path, panoptic_root: Path):
    data = json.loads(seq_json.read_text())
    seq_name = seq_json.stem

    cat_map = {int(c["id"]): c for c in data["categories"]}
    remapped_categories, thing_id_map = build_thing_categories(data["categories"])

    ann_per_image = defaultdict(list)
    for ann in data["annotations"]:
        ann_per_image[ann["image_id"]].append(ann)

    images = sorted(data["images"], key=lambda x: int(Path(x["file_name"]).stem))

    seq_panoptic_dir = panoptic_root / seq_name
    seq_panoptic_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    total_overwritten = 0

    for img in images:
        img_path = resolve_img_path(img_root, img["file_name"])
        h, w = int(img["height"]), int(img["width"])
        frame_num = int(Path(img_path).stem)

        anns = ann_per_image[img["id"]]

        panoptic_ids = np.zeros((h, w), dtype=np.int32)
        seg_meta = {}
        next_seg_id = 1
        overwritten_pixels = 0

        for ann in anns:
            rle = ann["segmentation"]
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.astype(bool)

            if mask.shape != (h, w):
                raise ValueError(
                    f"Mask shape mismatch in {seq_name} frame {frame_num}: "
                    f"{mask.shape} vs {(h, w)}"
                )

            if not mask.any():
                continue

            overwritten_pixels += int(((panoptic_ids > 0) & mask).sum())

            original_category_id = int(ann["category_id"])
            cat = cat_map[original_category_id]
            isthing = int(cat["isthing"])

            track_id = ann.get("tracking_id", ann.get("attributes", {}).get("tracking_id"))
            if isthing == 0:
                continue
            if track_id is None:
                continue

            track_id = int(track_id)
            mapped_category_id = int(thing_id_map[original_category_id])

            panoptic_ids[mask] = next_seg_id
            seg_meta[next_seg_id] = {
                "id": next_seg_id,
                "category_id": mapped_category_id,
                "original_category_id": original_category_id,
                "isthing": 1,
                "track_id": track_id,
                "iscrowd": 0,
                "attributes": ann.get("attributes", {}),
            }
            next_seg_id += 1

        total_overwritten += overwritten_pixels

        segments_info = []
        for seg_id, meta in seg_meta.items():
            ys, xs = np.where(panoptic_ids == seg_id)
            if ys.size == 0:
                continue

            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())

            info = dict(meta)
            info["area"] = int(xs.size)
            info["bbox"] = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
            segments_info.append(info)

        panoptic_rgb = id2rgb(panoptic_ids)
        panoptic_path = seq_panoptic_dir / f"{frame_num:06d}.png"
        Image.fromarray(panoptic_rgb).save(panoptic_path)

        frames.append(
            {
                "video_id": seq_name,
                "frame_id": frame_num,
                "image_id": int(img["id"]),
                "file_name": str(img_path),
                "image_rel_path": img["file_name"],
                "panoptic_file_name": str(panoptic_path),
                "height": h,
                "width": w,
                "segments_info": segments_info,
                "overlap_pixels_overwritten": overwritten_pixels,
            }
        )

    return {
        "video_id": seq_name,
        "num_frames": len(frames),
        "frames": frames,
        "sequence_overlap_pixels_overwritten": total_overwritten,
        "categories": remapped_categories,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root", type=Path, required=True)
    parser.add_argument("--ann-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    panoptic_root = args.out_root / "panoptic_maps"
    ann_out_root = args.out_root / "annotations"
    meta_out_root = args.out_root / "metadata"
    panoptic_root.mkdir(parents=True, exist_ok=True)
    ann_out_root.mkdir(parents=True, exist_ok=True)
    meta_out_root.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(args.ann_root.rglob("*.json"))
    if not ann_files:
        raise RuntimeError(f"No JSON files found under {args.ann_root}")

    available_sequences = {
        p.name for p in (args.img_root / "image_stitched").iterdir() if p.is_dir()
    }

    ann_map = {}
    for p in ann_files:
        if p.stem in available_sequences and p.stem not in ann_map:
            ann_map[p.stem] = p

    ann_files = [ann_map[k] for k in sorted(ann_map.keys())]

    if args.limit > 0:
        ann_files = ann_files[:args.limit]

    if not ann_files:
        raise RuntimeError("No annotation JSONs matched available image_stitched sequences.")

    print(f"Matched sequences: {len(ann_files)}")

    first_data = json.loads(ann_files[0].read_text())
    categories, _ = build_thing_categories(first_data["categories"])

    video_records = []
    missing = []

    for i, seq_json in enumerate(ann_files, start=1):
        seq_name = seq_json.stem
        try:
            print(f"[{i}/{len(ann_files)}] Processing {seq_name}")
            rec = process_sequence(seq_json, args.img_root, panoptic_root)
            video_records.append(rec)
        except Exception as e:
            print(f"[WARN] Skipping {seq_name}: {e}")
            missing.append({"sequence": seq_name, "error": str(e)})

    save_json(categories, meta_out_root / "categories.json")
    save_json(missing, meta_out_root / "missing_or_failed_sequences.json")

    seq_names = sorted([v["video_id"] for v in video_records])
    rng = random.Random(args.seed)
    rng.shuffle(seq_names)

    n_val = int(round(len(seq_names) * args.val_ratio))
    if len(seq_names) > 1 and n_val == 0:
        n_val = 1
    if n_val >= len(seq_names):
        n_val = max(0, len(seq_names) - 1)

    val_set = set(seq_names[:n_val])
    train_set = set(seq_names[n_val:])

    seq_splits = {
        "train": sorted(train_set),
        "val": sorted(val_set),
    }
    save_json(seq_splits, meta_out_root / "seq_splits.json")

    all_json = {
        "categories": categories,
        "videos": video_records,
    }
    train_json = {
        "categories": categories,
        "videos": [v for v in video_records if v["video_id"] in train_set],
    }
    val_json = {
        "categories": categories,
        "videos": [v for v in video_records if v["video_id"] in val_set],
    }

    save_json(all_json, ann_out_root / "all_video_panoptic.json")
    save_json(train_json, ann_out_root / "train_video_panoptic.json")
    save_json(val_json, ann_out_root / "val_video_panoptic.json")

    print("\nDone.")
    print("Processed videos:", len(video_records))
    print("Train videos:", len(train_json["videos"]))
    print("Val videos:", len(val_json["videos"]))
    print("Failed videos:", len(missing))
    print("Outputs:")
    print(" ", ann_out_root / "all_video_panoptic.json")
    print(" ", ann_out_root / "train_video_panoptic.json")
    print(" ", ann_out_root / "val_video_panoptic.json")
    print(" ", meta_out_root / "categories.json")
    print(" ", meta_out_root / "seq_splits.json")


if __name__ == "__main__":
    main()
