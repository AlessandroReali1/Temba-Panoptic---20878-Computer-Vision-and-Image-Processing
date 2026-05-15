import json
import sys
from pathlib import Path
from collections import defaultdict
from pycocotools import mask as mask_utils

if len(sys.argv) != 2:
    print("Usage: python inspect_jrdb_panoptic.py /path/to/sequence.json")
    sys.exit(1)

ann_file = Path(sys.argv[1])
data = json.loads(ann_file.read_text())

print("Annotation file:", ann_file)
print("Top-level keys:", list(data.keys()))
print("num images:", len(data["images"]))
print("num annotations:", len(data["annotations"]))
print("num categories:", len(data["categories"]))

cat_map = {c["id"]: c for c in data["categories"]}
ann_per_image = defaultdict(list)
for ann in data["annotations"]:
    ann_per_image[ann["image_id"]].append(ann)

thing_total = 0
thing_with_track = 0
crowd_total = 0
null_track = 0
overlap_frames_checked = 0
overlap_pixels_total = 0

for img in data["images"][:10]:
    anns = ann_per_image[img["id"]]
    canvas = None

    for ann in anns:
        cat = cat_map[ann["category_id"]]
        isthing = int(cat["isthing"]) == 1
        if isthing:
            thing_total += 1
            if ann.get("tracking_id") is not None:
                thing_with_track += 1
        if ann.get("iscrowd", 0) == 1:
            crowd_total += 1
        if ann.get("tracking_id") is None:
            null_track += 1

        rle = ann["segmentation"]
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if canvas is None:
            canvas = mask.astype("uint16")
        else:
            overlap_pixels_total += int(((canvas > 0) & (mask > 0)).sum())
            canvas = canvas + mask.astype("uint16")

    overlap_frames_checked += 1

print("thing_total:", thing_total)
print("thing_with_track:", thing_with_track)
print("crowd_total:", crowd_total)
print("null_track:", null_track)
print("overlap_frames_checked:", overlap_frames_checked)
print("overlap_pixels_total:", overlap_pixels_total)

print("\nExample image entry:")
print(data["images"][0])

print("\nExample annotation entry:")
print(data["annotations"][0])

print("\nExample category entry:")
print(data["categories"][0])
