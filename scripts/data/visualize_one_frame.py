import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

if len(sys.argv) != 4:
    print("Usage: python visualize_one_frame.py /path/to/sequence.json /path/to/images_root /path/to/output.png")
    sys.exit(1)

ann_file = Path(sys.argv[1])
img_root = Path(sys.argv[2])
out_file = Path(sys.argv[3])

def resolve_img_path(img_root: Path, rel_path: str) -> Path:
    rel = Path(rel_path)

    # Case 1: direct join works
    p1 = img_root / rel
    if p1.exists():
        return p1

    parts = rel.parts

    # Case 2:
    # image_stitched/SEQ/images/SEQ/000000.jpg
    # -> image_stitched/SEQ/000000.jpg
    if len(parts) >= 5 and parts[0] == "image_stitched" and parts[2] == "images":
        p2 = img_root / parts[0] / parts[1] / parts[-1]
        if p2.exists():
            return p2

    # Case 3:
    # if there is a duplicated sequence name
    # image_stitched/SEQ/SEQ/000000.jpg -> image_stitched/SEQ/000000.jpg
    if len(parts) >= 4 and parts[0] == "image_stitched" and parts[1] == parts[2]:
        p3 = img_root / parts[0] / parts[1] / parts[-1]
        if p3.exists():
            return p3

    raise FileNotFoundError(
        f"Could not resolve image path.\n"
        f"img_root={img_root}\n"
        f"rel_path={rel_path}"
    )

data = json.loads(ann_file.read_text())

img_info = data["images"][0]
img_path = resolve_img_path(img_root, img_info["file_name"])
print("Using image:", img_path)

image = np.array(Image.open(img_path).convert("RGB"))
anns = [a for a in data["annotations"] if a["image_id"] == img_info["id"]]

overlay = image.copy()
rng = np.random.default_rng(0)

for ann in anns:
    rle = ann["segmentation"]
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    color = rng.integers(0, 255, size=3, dtype=np.uint8)
    overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * color).astype(np.uint8)

plt.figure(figsize=(20, 4))
plt.imshow(overlay)
plt.axis("off")
plt.tight_layout()
plt.savefig(out_file, dpi=150)
print("saved to", out_file)
