#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

def color_from_id(x):
    x = int(x)
    return (
        int((37 * x + 17) % 255),
        int((97 * x + 53) % 255),
        int((173 * x + 29) % 255),
    )

def decode_rle(rle):
    if rle is None:
        return None
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)

def resize_keep_width(img, target_width):
    if target_width <= 0:
        return img
    h, w = img.shape[:2]
    if w == target_width:
        return img
    scale = target_width / float(w)
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)

def draw_one_frame(img, tracks, frame_idx, cat_id_to_name, alpha=0.45, score_thresh=0.0, draw_labels=True):
    overlay = img.copy()

    # Draw low-confidence tracks last? No, draw high-confidence first for stable visualization.
    tracks = sorted(tracks, key=lambda x: float(x.get("score", 0.0)), reverse=True)

    for tr in tracks:
        score = float(tr.get("score", 0.0))
        if score < score_thresh:
            continue

        segs = tr.get("segmentations", [])
        if frame_idx >= len(segs):
            continue

        mask = decode_rle(segs[frame_idx])
        if mask is None or not np.any(mask):
            continue

        track_id = int(tr.get("id", 0))
        cat_id = int(tr.get("category_id", -1))
        name = cat_id_to_name.get(cat_id, str(cat_id))

        color = color_from_id(track_id + 1009 * cat_id)

        overlay[mask] = (
            (1.0 - alpha) * overlay[mask] + alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        if draw_labels:
            ys, xs = np.where(mask)
            if len(xs) > 0:
                x0 = int(np.median(xs))
                y0 = int(np.median(ys))
                label = f"{name} #{track_id} {score:.2f}"
                cv2.putText(
                    overlay,
                    label,
                    (max(0, x0 - 20), max(20, y0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    label,
                    (max(0, x0 - 20), max(20, y0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    return overlay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--pred-json", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--resize-width", type=int, default=1880, help="0 keeps original width")
    ap.add_argument("--no-labels", action="store_true")
    args = ap.parse_args()

    val_json = Path(args.val_json).expanduser()
    pred_json = Path(args.pred_json).expanduser()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    val_data = json.loads(val_json.read_text())
    pred_data = json.loads(pred_json.read_text())

    cat_id_to_name = {int(c["id"]): c.get("name", str(c["id"])) for c in val_data["categories"]}

    pred_by_video = {}
    for tr in pred_data:
        pred_by_video.setdefault(str(tr["video_id"]), []).append(tr)

    videos = val_data["videos"]

    print("prediction videos:", sorted(pred_by_video.keys()))

    for video in videos:
        vid = str(video["video_id"])
        if vid not in pred_by_video:
            continue

        frames = sorted(video["frames"], key=lambda x: int(x["frame_id"]))
        tracks = pred_by_video[vid]

        video_out_dir = out_root / vid
        frames_out_dir = video_out_dir / "frames"
        frames_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nvideo={vid}")
        print(f"frames={len(frames)} tracks={len(tracks)}")

        first_img = cv2.imread(str(frames[0]["file_name"]))
        if first_img is None:
            raise RuntimeError(f"Could not read first frame: {frames[0]['file_name']}")

        first_vis = resize_keep_width(first_img, args.resize_width)
        h, w = first_vis.shape[:2]

        mp4_path = video_out_dir / f"{vid}_predictions.mp4"
        writer = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (w, h),
        )

        for frame_idx, fr in enumerate(frames):
            img = cv2.imread(str(fr["file_name"]))
            if img is None:
                raise RuntimeError(f"Could not read image: {fr['file_name']}")

            vis = draw_one_frame(
                img,
                tracks,
                frame_idx,
                cat_id_to_name,
                alpha=args.alpha,
                score_thresh=args.score_thresh,
                draw_labels=not args.no_labels,
            )

            vis = resize_keep_width(vis, args.resize_width)

            frame_path = frames_out_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), vis)
            writer.write(vis)

        writer.release()
        print("saved mp4:", mp4_path)
        print("saved frames:", frames_out_dir)

if __name__ == "__main__":
    main()
