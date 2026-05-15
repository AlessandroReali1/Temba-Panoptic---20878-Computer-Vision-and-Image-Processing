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

def resize_image(img, target_width):
    if target_width <= 0:
        return img

    h, w = img.shape[:2]
    if w == target_width:
        return img

    scale = target_width / float(w)
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_CUBIC)

def resize_mask(mask, target_shape):
    out_h, out_w = target_shape
    return cv2.resize(
        mask.astype(np.uint8),
        (out_w, out_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

def draw_label_with_background(
    img,
    text,
    x,
    y,
    font_scale,
    text_thickness,
    bg_alpha=0.75,
):
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    pad = max(6, int(6 * font_scale))

    x1 = max(0, x)
    y1 = max(0, y - th - baseline - 2 * pad)
    x2 = min(img.shape[1] - 1, x + tw + 2 * pad)
    y2 = min(img.shape[0] - 1, y + baseline + pad)

    roi = img[y1:y2, x1:x2].copy()
    rect = np.zeros_like(roi)
    rect[:] = (0, 0, 0)
    blended = cv2.addWeighted(rect, bg_alpha, roi, 1.0 - bg_alpha, 0)
    img[y1:y2, x1:x2] = blended

    cv2.putText(
        img,
        text,
        (x1 + pad, y2 - baseline - pad),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

def draw_one_frame(
    img,
    tracks,
    frame_idx,
    cat_id_to_name,
    alpha,
    score_thresh,
    font_scale,
    text_thickness,
    contour_thickness,
    draw_labels,
):
    overlay = img.copy()
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

        if mask.shape[:2] != img.shape[:2]:
            mask = resize_mask(mask, img.shape[:2])

        track_id = int(tr.get("id", 0))
        cat_id = int(tr.get("category_id", -1))
        name = cat_id_to_name.get(cat_id, str(cat_id))

        color = color_from_id(track_id + 1009 * cat_id)

        overlay[mask] = (
            (1.0 - alpha) * overlay[mask].astype(np.float32)
            + alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, color, contour_thickness)

        if draw_labels:
            ys, xs = np.where(mask)
            if len(xs) > 0:
                x0 = int(np.percentile(xs, 50))
                y0 = int(np.percentile(ys, 50))
                label = f"{name} #{track_id} {score:.2f}"
                draw_label_with_background(
                    overlay,
                    label,
                    max(0, x0 - 40),
                    max(30, y0),
                    font_scale,
                    text_thickness,
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
    ap.add_argument("--resize-width", type=int, default=3760)
    ap.add_argument("--font-scale", type=float, default=1.0)
    ap.add_argument("--text-thickness", type=int, default=2)
    ap.add_argument("--contour-thickness", type=int, default=3)
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--save-png", action="store_true")
    args = ap.parse_args()

    val_json = Path(args.val_json).expanduser()
    pred_json = Path(args.pred_json).expanduser()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    val_data = json.loads(val_json.read_text())
    pred_data = json.loads(pred_json.read_text())

    cat_id_to_name = {
        int(c["id"]): c.get("name", str(c["id"]))
        for c in val_data["categories"]
    }

    pred_by_video = {}
    for tr in pred_data:
        pred_by_video.setdefault(str(tr["video_id"]), []).append(tr)

    print("prediction videos:", sorted(pred_by_video.keys()))

    for video in val_data["videos"]:
        vid = str(video["video_id"])
        if vid not in pred_by_video:
            continue

        frames = sorted(video["frames"], key=lambda x: int(x["frame_id"]))
        tracks = pred_by_video[vid]

        video_out_dir = out_root / vid
        frames_out_dir = video_out_dir / "frames_png"
        jpg_out_dir = video_out_dir / "frames_jpg"

        frames_out_dir.mkdir(parents=True, exist_ok=True)
        jpg_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nvideo={vid}")
        print(f"frames={len(frames)} tracks={len(tracks)}")

        first_img = cv2.imread(str(frames[0]["file_name"]))
        if first_img is None:
            raise RuntimeError(f"Could not read first frame: {frames[0]['file_name']}")

        first_img = resize_image(first_img, args.resize_width)
        h, w = first_img.shape[:2]

        mp4_path = video_out_dir / f"{vid}_predictions_HQ.mp4"
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

            img = resize_image(img, args.resize_width)

            vis = draw_one_frame(
                img=img,
                tracks=tracks,
                frame_idx=frame_idx,
                cat_id_to_name=cat_id_to_name,
                alpha=args.alpha,
                score_thresh=args.score_thresh,
                font_scale=args.font_scale,
                text_thickness=args.text_thickness,
                contour_thickness=args.contour_thickness,
                draw_labels=not args.no_labels,
            )

            writer.write(vis)

            jpg_path = jpg_out_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(jpg_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 98])

            if args.save_png:
                png_path = frames_out_dir / f"{frame_idx:06d}.png"
                cv2.imwrite(str(png_path), vis)

        writer.release()
        print("saved mp4:", mp4_path)
        print("saved jpg frames:", jpg_out_dir)
        if args.save_png:
            print("saved png frames:", frames_out_dir)

if __name__ == "__main__":
    main()
