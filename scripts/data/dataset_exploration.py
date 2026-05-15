#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

def summarize_split(split_name, path):
    print("\n" + "=" * 100)
    print(f"SPLIT: {split_name}")
    print("FILE:", path)
    print("=" * 100)

    data = json.loads(Path(path).expanduser().read_text())

    categories = {int(c["id"]): c for c in data["categories"]}
    thing_cats = {
        cid: c for cid, c in categories.items()
        if int(c.get("isthing", 1)) == 1
    }

    videos = data["videos"]
    frame_counts = [len(v["frames"]) for v in videos]

    print("num_videos:", len(videos))
    print("num_frames:", sum(frame_counts))
    print("avg_frames_per_video:", float(np.mean(frame_counts)) if frame_counts else 0)
    print("num_categories_total:", len(categories))
    print("num_thing_categories:", len(thing_cats))

    class_seg_count = Counter()
    class_pixel_area = Counter()
    class_track_ids = defaultdict(set)
    track_presence = defaultdict(set)
    resolutions = Counter()

    total_segments = 0
    total_thing_segments = 0
    areas = []

    for v in videos:
        vid = str(v["video_id"])
        frames = sorted(v["frames"], key=lambda x: int(x["frame_id"]))

        for frame_idx, fr in enumerate(frames):
            h = int(fr["height"])
            w = int(fr["width"])
            resolutions[(h, w)] += 1

            for seg in fr.get("segments_info", []):
                total_segments += 1

                cid = int(seg["category_id"])
                area = int(seg.get("area", 0))

                class_seg_count[cid] += 1
                class_pixel_area[cid] += area
                areas.append(area)

                if int(seg.get("isthing", 1)) == 1 and seg.get("track_id", None) is not None:
                    total_thing_segments += 1
                    tid = int(seg["track_id"])
                    key = (vid, tid)
                    class_track_ids[cid].add(key)
                    track_presence[key].add(frame_idx)

    track_lengths = [len(frames) for frames in track_presence.values()]

    print("\nResolution counts:")
    for (h, w), n in resolutions.most_common():
        print(f"  {h}x{w}: {n} frames")

    print("\nSegments:")
    print("total_segments:", total_segments)
    print("total_thing_segments:", total_thing_segments)
    print("avg_segments_per_frame:", total_segments / max(sum(frame_counts), 1))
    print("avg_thing_segments_per_frame:", total_thing_segments / max(sum(frame_counts), 1))

    print("\nTracks:")
    print("num_unique_tracks:", len(track_presence))
    if track_lengths:
        print("avg_track_length_frames:", float(np.mean(track_lengths)))
        print("median_track_length_frames:", float(np.median(track_lengths)))
        print("min_track_length_frames:", int(np.min(track_lengths)))
        print("max_track_length_frames:", int(np.max(track_lengths)))

    print("\nObject size:")
    if areas:
        arr = np.asarray(areas)
        print("avg_segment_area_pixels:", float(np.mean(arr)))
        print("median_segment_area_pixels:", float(np.median(arr)))
        print("p10_area:", float(np.percentile(arr, 10)))
        print("p90_area:", float(np.percentile(arr, 90)))

    print("\nTop 15 classes by segment count:")
    for cid, n in class_seg_count.most_common(15):
        name = categories.get(cid, {}).get("name", str(cid))
        print(
            f"  {cid:3d} {name:25s} "
            f"segments={n:8d} "
            f"pixels={class_pixel_area[cid]:12d} "
            f"tracks={len(class_track_ids[cid]):6d}"
        )

def main():
    parser = argparse.ArgumentParser(description="Explore JRDB video panoptic JSON files.")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    args = parser.parse_args()

    summarize_split("train", args.train_json)
    summarize_split("val", args.val_json)

if __name__ == "__main__":
    main()
