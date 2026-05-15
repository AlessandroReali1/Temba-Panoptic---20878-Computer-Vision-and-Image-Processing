#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

REQUIRED_KEYS = {
    "video_id",
    "id",
    "category_id",
    "score",
    "segmentations",
}

def check_file(path: Path):
    print("\n" + "=" * 90)
    print("FILE:", path)
    print("exists:", path.exists())

    if not path.exists():
        return

    print("size_bytes:", path.stat().st_size)

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print("valid_json: no")
        print("error:", repr(e))
        return

    print("valid_json: yes")

    if not isinstance(data, list):
        print("schema_ok: no, expected top-level list")
        return

    print("num_predictions:", len(data))

    if len(data) == 0:
        print("warning: empty prediction file")
        return

    bad_items = []
    video_ids = set()
    segment_lengths = set()

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            bad_items.append((i, "not a dict"))
            continue

        missing = REQUIRED_KEYS - set(item.keys())
        if missing:
            bad_items.append((i, f"missing keys: {sorted(missing)}"))
            continue

        video_ids.add(str(item["video_id"]))

        segs = item["segmentations"]
        if not isinstance(segs, list):
            bad_items.append((i, "segmentations is not a list"))
        else:
            segment_lengths.add(len(segs))

    print("schema_ok:", "yes" if not bad_items else "no")
    print("video_ids:", sorted(video_ids))
    print("segment_lengths:", sorted(segment_lengths))

    if bad_items:
        print("bad_items_sample:", bad_items[:10])

def main():
    parser = argparse.ArgumentParser(
        description="Check whether JRDB prediction JSON files are valid."
    )
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    for p in args.paths:
        check_file(Path(p).expanduser())

if __name__ == "__main__":
    main()
