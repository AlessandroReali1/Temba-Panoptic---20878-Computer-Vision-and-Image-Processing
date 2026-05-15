#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Split a JRDB video panoptic JSON into one JSON per video."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    input_json = Path(args.input_json).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_json.read_text())

    for video in data["videos"]:
        video_id = str(video["video_id"])
        out_path = out_dir / f"{video_id}.json"

        one_video_data = {
            "categories": data["categories"],
            "videos": [video],
        }

        out_path.write_text(json.dumps(one_video_data))
        print("wrote", out_path)

if __name__ == "__main__":
    main()
