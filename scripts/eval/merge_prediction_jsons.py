#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Merge per-video JRDB prediction JSON files into one prediction JSON."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged = []

    for inp in args.inputs:
        path = Path(inp).expanduser()
        data = json.loads(path.read_text())

        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}, got {type(data)}")

        print(f"loaded {len(data)} predictions from {path}")
        merged.extend(data)

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged))

    print("wrote", out)
    print("total_predictions:", len(merged))
    print("video_ids:", sorted({str(x.get("video_id")) for x in merged}))

if __name__ == "__main__":
    main()
