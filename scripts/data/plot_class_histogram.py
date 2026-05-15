#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt

def pretty_label(name):
    if name == "skateboard/segway/hoverboard":
        return "skateboard/segway/\nhoverboard"
    return name

def main():
    parser = argparse.ArgumentParser(
        description="Plot class appearance frequency histogram for JRDB train+val."
    )
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--log-scale", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": Path(args.train_json).expanduser(),
        "val": Path(args.val_json).expanduser(),
    }

    categories = {}
    counts = Counter()
    split_counts = {"train": Counter(), "val": Counter()}

    for split, path in paths.items():
        data = json.loads(path.read_text())

        for c in data["categories"]:
            categories[int(c["id"])] = c

        for video in data["videos"]:
            for frame in video["frames"]:
                for seg in frame.get("segments_info", []):
                    cid = int(seg["category_id"])
                    counts[cid] += 1
                    split_counts[split][cid] += 1

    rows = []
    for cid, c in sorted(categories.items()):
        rows.append({
            "category_id": cid,
            "category_name": c.get("name", str(cid)),
            "isthing": int(c.get("isthing", 1)),
            "joint_count": counts[cid],
            "train_count": split_counts["train"][cid],
            "val_count": split_counts["val"][cid],
        })

    csv_path = out_dir / "class_appearance_counts_train_val.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("saved CSV:", csv_path)

    nonzero_rows = [r for r in rows if r["joint_count"] > 0]
    nonzero_rows = sorted(nonzero_rows, key=lambda r: r["joint_count"], reverse=True)

    plot_rows = nonzero_rows[:args.top_n]
    names = [pretty_label(r["category_name"]) for r in plot_rows]
    values = [r["joint_count"] for r in plot_rows]

    plt.figure(figsize=(18, 8))
    plt.bar(range(len(names)), values)

    if args.log_scale:
        plt.yscale("log")
        ylab = "Number of segment annotations, log scale"
        suffix = "log"
    else:
        ylab = "Number of segment annotations"
        suffix = "linear"

    plt.xticks(range(len(names)), names, rotation=90, fontsize=14)
    plt.yticks(fontsize=16)
    plt.ylabel(ylab, fontsize=18)
    plt.xlabel("Class", fontsize=18)
    plt.title(f"Class Appearance Frequency in JRDB Train + Validation", fontsize=22)
    plt.tight_layout()

    png_path = out_dir / f"class_appearance_histogram_top{args.top_n}_{suffix}.png"
    pdf_path = out_dir / f"class_appearance_histogram_top{args.top_n}_{suffix}.pdf"

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print("saved PNG:", png_path)
    print("saved PDF:", pdf_path)

if __name__ == "__main__":
    main()
