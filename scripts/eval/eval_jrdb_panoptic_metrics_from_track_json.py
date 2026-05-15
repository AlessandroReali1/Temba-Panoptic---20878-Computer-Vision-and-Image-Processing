#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

def rgb2id(color: np.ndarray) -> np.ndarray:
    color = np.asarray(color, dtype=np.int64)
    if color.ndim == 2:
        return color
    return color[..., 0] + 256 * color[..., 1] + 256 * 256 * color[..., 2]

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)

def categories_from_json(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(c["id"]): c for c in data["categories"]}

def is_thing(cat_map: Dict[int, Dict[str, Any]], category_id: int) -> bool:
    return int(cat_map[category_id].get("isthing", 1)) == 1

def decode_rle(rle: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if rle is None:
        return None
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)

def pair_hist(gt_seg, pred_seg):
    """
    Fast histogram of (gt_id, pred_id) pixel pairs.

    This replaces np.unique(..., axis=0), which is very slow on large
    full-resolution panoramic masks. We encode each pair as a single int64,
    run np.unique on the 1D encoded array, then decode the pairs.
    """
    gt = np.asarray(gt_seg, dtype=np.int64).reshape(-1)
    pred = np.asarray(pred_seg, dtype=np.int64).reshape(-1)

    # Remove pure background/void pairs. Intersections involving non-zero
    # GT or prediction IDs are preserved.
    keep = (gt != 0) | (pred != 0)
    gt = gt[keep]
    pred = pred[keep]

    if gt.size == 0:
        return {}

    # Assumes segment IDs fit in 32 bits, which is true for JRDB panoptic IDs.
    encoded = (gt << np.int64(32)) | pred

    uniq, cnt = np.unique(encoded, return_counts=True)

    out = {}
    for code, c in zip(uniq, cnt):
        g = int(code >> np.int64(32))
        pr = int(code & np.int64(0xFFFFFFFF))
        out[(g, pr)] = int(c)

    return out

def build_video_frame_index(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}

    anns = data.get("annotations", [])
    if anns:
        for video_ann in anns:
            vid = str(video_ann.get("video_id", video_ann.get("id", video_ann.get("name"))))
            frames = video_ann.get("annotations", video_ann.get("frames", video_ann.get("images", [])))
            out[vid] = frames
        return out

    vids = data.get("videos", [])
    if vids:
        for video_ann in vids:
            vid = str(
                video_ann.get(
                    "video_id",
                    video_ann.get("id", video_ann.get("name", video_ann.get("sequence_name")))
                )
            )
            frames = video_ann.get("annotations", video_ann.get("frames", video_ann.get("images", [])))
            out[vid] = frames
        return out

    return out

def resolve_png(root: Path, frame_rec: Dict[str, Any]) -> Path:
    fname = frame_rec.get("panoptic_file_name", frame_rec.get("file_name"))
    if fname is None:
        raise KeyError("Frame record must contain 'panoptic_file_name' or 'file_name'.")

    p = Path(fname)
    if p.is_absolute() and p.exists():
        return p

    p = root / fname
    if p.exists():
        return p

    p2 = root / Path(fname).name
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Cannot resolve GT panoptic PNG for '{fname}' under '{root}'.")

def group_predictions_by_video(preds: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = defaultdict(list)
    for p in preds:
        out[str(p["video_id"])].append(p)
    return out

def build_pred_frames_from_tracks(
    video_id: str,
    tracks: List[Dict[str, Any]],
    num_frames: int,
    height: int,
    width: int,
) -> List[Tuple[np.ndarray, List[Dict[str, Any]]]]:
    tracks = sorted(tracks, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    frames: List[Tuple[np.ndarray, List[Dict[str, Any]]]] = []

    for t in range(num_frames):
        pan_map = np.zeros((height, width), dtype=np.int32)
        segments_info: List[Dict[str, Any]] = []

        for tr in tracks:
            segs = tr.get("segmentations", [])
            if t >= len(segs):
                continue

            mask = decode_rle(segs[t])
            if mask is None:
                continue

            visible = np.logical_and(mask, pan_map == 0)
            if visible.sum() == 0:
                continue

            seg_id = int(tr.get("id", 0))
            cat_id = int(tr["category_id"])

            pan_map[visible] = seg_id
            segments_info.append(
                {
                    "id": seg_id,
                    "category_id": cat_id,
                    "track_id": seg_id,
                    "score": float(tr.get("score", 0.0)),
                    "area": int(visible.sum()),
                }
            )

        frames.append((pan_map, segments_info))

    return frames

def load_gt_frames(root: Path, frame_recs: List[Dict[str, Any]]) -> List[Tuple[np.ndarray, List[Dict[str, Any]]]]:
    out = []
    for fr in frame_recs:
        png_path = resolve_png(root, fr)
        seg = rgb2id(np.array(Image.open(png_path), dtype=np.uint8))
        info = fr.get("segments_info", [])
        out.append((seg, info))
    return out

@dataclass(frozen=True)
class TubeKey:
    video_id: str
    category_id: int
    track_id: int

def get_track_id(seg: Dict[str, Any], thing_flag: bool) -> int:
    if thing_flag:
        for k in ("track_id", "id"):
            if k in seg:
                return int(seg[k])
        raise KeyError(f"Thing segment has no usable track id: {seg}")
    return int(seg["category_id"])

def build_seginfo_maps(
    seg_map: np.ndarray,
    segments_info: List[Dict[str, Any]],
) -> Tuple[Dict[int, int], Dict[int, Dict[str, Any]]]:
    id_to_cat = {}
    id_to_info = {}
    for s in segments_info:
        sid = int(s["id"])
        id_to_cat[sid] = int(s["category_id"])
        id_to_info[sid] = s
    return id_to_cat, id_to_info

def make_category_map(seg_map: np.ndarray, id_to_cat: Dict[int, int], ignore_label: int = -1) -> np.ndarray:
    cat_map = np.full(seg_map.shape, ignore_label, dtype=np.int32)
    for sid, cid in id_to_cat.items():
        cat_map[seg_map == sid] = cid
    return cat_map

def pq_from_counts(stat: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    pqs, sqs, rqs = [], [], []
    for _, d in stat.items():
        tp = d["tp"]
        fp = d["fp"]
        fn = d["fn"]
        iou = d["iou"]
        denom = tp + 0.5 * fp + 0.5 * fn
        if denom <= 0:
            continue
        pq = iou / denom
        sq = (iou / tp) if tp > 0 else 0.0
        rq = tp / denom if denom > 0 else 0.0
        pqs.append(pq)
        sqs.append(sq)
        rqs.append(rq)

    if not pqs:
        return {"PQ": 0.0, "PQ_SQ": 0.0, "PQ_RQ": 0.0}

    return {
        "PQ": 100.0 * float(np.mean(pqs)),
        "PQ_SQ": 100.0 * float(np.mean(sqs)),
        "PQ_RQ": 100.0 * float(np.mean(rqs)),
    }

def update_pq_for_frame(
    stat: Dict[int, Dict[str, float]],
    gt_seg: np.ndarray,
    gt_info: List[Dict[str, Any]],
    pred_seg: np.ndarray,
    pred_info: List[Dict[str, Any]],
) -> None:
    gt_id_to_cat, _ = build_seginfo_maps(gt_seg, gt_info)
    pred_id_to_cat, _ = build_seginfo_maps(pred_seg, pred_info)

    gt_area = {int(s["id"]): int((gt_seg == int(s["id"])).sum()) for s in gt_info}
    pred_area = {int(s["id"]): int((pred_seg == int(s["id"])).sum()) for s in pred_info}

    inter = pair_hist(gt_seg, pred_seg)
    gt_matched = set()
    pred_matched = set()
    candidates = []

    for (gid, pid), inter_ij in inter.items():
        if gid == 0 or pid == 0:
            continue
        if gid not in gt_id_to_cat or pid not in pred_id_to_cat:
            continue
        gc = gt_id_to_cat[gid]
        pc = pred_id_to_cat[pid]
        if gc != pc:
            continue
        union = gt_area[gid] + pred_area[pid] - inter_ij
        if union <= 0:
            continue
        iou = inter_ij / union
        if iou > 0.5:
            candidates.append((iou, gid, pid, gc))

    candidates.sort(reverse=True, key=lambda x: x[0])

    for iou, gid, pid, cid in candidates:
        if gid in gt_matched or pid in pred_matched:
            continue
        gt_matched.add(gid)
        pred_matched.add(pid)
        stat[cid]["tp"] += 1.0
        stat[cid]["iou"] += float(iou)

    for s in gt_info:
        gid = int(s["id"])
        cid = int(s["category_id"])
        if gid not in gt_matched:
            stat[cid]["fn"] += 1.0

    for s in pred_info:
        pid = int(s["id"])
        cid = int(s["category_id"])
        if pid not in pred_matched:
            stat[cid]["fp"] += 1.0

class STQAccumulator:
    def __init__(self, cat_map: Dict[int, Dict[str, Any]]):
        self.cat_map = cat_map
        self.tp = defaultdict(float)
        self.fp = defaultdict(float)
        self.fn = defaultdict(float)
        self.gt_tube_area = defaultdict(float)
        self.pred_tube_area = defaultdict(float)
        self.intersections = defaultdict(float)

    def update_frame(
        self,
        video_id: str,
        gt_seg: np.ndarray,
        gt_info: List[Dict[str, Any]],
        pred_seg: np.ndarray,
        pred_info: List[Dict[str, Any]],
    ) -> None:
        gt_id_to_cat, gt_id_to_info = build_seginfo_maps(gt_seg, gt_info)
        pred_id_to_cat, pred_id_to_info = build_seginfo_maps(pred_seg, pred_info)

        gt_cat = make_category_map(gt_seg, gt_id_to_cat, ignore_label=-1)
        pred_cat = make_category_map(pred_seg, pred_id_to_cat, ignore_label=-1)

        cats = set(np.unique(gt_cat).tolist()) | set(np.unique(pred_cat).tolist())
        cats.discard(-1)

        for cid in cats:
            gt_mask = gt_cat == cid
            pred_mask = pred_cat == cid
            inter = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union == 0:
                continue
            self.tp[cid] += float(inter)
            self.fp[cid] += float(pred_mask.sum() - inter)
            self.fn[cid] += float(gt_mask.sum() - inter)

        for s in gt_info:
            sid = int(s["id"])
            cid = int(s["category_id"])
            gt_t = TubeKey(video_id, cid, get_track_id(s, is_thing(self.cat_map, cid)))
            self.gt_tube_area[gt_t] += float((gt_seg == sid).sum())

        for s in pred_info:
            sid = int(s["id"])
            cid = int(s["category_id"])
            pred_t = TubeKey(video_id, cid, get_track_id(s, is_thing(self.cat_map, cid)))
            self.pred_tube_area[pred_t] += float((pred_seg == sid).sum())

        inter_pairs = pair_hist(gt_seg, pred_seg)
        for (gid, pid), a in inter_pairs.items():
            if gid == 0 or pid == 0:
                continue
            if gid not in gt_id_to_info or pid not in pred_id_to_info:
                continue
            gs = gt_id_to_info[gid]
            ps = pred_id_to_info[pid]
            gc = int(gs["category_id"])
            pc = int(ps["category_id"])
            if gc != pc:
                continue
            gt_t = TubeKey(video_id, gc, get_track_id(gs, is_thing(self.cat_map, gc)))
            pred_t = TubeKey(video_id, pc, get_track_id(ps, is_thing(self.cat_map, pc)))
            self.intersections[(gt_t, pred_t)] += float(a)

    def compute(self) -> Dict[str, float]:
        ious = []
        cats = set(self.tp.keys()) | set(self.fp.keys()) | set(self.fn.keys())
        for cid in cats:
            denom = self.tp[cid] + self.fp[cid] + self.fn[cid]
            if denom > 0:
                ious.append(self.tp[cid] / denom)
        stq_sq = float(np.mean(ious)) if ious else 0.0

        total_gt_area = sum(self.gt_tube_area.values())
        if total_gt_area <= 0:
            stq_aq = 0.0
        else:
            pred_by_vc = defaultdict(list)
            for pt in self.pred_tube_area:
                pred_by_vc[(pt.video_id, pt.category_id)].append(pt)

            weighted_sum = 0.0
            for gt_t, gt_a in self.gt_tube_area.items():
                best = 0.0
                for pred_t in pred_by_vc[(gt_t.video_id, gt_t.category_id)]:
                    inter = self.intersections.get((gt_t, pred_t), 0.0)
                    if inter <= 0:
                        continue
                    pred_a = self.pred_tube_area[pred_t]
                    f = 2.0 * inter / (gt_a + pred_a)
                    if f > best:
                        best = f
                weighted_sum += gt_a * best
            stq_aq = weighted_sum / total_gt_area

        stq = math.sqrt(max(stq_sq, 0.0) * max(stq_aq, 0.0))
        return {
            "STQ": 100.0 * stq,
            "STQ_SQ": 100.0 * stq_sq,
            "STQ_AQ": 100.0 * stq_aq,
        }

def tube_pq_from_window_stats(
    gt_tube_area: Dict[TubeKey, float],
    pred_tube_area: Dict[TubeKey, float],
    intersections: Dict[Tuple[TubeKey, TubeKey], float],
) -> Dict[int, Dict[str, float]]:
    per_cat = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "iou": 0.0})

    candidates = []
    for (gt_t, pred_t), inter in intersections.items():
        if gt_t.category_id != pred_t.category_id:
            continue
        union = gt_tube_area[gt_t] + pred_tube_area[pred_t] - inter
        if union <= 0:
            continue
        iou = inter / union
        if iou > 0.5:
            candidates.append((iou, gt_t, pred_t, gt_t.category_id))

    candidates.sort(reverse=True, key=lambda x: x[0])

    gt_used = set()
    pred_used = set()

    for iou, gt_t, pred_t, cid in candidates:
        if gt_t in gt_used or pred_t in pred_used:
            continue
        gt_used.add(gt_t)
        pred_used.add(pred_t)
        per_cat[cid]["tp"] += 1.0
        per_cat[cid]["iou"] += float(iou)

    for gt_t in gt_tube_area:
        if gt_t not in gt_used:
            per_cat[gt_t.category_id]["fn"] += 1.0

    for pred_t in pred_tube_area:
        if pred_t not in pred_used:
            per_cat[pred_t.category_id]["fp"] += 1.0

    return per_cat

def vpq_metric(
    gt_frames: List[Tuple[np.ndarray, List[Dict[str, Any]]]],
    pred_frames: List[Tuple[np.ndarray, List[Dict[str, Any]]]],
    video_id: str,
    cat_map: Dict[int, Dict[str, Any]],
    k: int,
) -> float:
    acc = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "iou": 0.0})

    if len(gt_frames) < k:
        return 0.0

    for start in range(0, len(gt_frames) - k + 1):
        gt_tube_area = defaultdict(float)
        pred_tube_area = defaultdict(float)
        intersections = defaultdict(float)

        for off in range(k):
            gt_seg, gt_info = gt_frames[start + off]
            pred_seg, pred_info = pred_frames[start + off]

            gt_id_to_cat, gt_id_to_info = build_seginfo_maps(gt_seg, gt_info)
            pred_id_to_cat, pred_id_to_info = build_seginfo_maps(pred_seg, pred_info)

            for s in gt_info:
                sid = int(s["id"])
                cid = int(s["category_id"])
                gt_t = TubeKey(video_id, cid, get_track_id(s, is_thing(cat_map, cid)))
                gt_tube_area[gt_t] += float((gt_seg == sid).sum())

            for s in pred_info:
                sid = int(s["id"])
                cid = int(s["category_id"])
                pred_t = TubeKey(video_id, cid, get_track_id(s, is_thing(cat_map, cid)))
                pred_tube_area[pred_t] += float((pred_seg == sid).sum())

            inter_pairs = pair_hist(gt_seg, pred_seg)
            for (gid, pid), a in inter_pairs.items():
                if gid == 0 or pid == 0:
                    continue
                if gid not in gt_id_to_info or pid not in pred_id_to_info:
                    continue
                gs = gt_id_to_info[gid]
                ps = pred_id_to_info[pid]
                gc = int(gs["category_id"])
                pc = int(ps["category_id"])
                if gc != pc:
                    continue
                gt_t = TubeKey(video_id, gc, get_track_id(gs, is_thing(cat_map, gc)))
                pred_t = TubeKey(video_id, pc, get_track_id(ps, is_thing(cat_map, pc)))
                intersections[(gt_t, pred_t)] += float(a)

        per_cat = tube_pq_from_window_stats(gt_tube_area, pred_tube_area, intersections)
        for cid, d in per_cat.items():
            for key in ("tp", "fp", "fn", "iou"):
                acc[cid][key] += d[key]

    return pq_from_counts(acc)["PQ"]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--pred-json", type=Path, required=True)
    parser.add_argument("--save-json", type=Path, required=True)
    args = parser.parse_args()

    gt_data = load_json(args.gt_json)
    pred_tracks = load_json(args.pred_json)

    cat_map = categories_from_json(gt_data)
    gt_index = build_video_frame_index(gt_data)
    pred_by_video = group_predictions_by_video(pred_tracks)

    pq_stat = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "iou": 0.0})
    stq_acc = STQAccumulator(cat_map)
    vpq_vals = {1: [], 2: [], 4: [], 8: []}

    common_videos = sorted(set(gt_index.keys()) & set(pred_by_video.keys()))
    if not common_videos:
        raise RuntimeError("No common video ids between GT and prediction JSON.")

    for vid in common_videos:
        gt_frames = load_gt_frames(args.gt_root, gt_index[vid])

        h, w = gt_frames[0][0].shape
        pred_frames = build_pred_frames_from_tracks(
            video_id=vid,
            tracks=pred_by_video[vid],
            num_frames=len(gt_frames),
            height=h,
            width=w,
        )

        for (gt_seg, gt_info), (pred_seg, pred_info) in zip(gt_frames, pred_frames):
            update_pq_for_frame(pq_stat, gt_seg, gt_info, pred_seg, pred_info)
            stq_acc.update_frame(vid, gt_seg, gt_info, pred_seg, pred_info)

        for k in vpq_vals.keys():
            vpq_vals[k].append(vpq_metric(gt_frames, pred_frames, vid, cat_map, k))

    pq_out = pq_from_counts(pq_stat)
    stq_out = stq_acc.compute()
    vpq_out = {f"VPQ@{k}": float(np.mean(v)) if v else 0.0 for k, v in vpq_vals.items()}
    vpq_out["VPQ"] = float(np.mean(list(vpq_out.values()))) if vpq_out else 0.0

    result = {
        **pq_out,
        **stq_out,
        **vpq_out,
    }

    print("=" * 80)
    for key in ["PQ", "PQ_SQ", "PQ_RQ", "STQ", "STQ_SQ", "STQ_AQ", "VPQ", "VPQ@1", "VPQ@2", "VPQ@4", "VPQ@8"]:
        print(f"{key}: {result[key]:.6f}")

    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    with args.save_json.open("w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
