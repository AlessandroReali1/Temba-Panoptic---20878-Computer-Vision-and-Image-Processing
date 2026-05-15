#!/bin/bash
set -euo pipefail

if [ "$#" -lt 7 ]; then
  echo "Usage:"
  echo "  bash run_stepwise_eval.sh MODEL_TYPE CKPT CONFIG VAL_JSON_DIR OUT_ROOT GT_JSON GT_ROOT [ADAPTER_DEPTH]"
  echo
  echo "Example:"
  echo "  bash run_stepwise_eval.sh temba /path/model.pth /path/config.yaml /path/single_video_jsons /path/out /path/val_video_panoptic.json /path/jrdb_preproc/out 1"
  exit 1
fi

MODEL_TYPE="$1"
CKPT="$2"
CONFIG="$3"
VAL_JSON_DIR="$4"
OUT_ROOT="$5"
GT_JSON="$6"
GT_ROOT="$7"
ADAPTER_DEPTH="${8:-1}"

mkdir -p "$OUT_ROOT"

for VAL_JSON in "$VAL_JSON_DIR"/*.json; do
  VIDEO_ID="$(basename "$VAL_JSON" .json)"
  VIDEO_OUT="$OUT_ROOT/panoptic_eval_${VIDEO_ID}"

  echo "============================================================"
  echo "Running inference for $VIDEO_ID"
  echo "============================================================"

  if [ "$MODEL_TYPE" = "temba" ]; then
    python scripts/eval/run_jrdb_panoptic_inference.py \
      --model-type temba \
      --ckpt "$CKPT" \
      --config-file "$CONFIG" \
      --adapter-dim 256 \
      --adapter-depth "$ADAPTER_DEPTH" \
      --local-kernel-size 3 \
      --local-dilations 1 1 2 \
      --dts-dilations 1 2 3 \
      --dropout 0.0 \
      --val-json "$VAL_JSON" \
      --out-root "$VIDEO_OUT"
  else
    python scripts/eval/run_jrdb_panoptic_inference.py \
      --model-type baseline \
      --ckpt "$CKPT" \
      --config-file "$CONFIG" \
      --val-json "$VAL_JSON" \
      --out-root "$VIDEO_OUT"
  fi
done

MERGED_DIR="$OUT_ROOT/panoptic_eval_input_merged"
mkdir -p "$MERGED_DIR"

python scripts/eval/merge_prediction_jsons.py \
  --inputs "$OUT_ROOT"/panoptic_eval_*/jrdb_panoptic_track_predictions.json \
  --output "$MERGED_DIR/jrdb_panoptic_track_predictions.json"

python scripts/eval/check_prediction_jsons.py \
  "$MERGED_DIR/jrdb_panoptic_track_predictions.json"

python scripts/eval/eval_jrdb_panoptic_metrics_from_track_json.py \
  --gt-json "$GT_JSON" \
  --gt-root "$GT_ROOT" \
  --pred-json "$MERGED_DIR/jrdb_panoptic_track_predictions.json" \
  --save-json "$MERGED_DIR/jrdb_panoptic_metrics.json"

cat "$MERGED_DIR/jrdb_panoptic_metrics.json"
