# Data

This project uses JRDB-PanoTrack converted into a Detectron2/Mask2Former-compatible video panoptic format.

The raw dataset and generated panoptic maps are not included in this repository because they are large and must be obtained from the official dataset source.

## Preprocessing

The preprocessing script is:

    scripts/data/prepare_jrdb_panoptic.py

Example usage:

    python scripts/data/prepare_jrdb_panoptic.py \
      --img-root /path/to/JRDB \
      --ann-root /path/to/JRDB/annotations \
      --out-root /path/to/jrdb_preproc/out \
      --val-ratio 0.1 \
      --seed 42

## Expected preprocessed structure

Set the environment variable:

    export JRDB_PREPROC_ROOT=/path/to/jrdb_preproc/out

Expected files:

    $JRDB_PREPROC_ROOT/annotations/all_video_panoptic.json
    $JRDB_PREPROC_ROOT/annotations/train_video_panoptic.json
    $JRDB_PREPROC_ROOT/annotations/val_video_panoptic.json
    $JRDB_PREPROC_ROOT/metadata/categories.json
    $JRDB_PREPROC_ROOT/metadata/seq_splits.json
    $JRDB_PREPROC_ROOT/panoptic_maps/

The training code expects the following Detectron2 dataset names:

    jrdb_panovideo_train
    jrdb_panovideo_val

Dataset registration is implemented in:

    src/temba_panoptic/datasets/register_jrdb_vis_from_panoptic.py

## Single-video JSONs

For memory reasons, inference can be run one video at a time. Single-video JSON files can be generated with:

    python scripts/data/create_single_video_jsons.py \
      --input-json $JRDB_PREPROC_ROOT/annotations/val_video_panoptic.json \
      --out-dir data/single_video_jsons

These generated JSON files are not required to be committed.

## Dataset exploration

Dataset statistics and class-frequency plots can be generated with:

    python scripts/data/dataset_exploration.py \
      --train-json $JRDB_PREPROC_ROOT/annotations/train_video_panoptic.json \
      --val-json $JRDB_PREPROC_ROOT/annotations/val_video_panoptic.json

    python scripts/data/plot_class_histogram.py \
      --train-json $JRDB_PREPROC_ROOT/annotations/train_video_panoptic.json \
      --val-json $JRDB_PREPROC_ROOT/annotations/val_video_panoptic.json \
      --out-dir results/plots \
      --top-n 30 \
      --log-scale

## Important note

Do not commit raw JRDB frames, generated panoptic maps, or full preprocessed datasets to Git.

## Category scope

JRDB-PanoTrack provides both thing and stuff annotations. The preprocessing used in this project keeps the 61 tracked thing categories and discards stuff classes, because the experiments focus on temporal consistency and object identities across frames. The generated metadata/categories.json therefore contains only entries with isthing = 1.
