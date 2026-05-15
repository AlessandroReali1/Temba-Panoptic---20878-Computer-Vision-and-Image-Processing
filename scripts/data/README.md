# Data processing scripts

This folder contains the scripts used to prepare, inspect, and analyze the JRDB-PanoTrack data used in the project.

## Main preprocessing script

The main preprocessing script is:

    prepare_jrdb_panoptic.py

It converts raw JRDB-PanoTrack sequence annotation JSONs into the Detectron2/Mask2Former-compatible video panoptic format used by the project.

Example usage:

    python scripts/data/prepare_jrdb_panoptic.py \
      --img-root /path/to/JRDB \
      --ann-root /path/to/JRDB/annotations \
      --out-root /path/to/jrdb_preproc/out \
      --val-ratio 0.1 \
      --seed 42

Expected output:

    /path/to/jrdb_preproc/out/annotations/all_video_panoptic.json
    /path/to/jrdb_preproc/out/annotations/train_video_panoptic.json
    /path/to/jrdb_preproc/out/annotations/val_video_panoptic.json
    /path/to/jrdb_preproc/out/metadata/categories.json
    /path/to/jrdb_preproc/out/metadata/seq_splits.json
    /path/to/jrdb_preproc/out/panoptic_maps/

Set the following environment variable before training or evaluation:

    export JRDB_PREPROC_ROOT=/path/to/jrdb_preproc/out

## Utility scripts

- inspect_jrdb_panoptic.py  
  Prints statistics and example entries from a raw JRDB-PanoTrack sequence annotation file.

- visualize_one_frame.py  
  Visualizes the masks of one raw JRDB-PanoTrack frame by overlaying decoded annotations on the image.

- create_single_video_jsons.py  
  Splits a video panoptic JSON into one JSON file per video, useful for memory-safe one-video-at-a-time inference.

- dataset_exploration.py  
  Computes dataset statistics such as number of videos, frames, segments, tracks, class frequencies, and object sizes.

- plot_class_histogram.py  
  Generates class-frequency histograms used for dataset analysis and visualization.

## Dataset registration

The preprocessed dataset is registered through:

    src/temba_panoptic/datasets/register_jrdb_vis_from_panoptic.py

Expected Detectron2 dataset names:

    jrdb_panovideo_train
    jrdb_panovideo_val

## Note

Raw JRDB-PanoTrack data, generated panoptic maps, and full preprocessed datasets are not included in this repository because of size and licensing constraints.

## Category scope

The preprocessing script keeps tracked thing categories only. Stuff categories are discarded, and thing categories are remapped to contiguous IDs. This matches the experimental setup used in the project, which focuses on tracked object masks and temporal consistency.
