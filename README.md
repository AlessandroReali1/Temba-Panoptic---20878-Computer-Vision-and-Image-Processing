# Temba-Panoptic

Video Panoptic Segmentation with Mask2Former and lightweight TEMBA-inspired temporal adapters.

## Overview

This project studies Video Panoptic Segmentation on JRDB-PanoTrack, a crowded panoramic robotics dataset. The goal is to improve temporal consistency by inserting TEMBA-inspired blocks into the Mask2Former video segmentation pipeline.

The main idea is to keep the original Mask2Former spatial segmentation components and add temporal adapters between the pixel decoder and the transformer decoder. These adapters process multi-scale pixel-decoder features across consecutive frames before class and mask prediction.

We compare:

- Mask2Former baseline fine-tuning
- Mask2Former with additional transformer decoder layers
- TEMBA-Mask2Former with K=1, K=2, and K=3 TEMBA-inspired blocks per adapter

The best overall configuration is the TEMBA model with K=1 block per adapter.

## Repository structure

    configs/                 Model and experiment configuration files
    src/temba_panoptic/      TEMBA model code, config additions, and JRDB dataset registration
    scripts/data/            Dataset preprocessing, exploration, and plotting utilities
    scripts/train/           Training launchers and Slurm scripts
    scripts/eval/            Inference, JSON validation, merging, and panoptic metric evaluation
    scripts/visualization/   Qualitative visualization scripts
    results/                 Final tables, plots, and qualitative examples
    docs/                    Report, poster, and architecture diagrams
    checkpoints/             Instructions for external model checkpoints
    data/                    Instructions for external dataset files
    third_party/             Instructions for external dependencies

## TEMBA implementation note

The adapters used in this repository are simplified TEMBA-inspired modules. They do not implement the full MS-TEMBA/Mamba state-space block and do not include a selective state-space scan. Instead, temporal context is modeled through local and dilated depthwise temporal convolutions applied to multi-scale pixel-decoder features. The full state-space version is left as future work.

## Method summary

Mask2Former processes each frame using a ResNet backbone, pixel decoder, and transformer decoder. In this project, we add three scale-specific TEMBA adapters after the pixel decoder. Each adapter receives one multi-scale feature level and applies K stacked TEMBA-inspired blocks along the temporal dimension.

Each TEMBA-inspired block contains:

- local temporal convolution
- dilated temporal mixing
- feed-forward refinement
- LayerNorm
- residual connections

TEMBA-inspired blocks are initialized from scratch. The rest of the network is initialized from a Mask2Former R50 video checkpoint pretrained on YouTube-VIS 2021. The final models are trained end-to-end.

## Dataset

The raw JRDB-PanoTrack annotations can be converted into the expected project format with:

    python scripts/data/prepare_jrdb_panoptic.py \
      --img-root /path/to/JRDB \
      --ann-root /path/to/JRDB/annotations \
      --out-root /path/to/jrdb_preproc/out \
      --val-ratio 0.1 \
      --seed 42


The project uses JRDB-PanoTrack converted into a Detectron2/Mask2Former-compatible video panoptic format.

Expected preprocessed files:

    data/jrdb_preproc/out/annotations/train_video_panoptic.json
    data/jrdb_preproc/out/annotations/val_video_panoptic.json
    data/jrdb_preproc/out/panoptic_maps/

The dataset itself is not included in this repository.

See:

    data/README.md

## Checkpoints

Large model checkpoints are not included in this repository.

The starting checkpoint used in the experiments is:

    mask2former_ytvis2021_r50.pkl

See:

    checkpoints/README.md

## Setup

This repository depends on Detectron2 and Mask2Former. They should be installed or cloned separately.

Example environment activation used in the experiments:

    module purge
    module load miniconda3
    module load cuda/12.1
    eval "$(/software/miniconda3/bin/conda shell.bash hook)"
    source $HOME/venvs/<YOUR_ENV_NAME>/bin/activate

Example environment variables:

    export JRDB_PREPROC_ROOT=/path/to/project/jrdb_preproc/out
    export PYTHONPATH=/path/to/project/detectron2:/path/to/project/Mask2Former:${PYTHONPATH:-}

See:

    third_party/README.md

Important: the repository stores the project-specific TEMBA code under src/temba_panoptic, but training uses the external Mask2Former training entry point. Therefore, before training TEMBA models, the files must be copied into the Mask2Former package using:

    bash scripts/setup/apply_temba_patch_to_mask2former.sh

This reproduces the patched Mask2Former tree used in the experiments.

## Training

Training scripts are located in:

    scripts/train/

Main scripts:

    scripts/train/run_baseline_finetune.slurm
    scripts/train/run_more_transformer_blocks.slurm
    scripts/train/run_temba_1_block_balanced.slurm
    scripts/train/run_temba_2_blocks_balanced.slurm
    scripts/train/run_temba_3_blocks_balanced.slurm

TEMBA depth is controlled by:

    MODEL.TEMBA.ADAPTER_DEPTH

The main experiments use:

    K = 1, 2, 3

## Evaluation

Inference is run one video at a time to avoid memory issues. Per-video prediction JSONs are then merged and evaluated using PQ, STQ, and VPQ.

Useful scripts:

    scripts/eval/run_jrdb_panoptic_inference.py
    scripts/eval/merge_prediction_jsons.py
    scripts/eval/check_prediction_jsons.py
    scripts/eval/eval_jrdb_panoptic_metrics_from_track_json.py

Main metrics:

    PQ, PQ_SQ, PQ_RQ
    STQ, STQ_SQ, STQ_AQ
    VPQ, VPQ@1, VPQ@2, VPQ@4, VPQ@8
    number of predicted tracks

## Main results

| Method | PQ | PQ_SQ | PQ_RQ | STQ | STQ_SQ | STQ_AQ | VPQ | Tracks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3.57 | 14.89 | 4.82 | 17.74 | 5.99 | 52.54 | 3.46 | 85 |
| +3 decoder layers | 3.00 | 14.92 | 4.05 | 17.81 | 6.10 | 52.79 | 3.33 | 96 |
| TEMBA K=1 | 4.58 | 19.94 | 6.40 | 19.93 | 7.63 | 52.08 | 3.90 | 87 |
| TEMBA K=2 | 3.46 | 21.47 | 4.58 | 17.15 | 5.62 | 52.36 | 3.21 | 94 |
| TEMBA K=3 | 3.69 | 19.85 | 5.06 | 19.31 | 7.09 | 52.60 | 3.18 | 90 |

The K=1 TEMBA model gives the best overall performance. The K=2 model obtains the highest PQ_SQ, suggesting good mask overlap for matched predictions, but lower recognition quality reduces the overall PQ.

## Visualization

Qualitative outputs can be generated with:

    scripts/visualization/visualize_jrdb_predictions.py
    scripts/visualization/visualize_jrdb_predictions_hq.py

Representative results are stored in:

    results/qualitative/

Large videos and full frame dumps should generally be kept outside Git unless Git LFS is used.

## Notes

Large files are intentionally excluded:

- raw dataset files
- preprocessed panoptic maps
- model checkpoints
- Slurm logs
- full prediction JSONs
- full visualization videos
- virtual environments


## Future work: full state-space TEMBA

The current implementation uses a simplified TEMBA-inspired convolutional adapter. It does not include a selective state-space scan and does not use a state dimension hyperparameter. A natural future extension is to replace the convolutional temporal mixer with a full TEMBA/MS-TEMBA state-space module. In that setting, a state dimension parameter, often denoted `d_state`, would become a real architectural hyperparameter controlling the size of the recurrent temporal state used by the selective scan.

## Evaluation note

The repository includes a custom evaluator for PQ, STQ, and VPQ computed from the generated panoptic track predictions. The reported values are intended primarily for relative comparison between the model variants evaluated under the same pipeline.

## Training entry point note

Training is launched through `scripts/train/train_net_video.py`, a lightweight wrapper around the official Mask2Former video training entry point. The full Mask2Former and Detectron2 repositories remain external dependencies and must be installed separately. This keeps the repository focused on the project-specific TEMBA modifications while making the training command reproducible from the local scripts.

## Mask2Former CUDA operators

Do not install `MultiScaleDeformableAttention` from PyPI. It is a CUDA extension built from the Mask2Former source tree. After installing Mask2Former, compile the custom operators following the official Mask2Former instructions, for example:

    cd $MASK2FORMER_ROOT/mask2former/modeling/pixel_decoder/ops
    sh make.sh

The exact command may depend on the Mask2Former version, CUDA version, and cluster setup.

## Config placement note

The configs under `configs/jrdb/` are meant to be used inside the external Mask2Former repository after patching. Their `_BASE_` fields may reference Mask2Former configs such as `../youtubevis_2021/...`, which are not stored in this repository. Run the patch script or manually copy `configs/jrdb/` into `$MASK2FORMER_ROOT/configs/jrdb/` before training.
