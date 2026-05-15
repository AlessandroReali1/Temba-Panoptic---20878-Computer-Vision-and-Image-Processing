# Training scripts

This folder contains the launchers and Slurm scripts used to train the project models.

## Required Mask2Former patch

The TEMBA training scripts require a patched Mask2Former tree. Before training, set:

    export DETECTRON2_ROOT=/path/to/detectron2
    export MASK2FORMER_ROOT=/path/to/Mask2Former
    export JRDB_PREPROC_ROOT=/path/to/jrdb_preproc/out

Then apply the project patch:

    bash scripts/setup/apply_temba_patch_to_mask2former.sh

Verify the installation:

    python scripts/setup/check_temba_patch.py

This copies the TEMBA config, model definitions, segmentation head, and JRDB dataset registration into the external Mask2Former package. This is necessary because the training scripts use Mask2Former's train_net_video.py entry point.

## Models

- run_baseline_finetune.slurm: fine-tunes the Mask2Former baseline.
- run_more_transformer_blocks.slurm: trains the baseline with extra transformer decoder layers.
- run_temba_1_block_balanced.slurm: trains TEMBA end-to-end with K=1 block per adapter.
- run_temba_2_blocks_balanced.slurm: trains TEMBA end-to-end with K=2 blocks per adapter.
- run_temba_3_blocks_balanced.slurm: trains TEMBA end-to-end with K=3 blocks per adapter.

Here, K is the number of TEMBA blocks inside each scale-specific adapter. Since the model uses three adapters, K=1, K=2, and K=3 correspond to 3, 6, and 9 total TEMBA blocks.

## Important variables to edit

Before running on a new cluster, update:

- Slurm account
- Slurm partition
- GPU type / GPU request
- virtual environment path
- dataset path
- starting checkpoint path
- output directory
- paths to Detectron2 and Mask2Former

The original experiments used paths similar to:

    export JRDB_PREPROC_ROOT=/path/to/project/jrdb_preproc/out
    export PYTHONPATH=/path/to/project/detectron2:/path/to/project/Mask2Former:${PYTHONPATH:-}

The original virtual environment was activated with:

    module purge
    module load miniconda3
    module load cuda/12.1
    eval "$(/software/miniconda3/bin/conda shell.bash hook)"
    source $HOME/venvs/<YOUR_ENV_NAME>/bin/activate

## Starting checkpoint

All models were initialized from the Mask2Former R50 video checkpoint pretrained on YouTube-VIS 2021:

    mask2former_ytvis2021_r50.pkl

The checkpoint is not included in this repository because it is large. Place it locally and update the MODEL.WEIGHTS path in the Slurm scripts.

## Dataset

The scripts expect JRDB-PanoTrack to be preprocessed into the video panoptic format used by Mask2Former/Detectron2.

Expected files:

    $JRDB_PREPROC_ROOT/annotations/train_video_panoptic.json
    $JRDB_PREPROC_ROOT/annotations/val_video_panoptic.json
    $JRDB_PREPROC_ROOT/panoptic_maps/

The registered dataset names are:

    jrdb_panovideo_train
    jrdb_panovideo_val

Dataset registration is handled by:

    src/temba_panoptic/datasets/register_jrdb_vis_from_panoptic.py

## TEMBA configuration

The TEMBA depth is controlled by:

    MODEL.TEMBA.ADAPTER_DEPTH

The main experiments use:

    MODEL.TEMBA.ADAPTER_DIM 256
    MODEL.TEMBA.ADAPTER_DEPTH 1 / 2 / 3
    MODEL.TEMBA.LOCAL_KERNEL_SIZE 3
    MODEL.TEMBA.LOCAL_DILATIONS [1, 1, 2]
    MODEL.TEMBA.DTS_DILATIONS [1, 2, 3]
    MODEL.TEMBA.DROPOUT 0.0

## Training examples

Baseline fine-tuning:

    sbatch scripts/train/run_baseline_finetune.slurm

More transformer decoder layers:

    sbatch scripts/train/run_more_transformer_blocks.slurm

TEMBA end-to-end, balanced loss, K=1:

    sbatch scripts/train/run_temba_1_block_balanced.slurm

TEMBA end-to-end, balanced loss, K=2:

    sbatch scripts/train/run_temba_2_blocks_balanced.slurm

TEMBA end-to-end, balanced loss, K=3:

    sbatch scripts/train/run_temba_3_blocks_balanced.slurm

## Notes

The Slurm scripts are cluster-specific. They are included to document the exact experiments, but paths and resource requests may need to be changed before running on another machine.

Large training outputs are not committed to the repository. This includes:

    output/
    logs/
    *.pth
    *.pkl
    events.out.tfevents*
    last_checkpoint

Use the repository .gitignore to avoid committing checkpoints, logs, and full training outputs.

