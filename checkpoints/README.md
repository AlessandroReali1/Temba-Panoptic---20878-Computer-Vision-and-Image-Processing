# Checkpoints

Large model checkpoints are not stored in this repository.

## Starting checkpoint

All experiments were initialized from the Mask2Former R50 video checkpoint pretrained on YouTube-VIS 2021:

    mask2former_ytvis2021_r50.pkl

Place this file locally and update the path in the training scripts or config overrides.

Example path used during development:

    /path/to/project/starting_checkpoint/mask2former_ytvis2021_r50.pkl

## Trained checkpoints

The main trained checkpoints evaluated in the project are:

    baseline fine-tuned checkpoint
    more transformer decoder layers checkpoint
    TEMBA end-to-end balanced, K=1
    TEMBA end-to-end balanced, K=2
    TEMBA end-to-end balanced, K=3

Example checkpoint names produced by Detectron2:

    model_0005999.pth
    model_0011999.pth
    model_0017999.pth
    model_0023999.pth
    model_0029999.pth

These files are excluded from Git because they are large.

## Reproducing results

To reproduce the reported experiments:

1. download or place the starting Mask2Former checkpoint locally;
2. update MODEL.WEIGHTS in the Slurm scripts or config overrides;
3. train the desired model;
4. run inference one video at a time;
5. merge prediction JSONs;
6. compute PQ, STQ, and VPQ.

Do not commit checkpoint files to the repository unless using Git LFS.
