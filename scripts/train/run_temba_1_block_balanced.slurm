#!/bin/bash
#SBATCH --job-name=<PART>
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<PART>
#SBATCH --gres=<GPU>
#SBATCH --mem=<MEM>
#SBATCH --time=<TIME>
#SBATCH --output=/path/to/project/end_to_end_1_block_balanced/logs/%j.out
#SBATCH --error=/path/to/project/end_to_end_1_block_balanced/logs/%j.err

set -euo pipefail

module purge
module load miniconda3
module load cuda/12.1
eval "$(/software/miniconda3/bin/conda shell.bash hook)"
source ~/venvs/<YOUR_ENV_NAME>/bin/activate

export PYTHONPATH="/path/to/detectron2:/path/to/Mask2Former:${PYTHONPATH:-}"
export JRDB_PREPROC_ROOT="/path/to/jrdb_preproc/out"
export JRDB_CLASS_WEIGHTS_JSON="configs/class_balance/jrdb_track_class_weights_effective_num.json"
export PYTHONWARNINGS="ignore::FutureWarning"
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train/launch_train_baseline.py \
  --num-gpus 1 \
  --config-file /path/to/Mask2Former/configs/jrdb/video_maskformer2_R50_jrdb_baseline.yaml \
  MODEL.WEIGHTS /path/to/checkpoints/mask2former_ytvis2021_r50.pkl \
  OUTPUT_DIR /path/to/project/end_to_end_1_block_balanced/output \
  MODEL.META_ARCHITECTURE VideoMaskFormerTembaE2E \
  MODEL.SEM_SEG_HEAD.NAME MaskFormerHeadTemba \
  MODEL.TEMBA.ADAPTER_DIM 256 \
  MODEL.TEMBA.ADAPTER_DEPTH 1 \
  MODEL.TEMBA.LOCAL_KERNEL_SIZE 3 \
  MODEL.TEMBA.LOCAL_DILATIONS "[1,1,2]" \
  MODEL.TEMBA.DTS_DILATIONS "[1,2,3]" \
  MODEL.TEMBA.DROPOUT 0.0 \
  SOLVER.MAX_ITER 30000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  TEST.EVAL_PERIOD 0 \
  SOLVER.IMS_PER_BATCH 1 \
  DATALOADER.NUM_WORKERS 2
