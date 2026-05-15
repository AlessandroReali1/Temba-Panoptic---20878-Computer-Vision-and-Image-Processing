# Third-party dependencies

This repository contains only the project-specific code for TEMBA-Panoptic. It does not vendor large external repositories such as Detectron2 or Mask2Former.

## Required external repositories

The project depends on:

    Detectron2
    Mask2Former

During development, the expected paths were similar to:

    /path/to/project/detectron2
    /path/to/project/Mask2Former

and the PYTHONPATH was set as:

    export PYTHONPATH=/path/to/project/detectron2:/path/to/project/Mask2Former:${PYTHONPATH:-}

## Why these are external

Detectron2 and Mask2Former are large research codebases with their own installation requirements. This repository stores only the files modified or added for the project:

    TEMBA adapter code
    TEMBA config additions
    TEMBA Mask2Former model wrappers
    JRDB dataset registration
    training scripts
    evaluation scripts
    visualization scripts

## Installation notes

Install Detectron2 and Mask2Former following their official instructions and ensure they are compatible with your PyTorch and CUDA versions.

The original experiments used CUDA 12.1 and a Python virtual environment named:

    <YOUR_ENV_NAME>

Example activation:

    module purge
    module load miniconda3
    module load cuda/12.1
    eval "$(/software/miniconda3/bin/conda shell.bash hook)"
    source $HOME/venvs/<YOUR_ENV_NAME>/bin/activate

After installation, make sure the following imports work:

    python -c "import detectron2"
    python -c "import mask2former"
    python -c "import mask2former_video"

## Notes for reproducing the project

Before training or evaluation, set:

    export JRDB_PREPROC_ROOT=/path/to/jrdb_preproc/out
    export PYTHONPATH=/path/to/detectron2:/path/to/Mask2Former:${PYTHONPATH:-}

Then use the scripts under:

    scripts/train/
    scripts/eval/
