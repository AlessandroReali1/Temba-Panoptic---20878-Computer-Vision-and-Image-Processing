# JRDB Mask2Former configs

The YAML files in this folder are intended to be used inside a patched Mask2Former tree.

Some configs use `_BASE_` paths such as:

    ../youtubevis_2021/...

Those base configs are provided by the external Mask2Former repository, not by this project repository. Therefore, before training, copy or patch the project-specific files into the Mask2Former tree using:

    bash scripts/setup/apply_temba_patch_to_mask2former.sh

or manually copy:

    configs/jrdb/*

to:

    $MASK2FORMER_ROOT/configs/jrdb/

This repository keeps these configs so that the experiment settings are versioned together with the project code.
