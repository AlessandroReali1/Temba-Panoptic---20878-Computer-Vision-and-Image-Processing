# Class balancing configuration

This folder stores files related to the class-balanced loss used in the balanced TEMBA experiments.

JRDB-PanoTrack has a long-tailed class distribution: frequent classes such as pedestrian appear much more often than rare object classes. To reduce the effect of this imbalance, the balanced experiments reweight the classification term of the Mask2Former loss.

The standard Mask2Former loss is:

    L = 2 * L_CE + 5 * L_MASK + 5 * L_DICE

In the balanced experiments, L_CE is replaced or modified with class-frequency weights. The mask and Dice losses remain unchanged.

Expected files in this folder may include:

    jrdb_class_weights.json
    jrdb_class_weights_effective_num.json

These files are small and can be committed if they do not contain dataset-private information.

If the class-weight file is missing, regenerate it from the training annotation JSON using the dataset exploration or class-frequency scripts under:

    scripts/data/

Important note:

There is no TEMBA-specific loss. TEMBA is trained through the same segmentation losses as the rest of Mask2Former.
