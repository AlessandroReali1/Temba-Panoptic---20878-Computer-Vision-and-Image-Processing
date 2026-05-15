# Balanced classification loss

The balanced-loss experiments use the standard Mask2Former objective:

L = 2 * loss_ce + 5 * loss_mask + 5 * loss_dice

No TEMBA-specific loss is introduced. TEMBA is supervised only through the final Mask2Former segmentation losses.

The balanced version modifies the classification cross-entropy term `loss_ce` by using JRDB class-frequency weights. The mask loss and Dice loss are unchanged.

The implementation is stored in:

criterion_balanced.py
