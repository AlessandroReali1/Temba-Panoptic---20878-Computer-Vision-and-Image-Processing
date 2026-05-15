#!/bin/bash
set -euo pipefail

if [ -z "${MASK2FORMER_ROOT:-}" ]; then
  echo "ERROR: MASK2FORMER_ROOT is not set."
  echo "Example:"
  echo "  export MASK2FORMER_ROOT=/path/to/Mask2Former"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Repository root: $REPO_ROOT"
echo "Mask2Former root: $MASK2FORMER_ROOT"

mkdir -p "$MASK2FORMER_ROOT/mask2former_video/modeling"
mkdir -p "$MASK2FORMER_ROOT/mask2former_video/data_video/datasets"

echo "Copying TEMBA config..."
cp "$REPO_ROOT/src/temba_panoptic/config/temba_config.py" \
   "$MASK2FORMER_ROOT/mask2former_video/temba_config.py"

echo "Copying JRDB dataset registration..."
cp "$REPO_ROOT/src/temba_panoptic/datasets/register_jrdb_vis_from_panoptic.py" \
   "$MASK2FORMER_ROOT/mask2former_video/data_video/datasets/register_jrdb_vis_from_panoptic.py"

echo "Copying TEMBA modeling files..."
cp "$REPO_ROOT/src/temba_panoptic/modeling/temba_adapter.py" \
   "$MASK2FORMER_ROOT/mask2former_video/modeling/temba_adapter.py"

cp "$REPO_ROOT/src/temba_panoptic/modeling/mask_former_head_temba.py" \
   "$MASK2FORMER_ROOT/mask2former_video/modeling/mask_former_head_temba.py"

cp "$REPO_ROOT/src/temba_panoptic/modeling/video_maskformer_temba.py" \
   "$MASK2FORMER_ROOT/mask2former_video/video_maskformer_temba.py"

cp "$REPO_ROOT/src/temba_panoptic/modeling/video_maskformer_temba_e2e.py" \
   "$MASK2FORMER_ROOT/mask2former_video/video_maskformer_temba_e2e.py"

if [ -f "$REPO_ROOT/src/temba_panoptic/losses/criterion_balanced.py" ]; then
  echo "Copying balanced criterion helper..."
  cp "$REPO_ROOT/src/temba_panoptic/losses/criterion_balanced.py" \
     "$MASK2FORMER_ROOT/mask2former_video/modeling/criterion_balanced.py"
fi

echo "Patch applied."
echo
echo "Now verify with:"
echo "  python scripts/setup/check_temba_patch.py"

# Copy JRDB experiment configs into the external Mask2Former tree.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -z "${MASK2FORMER_ROOT:-}" ]; then
  echo "ERROR: Set MASK2FORMER_ROOT=/path/to/Mask2Former before applying the patch."
  exit 1
fi

mkdir -p "$MASK2FORMER_ROOT/configs/jrdb"
cp -r "$REPO_ROOT/configs/jrdb/"* "$MASK2FORMER_ROOT/configs/jrdb/"
echo "Copied JRDB configs to $MASK2FORMER_ROOT/configs/jrdb/"
