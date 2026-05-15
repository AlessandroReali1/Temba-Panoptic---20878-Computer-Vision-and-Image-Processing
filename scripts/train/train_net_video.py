#!/usr/bin/env python3
"""
Repository-local training entry point.

This project relies on the official Mask2Former video training script.
To keep the repository lightweight and avoid vendoring the full external
codebase, this wrapper executes train_net_video.py from MASK2FORMER_ROOT.

Set before running:

    export MASK2FORMER_ROOT=/path/to/Mask2Former
    export DETECTRON2_ROOT=/path/to/detectron2
    export PYTHONPATH=$DETECTRON2_ROOT:$MASK2FORMER_ROOT:${PYTHONPATH:-}
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    mask2former_root = Path(os.environ.get("MASK2FORMER_ROOT", "/path/to/Mask2Former"))
    train_script = mask2former_root / "train_net_video.py"

    if not train_script.exists():
        raise FileNotFoundError(
            "Could not find Mask2Former train_net_video.py. "
            f"Expected: {train_script}. "
            "Set MASK2FORMER_ROOT=/path/to/Mask2Former."
        )

    sys.path.insert(0, str(mask2former_root))
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
