import os
from pathlib import Path
import importlib
import runpy
import sys

home = Path.home()

detectron2_root = Path(os.environ.get("DETECTRON2_ROOT", "/path/to/detectron2"))
sys.path.insert(0, str(detectron2_root))
mask2former_root = Path(os.environ.get("MASK2FORMER_ROOT", "/path/to/Mask2Former"))
sys.path.insert(0, str(mask2former_root))

# Register JRDB datasets
importlib.import_module("mask2former_video.data_video.datasets.register_jrdb_vis_from_panoptic")

script = Path(__file__).resolve().parent / "train_net_video.py"
sys.argv = [str(script)] + sys.argv[1:]
runpy.run_path(str(script), run_name="__main__")
