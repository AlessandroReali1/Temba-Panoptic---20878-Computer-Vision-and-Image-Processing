
from yacs.config import CfgNode as CN

def add_temba_adapter_config(cfg):
    if hasattr(cfg.MODEL, "TEMBA"):
        return

    cfg.MODEL.TEMBA = CN()
    cfg.MODEL.TEMBA.ADAPTER_DIM = 256
    cfg.MODEL.TEMBA.ADAPTER_DEPTH = 1
    cfg.MODEL.TEMBA.LOCAL_KERNEL_SIZE = 3
    cfg.MODEL.TEMBA.LOCAL_DILATIONS = [1, 1, 2]
    cfg.MODEL.TEMBA.DTS_DILATIONS = [1, 2, 3]
    cfg.MODEL.TEMBA.DROPOUT = 0.0
