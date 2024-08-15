from yacs.config import CfgNode as CN

import os

_C = CN()
_C.GPUS = (0, )
_C.WORKERS = 16
_C.PIN_MEMORY = True
_C.AUTO_RESUME = True
_C.PRINT_FREQ = 10


_C.DATASET = CN()
_C.DATASET.ROOT = "./Dataloader"
_C.DATASET.CHANNEL = 3
_C.DATASET.DATASET = 'CelebA'

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1

_C.MODEL = CN()
_C.MODEL.NAME = "UGL"
_C.MODEL.IMG_SIZE = 256
_C.MODEL.OUT_DIM = 64
_C.MODEL.NUM_Part = 5
_C.MODEL.BACKGROUND = 1

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

_C.LOSS = CN()

_C.TRAIN = CN()
_C.TRAIN.TRAIN = True
_C.TRAIN.SHUFFLE = True
_C.TRAIN.LR = 0.0005
_C.TRAIN.LR_FACTOR = 0.2
_C.TRAIN.LR_STEP = [90, 95]
_C.TRAIN.OPTIMIZER = "adamw"
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.PRE = "./pretrain/dino_deitsmall8_pretrain.pth"
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.NUM_EPOCH = 100

_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 32

_C.CELE = CN()
_C.CELE.ROOT = './Dataset/CelebA/'
_C.CELE.FRACTION = 1.0
_C.CELE.SCALE = 0.05
_C.CELE.ROTATION = 15
_C.CELE.TRANSLATION = 0.05
_C.CELE.FLIP = True

_C.CUB = CN()
_C.CUB.ROOT = './Dataset/CUB/'
_C.CUB.FRACTION = 1.0
_C.CUB.SCALE = 0.05
_C.CUB.ROTATION = 15
_C.CUB.TRANSLATION = 0.05
_C.CUB.FLIP = True

_C.PartImage = CN()
_C.PartImage.ROOT = './Dataset/PartImageNet_Processed'
_C.PartImage.FRACTION = 1.0
_C.PartImage.SCALE = 0.05
_C.PartImage.ROTATION = 15
_C.PartImage.TRANSLATION = 0.05
_C.PartImage.FLIP = True

def update_config(cfg, args):
    cfg.defrost()

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    if args.target:
        cfg.TARGET = args.target

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.freeze()
