from yacs.config import CfgNode as CN

# hardware
_C = CN()
_C.PROJECT_NAME = "nlos_backbone_jointsChecked_1227"
_C.DEVICE = (2)  # ATTENTION: nlos_unet load in 'meas2vol.py' only support ONE GPU
_C.NUM_WORKERS = 16
_C.WANDB = False
# model
_C.MODEL = CN()
_C.MODEL.DNUM = 1
_C.MODEL.BASEDIM = 1
_C.MODEL.BIN_LEN = 0.01
_C.MODEL.WALL_SIZE = 2.
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.OUT_CHANNELS = 1
_C.MODEL.TIME_SIZE = 512
_C.MODEL.IMAGE_SIZE = [256, 256]
_C.MODEL.HEATMAP_SIZE = [64, 64, 64]
_C.MODEL.TIME_DOWNSAMPLE_RATIO = 2 #8
_C.MODEL.IMAGE_DOWNSAMPLE_RATIO = 1 #4
_C.MODEL.PRETRAIN_AUTOENCODER = True
_C.MODEL.PRETRAIN_AUTOENCODER_PATH = "./lib/nlos_unet.pth"

_C.MODEL.MODE = 'lct'
_C.MODEL.NUM_JOINTS = 24
# data
_C.DATASET = CN()
_C.DATASET.PAN = 0.75
_C.DATASET.RATIO = 1.5
_C.DATASET.HEATMAP_SIZE = [64,64,64]
# _C.DATASET.TRAIN_PATH = "/data1/nlospose/zip/train/"
# _C.DATASET.TRAIN_PATH = "/home/liuping/data/mini_test/"
# _C.DATASET.TRAIN_PATH = "/data1/nlospose/person/person10/train/"

# _C.DATASET.TRAIN_PATH = "/home/liuping/data/mini_test_person2/"

# _C.DATASET.TRAIN_PATH = "/data1/nlospose/person_v2/person02/val/"
_C.DATASET.TRAIN_PATH = "/data1/nlospose/pose_v1/pose00/train"

# _C.DATASET.EVAL_PATH = "/home/liuping/data/mini_test/"
# _C.DATASET.EVAL_PATH = "/data1/nlospose/person/person10/val"

# _C.DATASET.EVAL_PATH = "/data1/nlospose/person_v2/person02/test/"
_C.DATASET.EVAL_PATH = "/data1/nlospose/pose_v1/pose00/train"

_C.DATASET.VOL_SIZE = [256,256,256]
_C.DATASET.DAWNSAMPLE_CNT = 0 #2
_C.DATASET.NUM_JOINTS = 24


# training process
_C.TRAIN = CN()
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 15
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.LR = 1e-1
_C.TRAIN.LR_STEP = [1,5,9]
_C.TRAIN.LR_FACTOR = 0.1