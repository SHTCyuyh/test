from pickle import TRUE
from yacs.config import CfgNode as CN

# hardware
_C = CN()
_C.PROJECT_NAME = "nlosformer_depth_embed_0816"
_C.DEVICE = (2)  # ATTENTION: nlos_unet load in 'meas2vol.py' only support ONE GPU
_C.NUM_WORKERS = 18
_C.WANDB = True
# model
_C.MODEL = CN()
_C.MODEL.DNUM = 1
_C.MODEL.BASEDIM = 1
_C.MODEL.BIN_LEN = 0.01 * 8
_C.MODEL.WALL_SIZE = 2.
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.OUT_CHANNELS = 1
_C.MODEL.TIME_SIZE = 64
_C.MODEL.IMAGE_SIZE = [64, 64]
_C.MODEL.HEATMAP_SIZE = [64, 64, 64]
_C.MODEL.TIME_DOWNSAMPLE_RATIO = 1 #8
_C.MODEL.IMAGE_DOWNSAMPLE_RATIO = 1 #4
_C.MODEL.PRETRAIN_AUTOENCODER = True
_C.MODEL.PRETRAIN_AUTOENCODER_PATH = "./lib/nlos_unet.pth"
_C.MODEL.BACKBONE = '3d_resnet50'
_C.MODEL.SAVEMODELPATH = './checkpoint'

_C.MODEL.MODE = 'lct'
_C.MODEL.NUM_JOINTS = 24
# data
_C.DATASET = CN()
_C.DATASET.PAN = 0.75
_C.DATASET.RATIO = 4
_C.DATASET.HEATMAP_SIZE = [64,64,64]
# _C.DATASET.TRAIN_PATH = "/data1/nlospose/zip/train/"
# _C.DATASET.TRAIN_PATH = "/home/liuping/data/mini_test/"
# _C.DATASET.TRAIN_PATH = "/data1/nlospose/person/person10/train/"

# _C.DATASET.TRAIN_PATH = "/home/liuping/data/mini_test_person2/"

# _C.DATASET.TRAIN_PATH = "/data1/nlospose/person_v2/person02/val/"
# _C.DATASET.TRAIN_PATH = "/data1/nlospose/pose_v1/pose00/train"
_C.DATASET.TRAIN_PATH = '/data2/og_data/person16'

# _C.DATASET.EVAL_PATH = "/home/liuping/data/mini_test/"
# _C.DATASET.EVAL_PATH = "/data1/nlospose/person/person10/val"

# _C.DATASET.EVAL_PATH = "/data1/nlospose/person_v2/person02/test/"
# _C.DATASET.EVAL_PATH = "/data1/nlospose/pose_v1/pose00/train"
_C.DATASET.EVAL_PATH = '/data2/og_data/person15'

_C.DATASET.VOL_SIZE = [256,256,256]
_C.DATASET.DAWNSAMPLE_CNT = 2 #2
_C.DATASET.NUM_JOINTS = 24


# training process
_C.TRAIN = CN()
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 10
_C.TRAIN.BATCH_SIZE = 3
_C.TRAIN.LR = 1e-5
_C.TRAIN.LRD = 0.99
_C.TRAIN.LR_STEP = [1,2,3,4,5,6,7,8,9]
_C.TRAIN.LR_FACTOR = 0.5
_C.TRAIN.OPTIMIZER = 'AdamW'


_C.mode = 'train'
_C.motion_num = 4
_C.step = 4

#person number
_C.persons = ['person16']
# _C.person = {}

person16_motion_train = {
    'motion1':[0, 0, 800],
    'motion2':[0, 1021, 1820],
    'motion3':[0, 2035, 2835],
    'motion4':[0, 3047, 3847]
}
person16_motion_test = {
    'motion1':[0, 801, 1019],
    'motion2':[0, 1821, 2034],
    'motion3':[0, 2836, 3046],
    'motion4':[0, 3848, 4066],
}

_C.person16 = [person16_motion_train, person16_motion_test]

