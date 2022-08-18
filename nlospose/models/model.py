import sys
from turtle import Turtle
import torch
from torch import nn
# from posenet import get_config, get_pose_net
from torchsummary import summary

from einops import rearrange, repeat
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append('./')
sys.path.append('./nlospose/models')

from config import _C as cfg
from lib.vis_3view import vis_3view
from lib.utils import freeze_layer, downsample
from nlos_dataloader import NlosDataset
from feature_propagation import FeaturePropagation, normalize
from feature_extraction import FeatureExtraction
from unet.unet3d import UNet3d
from posenet3d import PoseNet3d
from nlospose.models.posenet3d_50_new import get_config, get_pose_net
from v2vnet import V2VModel
from vis_layer import VisibleNet
from model_poseformer import PoseTransformer
from heatmap import heatmap
import numpy as np




class Meas2Pose(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feature_extraction = FeatureExtraction(
            basedim=cfg.MODEL.BASEDIM,
            in_channels=cfg.MODEL.IN_CHANNELS,
            stride=1,
        )
        self.feature_propagation = FeaturePropagation(
            time_size=cfg.MODEL.TIME_SIZE // cfg.MODEL.TIME_DOWNSAMPLE_RATIO,
            image_size=cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.IMAGE_DOWNSAMPLE_RATIO,
            wall_size=cfg.MODEL.WALL_SIZE,
            bin_len=cfg.MODEL.BIN_LEN * cfg.MODEL.TIME_DOWNSAMPLE_RATIO,
            dnum=cfg.MODEL.DNUM,
            dev=cfg.DEVICE,
        )
        self.poseformer = PoseTransformer()

        if cfg.MODEL.PRETRAIN_AUTOENCODER == True:
            self.unet3d = torch.load(cfg.MODEL.PRETRAIN_AUTOENCODER_PATH, map_location=f"cuda:{cfg.DEVICE}")
            # freeze_layer(self.unet3d)
        else:
            self.unet3d = UNet3d(
                in_channels=1,
                n_channels=4,
            )
        
        self.heatmap = heatmap(output_3d=True)

        # voxel2voxel net
        # self.pose_net = V2VModel(1, cfg.DATASET.NUM_JOINTS)

        # res3d net
        if cfg.MODEL.BACKBONE == '3d_resnet18':
            self.pose_net = PoseNet3d(512, 2, 256, 4, 1, 24, 64, True, pretrained=False, progress=True)
        elif cfg.MODEL.BACKBONE == '3d_resnet50':
            self.pose_net = get_pose_net()

        
        # 2d pose net
        # self.vis_net = VisibleNet(basedim=3)
        # self.pose_net_cfg = get_config() 
        # self.pose_net = get_pose_net(self.pose_net_cfg, num_joints=24)

    def forward(self, x):
        # x = self.feature_extraction(x)
        num = x.shape[0]
        tbens = []
        tends = []
        for i in range (num):
            tbens.append(0)
            tends.append(x.shape[2])
        residual_x = self.feature_propagation(x, tbens, tends)
        # x = residual_x
        x = normalize(residual_x)
        # x = self.unet3d(x)
        # x = x + residual_x

        feature = x
        # x_with_skip = x + feature
        # x = downsample(x, 1)
        # x = self.vis_net(x)
        # x = rearrange(x, 'b c t h w -> b (c t) h w')

        # x = rearrange(x[0, 0], '(b c h) w -> b c h w', b=1, c=1)

        # x = self.pose_net(x)
        x = self.pose_net(x)  #torch.Size([2, 24, 64, 64, 64])
        x = self.heatmap(x)
        
        return x, feature


if __name__ == '__main__':

    # datapath = "/home/liuping/data/mini_test/"
    # datapath = "/data1/nlospose/zip/train/"
    # datapath = "/data1/nlospose/using/train/"

    # datapath = '/data1/nlospose/pose_v1/person00_mini/'
    # datapath = '/data1/nlospose/pose_v1/pose00/train'
    datapath = '/data2/motion_test'
    train_dataset = NlosDataset(cfg, datapath)
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=False, pin_memory=True)

    input, vol, joints, person_id = next(iter(train_dataloader))
    # import numpy as np
    # input = np.zeros((8,1,32,32,64))  #torch.Size input ([8, 1, 64, 64, 64])
    # input = torch.from_numpy(input)   

    input = input.to(cfg.DEVICE)

    model = Meas2Pose(cfg).to(cfg.DEVICE)
    # input = torch.ones((1, 1, 128, 256, 256)).to(cfg.DEVICE)
    output, feature= model(input)  #torch.Size([2, 24, 64, 64, 64])    x b , 72
    vis_3view(feature, "lct_recon")
    vis_3view(vol, "recon")
    # vis_3view(output, "output")
    # vis_3view(output+feature, "lct_recon + recon")
    # summary(model, (1, 256, 256, 256))

    # layer = torch.load('./results/model/nlos_unet.pth')
    # freeze_layer(cfg, layer)
    # output = layer(input)
	
    print(output.shape)
