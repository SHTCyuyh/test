import sys
import torch
from torch import nn
# from posenet import get_config, get_pose_net
from torchsummary import summary

from einops import rearrange, repeat
import matplotlib.pyplot as plt
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
from v2vnet import V2VModel
from vis_layer import VisibleNet


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
        if cfg.MODEL.PRETRAIN_AUTOENCODER == True:
            self.unet3d = torch.load(cfg.MODEL.PRETRAIN_AUTOENCODER_PATH, map_location=f"cuda:{cfg.DEVICE}")
            freeze_layer(self.unet3d)
        else:
            self.unet3d = UNet3d(
                in_channels=1,
                n_channels=4,
            )
        # self.vis_net = VisibleNet(basedim=3)
        # self.pose_net = V2VModel(1, cfg.DATASET.NUM_JOINTS)
        self.pose_net = PoseNet3d(512, 2, 256, 4, 1, 24, 64, True, pretrained=False, progress=True)
        # self.pose_net_cfg = get_config() 
        # self.pose_net = get_pose_net(self.pose_net_cfg, num_joints=24)

    def forward(self, x):
        x = self.feature_extraction(x)
        feature = self.feature_propagation(x, [0, 0], [x.shape[2], x.shape[2]])
        feature = normalize(feature)
        x = self.unet3d(feature)
        x = x + feature
        # x_with_skip = x + feature
        x = downsample(x, 1)
        # x = self.vis_net(x)
        # x = rearrange(x, 'b c t h w -> b (c t) h w')

        # x = rearrange(x[0, 0], '(b c h) w -> b c h w', b=1, c=1)

        # x = self.pose_net(x)
        x = self.pose_net(x)
        
        return x, feature


if __name__ == '__main__':

    # datapath = "/home/liuping/data/mini_test/"
    # datapath = "/data1/nlospose/zip/train/"
    # datapath = "/data1/nlospose/using/train/"
    datapath = "/data1/nlospose/person/person10/val/"
    train_dataset = NlosDataset(cfg, datapath)
    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=False, pin_memory=True)

    input, vol, joints, person_id = next(iter(train_dataloader))
    input = input.to(cfg.DEVICE)
    model = Meas2Vol(cfg).to(cfg.DEVICE)
    # input = torch.ones((1, 1, 128, 256, 256)).to(cfg.DEVICE)
    output, feature= model(input)
    vis_3view(feature, "lct_recon")
    vis_3view(vol, "recon")
    vis_3view(output, "output")
    # vis_3view(output+feature, "lct_recon + recon")
    # summary(model, (1, 256, 256, 256))

    # layer = torch.load('./results/model/nlos_unet.pth')
    # freeze_layer(cfg, layer)
    # output = layer(input)
	
    print(output.shape)
