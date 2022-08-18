import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from einops import rearrange

from feature_extraction import FeatureExtraction, FeatureExtracion_UNet
from feature_propagation import FeaturePropagation, VisibleNet, normalize_feature
# from models.NlosPoseSformer import NlosPoseSformer
from posenet import get_config, get_pose_net
# from models.tokenpose import TokenPose_L
from nlospose.models.posenet3d_50_new import get_pose_net_50
from unet.unet3d import UNet3d, freeze_layer
from heatmap import heatmap

# from models.token_config import _C as tokenpose_cfg


class NlosPose(nn.Module):
	def __init__(self, cfg):
		super().__init__()

		self.time_begin = 0
		self.time_end = cfg.MODEL.TIME_SIZE

		self.feature_extraction = FeatureExtraction(
			basedim=cfg.MODEL.BASEDIM,
			in_channels=cfg.MODEL.IN_CHANNELS,
			stride=1
		)
		self.feature_propagation = FeaturePropagation(
			time_size=cfg.MODEL.TIME_SIZE,
			image_size=cfg.MODEL.IMAGE_SIZE[0],
			wall_size=cfg.MODEL.WALL_SIZE,
			bin_len=cfg.MODEL.BIN_LEN,
			dnum=cfg.MODEL.DNUM,
			dev=cfg.DEVICE
		)

		if cfg.MODEL.PRETRAIN_AUTOENCODER == True:
			self.autoencoder = torch.load(cfg.MODEL.PRETRAIN_AUTOENCODER_PATH, map_location=f"cuda:{cfg.DEVICE}")
			# freeze_layer(self.autoencoder)
		else:self.autoencoder = UNet3d(
                in_channels=1,
                n_channels=4,
            )
		if cfg.MODEL.BACKBONE == 'posenet2d':
			self.vis_net = VisibleNet(basedim=3)
			self.pose_net_cfg = get_config()
			self.pose_net = \
				get_pose_net(self.pose_net_cfg, num_joints=cfg.MODEL.NUM_JOINTS)
		elif cfg.MODEL.BACKBONE == '3d_resnet50':
			self.pose_net = get_pose_net_50()

		self.headmap = heatmap(output_3d=True)

	def forward(self, meas): # (2,1,128,64,64)
        # num = meas.shape[0]
        # tbens = []
        # tends = []
        # for i in range (num):
        #     tbens.append(0)
        #     tends.append(x.shape[2])		
		meas = self.feature_extraction(meas) 
		 # (2,2,64,32,32)
		# feature = self.feature_extraction_unet(meas.squeeze())
		feature = self.feature_propagation(meas, [self.time_begin, self.time_begin, self.time_begin], [self.time_end , self.time_end, self.time_end])  #(2, 1, 128, 64, 64)
		feature = normalize_feature(feature)
		
		refine_feature = self.autoencoder(feature)		
		# output = rearrange(output, 'b c d h w -> b (c d) h w')
		output = self.pose_net(feature + refine_feature)
		# output = rearrange(output, 'b (n d) h w -> b n d h w', n = self.pose_net_cfg.num_joints)
		output = self.headmap(output)
		B = output.size(0)
		output = output.view(B, -1, 3)
		return output





if __name__ == '__main__':
	from config import _C as cfg
	cfg['DEVICE'] = 1
	cfg.MODEL.BACKBONE = 'posenet3d_50'
	model = NlosPose(cfg)
	video = torch.randn(1,1,cfg.MODEL.TIME_SIZE,cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
	video = video.to('cuda:1')
	model = model.to('cuda:1')
	pre = model(video)


