from asyncio import FastChildWatcher
from importlib.resources import path
import math
import logging
from functools import partial
from collections import OrderedDict
from os import pathconf_names
from re import I, X
from tkinter.dnd import DndHandler
from turtle import forward
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from functools import reduce, lru_cache
from operator import mul

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention3D(nn.Module):
    '''
    '''
    def __init__(self, dim, patch_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_mask=False):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size #d,h,w
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_mask = use_mask

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x, transients, mask=None):
        B, N, C = x.shape  #B N 
        assert x.shape == transients.shape,\
        'wrong meas and lct_vol'


        q_vec = transients
        k_vec = x
        v_vec = x

        q = self.q_linear(q_vec).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_linear(k_vec).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_linear(v_vec).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1) * self.scale 
        attn_mask = get_attn_subsequent_mask(attn).bool()
        attn.masked_fill_(attn_mask, -1e9)
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)        
        return x


class Block3D(nn.Module):
    def __init__(self, dim, num_heads, patch_size=(4,4,4), mlp_ratio=4, 
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = Attention3D(
            dim, patch_size=self.patch_size, num_heads=num_heads, 
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_mask=False
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, transients,  mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), transients))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x          

class Patchmerge(nn.Module):
    pass


class Patchembed3D(nn.Module):
    def __init__(self, patch_size=(4,64,64), in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj =  nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            norm_layer = norm_layer(embed_dim)
        else:
            self.norm = None
        
    def forward(self, x):
        _, _, D, H, W = x.size()  #B C D H W
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        
        x = self.proj(x) # B C D H W

        if self.norm is not None:
            D, H, W = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, H, W)
        
        return x

# class head(nn.Module):
#     def __init__(self, norm_layer=nn.LayerNorm, embed_dim=16, num_joints=24):
#         super().__init__() 
#         self.norm_layer = norm_layer
#         self.num_joints = num_joints
#         self.liner = nn.Linear(num_joints)

#     def forward(self, x):
#         x = rearrange(x,'b c d h w -> b c (d h w)')
#         x = self.norm_layer(x)
#         x = self.liner(x)

#         return x

# class computemask()

def get_attn_subsequent_mask(seq):
    attn_shape = seq.size()
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)     # 上三角矩阵
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.device is not None:
       Device = seq.device
       subsequent_mask = subsequent_mask.to(Device)
    return subsequent_mask

class nlosformer(nn.Module):
    def __init__(self, num_joints=24, input_size=(64,64,64),patch_size=(1,1,1), in_chans=1, embed_dim=96,
                 depths=2, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                 patch_norm=False
                 ):
        super().__init__()
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_layers = depths
        self.patch_norm = patch_norm
        self.input_size = input_size
       

        ## 3D patch embedding
        self.patch_embed = Patchembed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=1,
            norm_layer=norm_layer if self.patch_norm else None
        )
        dd, dh, dw = int(list(input_size)[0]/list(patch_size)[0]), int(list(input_size)[1]/list(patch_size)[1]), int(list(input_size)[2]/list(patch_size)[2])
        self.num_patchs = dd * dh * dw 
        self.pos_embed = nn.Parameter(torch.zeros(1, dd, dh, dw, 1))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.blocks =  nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Block3D(
                dim=dd, num_heads=num_heads, patch_size=patch_size, mlp_ratio=mlp_ratio, 
                 qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i_layer],
                 norm_layer=norm_layer
            )
            self.blocks.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))
        # self.liner = nn.Linear(self.num_patchs * embed_dim, self.num_joints * 3)

        ####### A easy way to implement weighted mean
        # self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)
        out_dim = num_joints * 3
        self.norm = nn.LayerNorm(dd)
        self.linear = nn.Linear(self.num_patchs , out_dim)
        # self.head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(self.num_patchs *embed_dim , out_dim),
        # )

    
    def forward(self, x, transients):  # B C D H W
        x = self.patch_embed(x)
        x = rearrange(x, 'n c d h w -> n d h w c')
        B, C, D, H, W, = x.shape
        x += self.pos_embed
        x = self.pos_drop(x)

        transients = self.patch_embed(transients)
        transients = rearrange(transients, 'n c d h w -> n d h w c')
        # B, C, D, H, W, = x.shape
        transients += self.pos_embed
        transients = self.pos_drop(transients)

        x = rearrange(x, 'n d h w c -> n (h w) (d c)')
        transients = rearrange(transients, 'n d h w c -> n (h w) (d c)')

        for blk in self.blocks:
            x = blk(x, transients)
        
        # x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'B N C -> B (N C)')
        x = self.linear(x)
        x = x.view(B, -1, 3)


        return x

class transient_attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)    
        # self.embed = transient_preprocess()  
    def forward(self, x):
        # x = self.patchembed(x) # b c d w h -> b c d/p w/p h/p
        # x = rearrange(x, 'b c d h w -> b d (c h w) ')
        # x = self.embed(x)  #b d hw  make depth as channels
        x = x.permute(0, 2, 1) #b hw d
        # attn_mask = get_attn_subsequent_mask(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # if attn_mask is not None:
        attn_mask = get_attn_subsequent_mask(attn).bool()
        attn.masked_fill_(attn_mask, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x 

if __name__ == '__main__':
    model = nlosformer()
    x = np.random.random((2,1,64,64,64))
    x = x.astype(np.float32)
    # x.dtype   # b c d h w
    x = torch.from_numpy(x)
    # x = 
    y = x.contiguous()
    out = model(x, y)  #[B ,CLS]
    print(out.shape)
    
            


