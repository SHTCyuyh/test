import os
from re import T
import sys
from turtle import Turtle
sys.path.append('./')
sys.path.append('/home/yuyh/network_backbone_posenet3d/nlospose/models')
# from lib.visualizer import joints_log

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from scipy.ndimage import zoom
from einops import rearrange
# from lib.vis_3view import vis_3view
import scipy.io as sio

from config import _C as cfg
    
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class NlosDataset(Dataset):
    
    def __init__(self, cfg, datapath):
        super().__init__()
        self.pan = cfg.DATASET.PAN
        self.ratio = cfg.DATASET.RATIO
        self.vol_size = cfg.DATASET.VOL_SIZE
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.downsample_cnt = cfg.DATASET.DAWNSAMPLE_CNT

        self.measFiles = []
        self.volFiles = []
        self.jointsFils = []
        self.framenum  = 9
        self.volPath = os.path.join(datapath, 'vol')
        self.jointsPath = os.path.join(datapath, 'joints')
        self.measPath = os.path.join(datapath, 'meas')
        measNames = os.listdir(self.measPath)
        volNames = os.listdir(self.volPath)
        for measName in measNames:
            assert os.path.splitext(measName)[1] == '.hdr', \
                f'Data type should be .hdr,not {measName} in {self.measPath}'
            measFile = os.path.join(self.measPath, measName)
            self.measFiles.append(measFile)

            volFile = os.path.join(self.volPath, os.path.splitext(measName)[0] + '.mat')
            assert os.path.isfile(volFile), \
                f'Do not have related vol {volFile}'
            self.volFiles.append(volFile)
            jointsFile = os.path.join(self.jointsPath, os.path.splitext(measName)[0] + '.joints')
            assert os.path.isfile(jointsFile), \
                f'Do not have related joints {jointsFile}'
            self.jointsFils.append(jointsFile)
        
        # self.measFiles = sorted(self.measFiles)
        # self.volFiles = sorted(self.volFiles)
        # self.jointsFils = sorted(self.jointsFils)

        self.len = len(self.measFiles)
        # temp1 = self.measFiles
        # temp2 = self.volFiles
        # temp3 = self.jointsFils
        # self.measFiles = []
        # self.volFiles = []
        # self.jointsFils = []
        # for i in range(self.len):
        #     self.measFiles.append(temp1[i:i+self.framenum])
        #     self.volFiles.append(temp2[i:i+self.framenum])
        #     self.jointsFils.append(temp3[i:i+self.framenum])
        
        # print(self.measFiles)


    def __getitem__(self, index):
        measFile = self.measFiles[index]
        volFile = self.volFiles[index]
        jointFile = self.jointsFils[index]
        # assert self.framenum == len(measFiles)
        # all_meas = []
        # all_joints = []
        # all_vol = []
        # all_person_id = []

    
        # measFile = measFiles[i]
        # volFile = volFiles[i]
        # jointFile = jointFiles[i]
        try:
            meas = cv2.imread(measFile, -1)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
        except TypeError:
            measFile = self.measFiles[0]
            jointFile = self.jointsFils[0]
            meas = cv2.imread(measFile, -1)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
            print(
                f'--------------------\nNo.{index} meas is TypeError. \n--------------------------\n')
        except:
            measFile = self.measFiles[0]
            jointFile = self.jointsFils[0]
            meas = cv2.imread(measFile, -1)
            meas = meas / np.max(meas)
            meas = cv2.cvtColor(meas, cv2.COLOR_BGR2GRAY)
            meas = meas / np.max(meas)
            print(
                f'--------------------\nNo.{index} meas is wrong. \n--------------------------\n')
        meas = rearrange(meas, '(t h) w ->t h w', t=600)[:512]
        vol = sio.loadmat(volFile)['vol']
        joints = np.loadtxt(jointFile)
        # meas = zoom(meas[:512, :, :], [0.5, 1, 1])  # too slow
        meas = (meas[::2] + meas[1::2]) / 2  # [256,256,256]
        for i in range(self.downsample_cnt):
            meas = (meas[::2] + meas[1::2]) / 2
            meas = (meas[:,::2] + meas[:,1::2]) / 2
            meas = (meas[:,:,::2] + meas[:,:,1::2]) / 2
            vol = (vol[::2] + vol[1::2]) / 2
            vol = (vol[:,::2] + vol[:,1::2]) / 2
            vol = (vol[:,:,::2] + vol[:,:,1::2]) / 2
        
        meas = rearrange(meas, 't h w -> 1 t h w')
        vol = rearrange(vol, 'd h w -> 1 d h w')
        joints[:, 0] = (joints[:, 0]*128+128)
        joints[:, 1] = 256-(joints[:, 1]*128+128)
        joints[:, 2] = 225-(joints[:, 2]*128+128)
        w = joints[:, 0].copy()
        h = joints[:, 1].copy()
        d = joints[:, 2].copy()
        joints[:, 0] = d 
        joints[:, 1] = h 
        joints[:, 2] = w 
        _, person_name = os.path.split(measFile)
        person_id, _ = os.path.splitext(person_name)
        joints = joints / (self.vol_size[0] / self.heatmap_size[0])
        
        # all_meas.append(meas)
        # all_vol.append(vol)
        # all_joints.append(joints)
        # all_person_id.append(person_id)
    
    # new_meas = np.stack(all_meas, axis=0)
    # new_vol = np.stack(all_vol, axis=0)
    # new_joints = np.stack(all_joints, axis=0)
    # new_person_id = np.stack(all_person_id, axis=0)

    

        return meas,vol, joints, person_id


    def __len__(self):
        
        return self.len

class data_prefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()
# input, vol, target_joints, person_id
    def preload(self):
        try:
            self.next_input,self.next_vol, self.next_target, self.next_personid = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_vol = None
            self.next_target = None
            self.next_personid = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.to(device=self.device,non_blocking=True)
            # self.next_vol = self.next_vol.to(device=self.device,non_blocking=True)
            self.next_target = self.next_target.to(device=self.device,non_blocking=True)
            # self.next_personid = self.next_personid

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input, vol, target_joints, person_id = self.next_input,self.next_vol, self.next_target, self.next_personid
        self.preload()
        return input, vol, target_joints, person_id

if __name__ == '__main__':
    import time
    # a = np.random.random((224, 224))
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    # print(a.sum())

    # plt.show()
    testDataLoader = True
    test_perfetcher = False
    test_generator = False
    if testDataLoader:
        datapath = "/data2/og_data/person16"
        train_data = NlosDataset(cfg, datapath)
        trainloader = DataLoader(train_data, batch_size=2, shuffle=False, pin_memory=True)

        # print(type(trainloader))
        # dataiter = iter(trainloader)
        # new_meas, new_vol, new_joints, new_person_id= dataiter.next()

        # print(new_person_id)
        t = time.time()
        for step, (input, vol, target_joints, person_id) in enumerate(trainloader):
            print(person_id)
            input = input.to("cuda:2")
            use_t = time.time() - t
            t = time.time()
            print(use_t)  #7-10

    if test_perfetcher:
        datapath = "/data2/og_data/person16"
        train_data = NlosDataset(cfg, datapath)
        trainloader = DataLoader(train_data, batch_size=3, shuffle=False, pin_memory=True)
        prefetcher = data_prefetcher(trainloader, cfg.DEVICE)
        t = time.time()
        input, vol, target_joints, person_id = prefetcher.next()
        # input = input.to("cuda:2")
        step = 0
        while input is not None:
            step += 1
            input, vol, target_joints, person_id = prefetcher.next()
            # input = input.to("cuda:2")
            use_t = time.time() - t
            t = time.time()
            print(use_t)

    if test_generator:
        datapath = "/data2/og_data/person16"
        train_data = NlosDataset(cfg, datapath)
        trainloader = DataLoaderX(train_data, batch_size=3, shuffle=False, pin_memory=True)

        # print(type(trainloader))
        # dataiter = iter(trainloader)
        # new_meas, new_vol, new_joints, new_person_id= dataiter.next()

        # print(new_person_id)
        t = time.time()
        for step, (input, vol, target_joints, person_id) in enumerate(trainloader):
            print(person_id)
            input = input.to("cuda:2")
            use_t = time.time() - t
            t = time.time()
            print(use_t)  #7-10       

 

