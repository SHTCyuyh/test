import os
import sys

from matplotlib import offsetbox
sys.path.append('./')
sys.path.append('/home/yuyh/network_backbone_posenet3d/nlospose/models')
# from lib.visualizer import joints_log

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from einops import rearrange
# from lib.vis_3view import vis_3view
import scipy.io as sio

from config import _C as cfg

    

class NlosDataset(Dataset):
    
    def __init__(self, cfg, datapath, framenum=9, mode=None):
        super().__init__()
        self.pan = cfg.DATASET.PAN
        self.ratio = cfg.DATASET.RATIO
        self.vol_size = cfg.DATASET.VOL_SIZE
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.downsample_cnt = cfg.DATASET.DAWNSAMPLE_CNT
        self.persons = cfg.persons

        self.measFiles = []
        self.volFiles = []
        self.jointsFils = []
        self.framenum  = framenum
        # self.volPath = os.path.join(datapath, 'vol')
        # self.jointsPath = os.path.join(datapath, 'joints')
        # self.measPath = os.path.join(datapath, 'meas')
        self.motion_num = cfg.motion_num
        self.motions = {}      
        for person in self.persons:
            for num in range(self.motion_num):
                if mode == 'train':
                   self.motions[f'{person}_motion{num+1}'] = cfg[person][0][f'motion{num+1}']
                if mode == 'test':
                   self.motions[f'{person}_motion{num+1}'] = cfg[person][1][f'motion{num+1}']
        
        for person in self.persons:
            self.volPath = os.path.join(datapath, person, 'vol')
            self.jointsPath = os.path.join(datapath, person, 'joints')
            self.measPath = os.path.join(datapath, person, 'meas')
            measNames = os.listdir(self.measPath)
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
            
            temp1 = sorted(self.measFiles)
            temp2 = sorted(self.volFiles)
            temp3 = sorted(self.jointsFils)
            self.measFiles = []
            self.volFiles = []
            self.jointsFils = []
            step = cfg.step
            for key in self.motions:
                offset, be, end = self.motions[key]
                bename = os.path.join(self.measPath,f'{person}-{str(be).zfill(5)}.hdr')
                number = int(((end+1 - be)-self.framenum)/step)
                for i in range(number):
                    if i == 0: index = temp1.index(bename)
                    else:
                        index += step
                    self.measFiles.append(temp1[index:index+self.framenum])
                    self.volFiles.append(temp2[index:index+self.framenum])
                    self.jointsFils.append(temp3[index:index+self.framenum])
                    if i == number-2: break
                

        num = len(self.measFiles)
        self.len = num
        
        # print(self.measFiles)


    def __getitem__(self, index):
        measFiles = self.measFiles[index]
        volFiles = self.volFiles[index]
        jointFiles = self.jointsFils[index]
        assert self.framenum == len(measFiles)
        all_meas = []
        all_joints = []
        all_vol = []
        all_person_id = []

        for i in range (self.framenum):
            measFile = measFiles[i]
            volFile = volFiles[i]
            jointFile = jointFiles[i]
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
            
            all_meas.append(meas)
            all_vol.append(vol)
            all_joints.append(joints)
            all_person_id.append(person_id)
        
        new_meas = np.stack(all_meas, axis=0)
        new_vol = np.stack(all_vol, axis=0)
        new_joints = np.stack(all_joints, axis=0)
        # new_person_id = np.stack(all_person_id, axis=0)

        

        return new_meas, new_vol, new_joints, all_person_id


    def __len__(self):
        
        return self.len



if __name__ == '__main__':
    # a = np.random.random((224, 224))
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    # print(a.sum())

    # plt.show()

    testDataLoader = True
    if testDataLoader:
        datapath = "/data2/og_data"
        train_data = NlosDataset(cfg, datapath, mode='test')
        trainloader = DataLoader(train_data, batch_size=2, shuffle=False, pin_memory=True)


        for step, (input, vol, target_joints, person_id) in enumerate(trainloader):
            print(person_id)

    

 

