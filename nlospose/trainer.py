import torch
import torch.nn as nn
import wandb
import time
import numpy as np

import matplotlib.pyplot as plt
from einops import rearrange, repeat
import sys
sys.path.append('./nlospose/models')
from criterion import L2JointLocationLoss, softmax_integral_tensor

from lib.visualizer import threeviews_log, volume_log, joints_log, threeviews_log

def train(model, dataloader, criterion, optimizer, cfg, epoch):
    if cfg.WANDB:
        wandb.watch(model)
    
    model.train()
    total_ct =  (cfg.TRAIN.END_EPOCH - cfg.TRAIN.BEGIN_EPOCH) * len(dataloader)
    time_begin = time.time()
    for step, (input, vol, target_joints, person_id) in enumerate(dataloader):
        example_ct = epoch * len(dataloader) + step
        
        input = input.to(cfg.DEVICE)
        output, feature = model(input)
        target_joints = rearrange(target_joints, 'b n d -> b (n d)').to(cfg.DEVICE)
        target_weights = torch.ones_like(target_joints).to(cfg.DEVICE)
        loss = criterion(output, target_joints, target_weights)

        # if example_ct % 5 == 0:
        #     train_log(cfg, loss, example_ct, epoch)
            
        # if example_ct % 50 == 0:
        #     used_time = time.time() - time_begin
        #     print(f'50 examples used {used_time}, finished {(example_ct / total_ct)*100}% ,'\
        #         + f"leave {used_time * total_ct / 50 / 3600} h")
        #     time_begin = time.time()
            
        # if example_ct % 100 == 0:
        #     # mlab.contour3d(input[0,0].detach().cpu().numpy())
        #     # mlab.savefig(f"./obj/input_{example_ct}.obj")
        #     # mlab.contour3d(output[0,0].detach().cpu().numpy())
        #     # mlab.savefig(f"./obj/output_{example_ct}.obj")

        #     torch.save(vol[0,0], f"./results/obj/vol_{example_ct}.pt")
        #     # torch.save(output[0,0], f"./results/obj/output_{example_ct}.pt")
        #     # torch.save(feature[0,0], f"./results/obj/feature_{example_ct}.pt")
        #     im_vol = vol[0,0].sum(0).detach().cpu().numpy()
        #     plt.imshow(im_vol)
        #     wandb.log({"im_vol": plt})
        #     plt.savefig(f"./results/fig/vol_{example_ct}.jpg")
        #     # im_output = output[0,0].sum(0).detach().cpu().numpy()
        #     # plt.imshow(im_output)
        #     # wandb.log({"im_output": plt})
        #     # plt.savefig(f"./results/fig/output_{example_ct}.jpg")
        #     # im_feature = feature[0,0].sum(0).detach().cpu().numpy()
        #     # plt.imshow(im_feature)
        #     # plt.savefig(f"./results/fig/feature_{example_ct}.jpg")
        # if example_ct % 200 == 0:
        #     # vis_3view(feature, f'feature_{example_ct}')
        #     # vis_3view(output, f'output_{example_ct}')
        #     # vis_3view(vol, f'volume_{example_ct}')

        #     pred = softmax_integral_tensor(output, 24, True, \
        #         cfg.DATASET.HEATMAP_SIZE[0], cfg.DATASET.HEATMAP_SIZE[1], cfg.DATASET.HEATMAP_SIZE[2])
        #     pred = pred.reshape(input.shape[0], 24, 3).detach().cpu().numpy()
        #     np.savetxt(f"./joint/pred_{example_ct}.txt", pred[0])

        #     im_pred = np.zeros((64,64))
        #     for i in range(24):
        #         x = pred[0,i,0].astype(np.int64)
        #         y = pred[0,i,1].astype(np.int64)
        #         im_pred[x:x+2,y:y+2] = 10
        #     plt.imshow(im_pred)
        #     plt.savefig(f"./fig/pred_2d_{example_ct}.jpg")

        #     gt = target_joints
        #     gt = gt.reshape(input.shape[0], 24, 3).detach().cpu().numpy()
        #     np.savetxt(f"./joint/gt_{example_ct}.txt", gt[0])

        #     im_gt = np.zeros((64,64))
        #     for i in range(24):
        #         x = gt[0,i,0].astype(np.int64)
        #         y = gt[0,i,1].astype(np.int64)
        #         im_gt[x:x+2,y:y+2] = 10
        #     plt.imshow(im_gt)
        #     plt.savefig(f"./fig/gt_2d_{example_ct}.jpg")
        # loss = criterion(output, vol.to(cfg.DEVICE))

        if example_ct % 5 == 0:
            train_log(loss, example_ct, epoch)
            
        if example_ct % 50 == 0:
            used_time = time.time() - time_begin
            print(f'50 examples used {used_time}, finished {(example_ct / total_ct)*100}% ,'\
                + f"leave {used_time * total_ct / 50 / 3600} h")
            time_begin = time.time()
            
        if example_ct % 100 == 0:
            volume_log(vol, './results/volume', "volume", example_ct)
            volume_log(output, './results/volume', "output", example_ct)
            volume_log(feature, './results/volume', 'feature', example_ct)

            # torch.save(vol[0,0], f"./results/obj/vol_{example_ct}.pt")
            # torch.save(output[0,0], f"./results/obj/output_{example_ct}.pt")
            # torch.save(feature[0,0], f"./results/obj/feature_{example_ct}.pt")
            # im_vol = vol[0,0].sum(0).detach().cpu().numpy()
            # plt.imshow(im_vol)
            # wandb.log({"im_vol": plt})
            # plt.savefig(f"./results/fig/vol_{example_ct}.jpg")
            # im_output = output[0,0].sum(0).detach().cpu().numpy()
            # plt.imshow(im_output)
            # wandb.log({"im_output": plt})
            # plt.savefig(f"./results/fig/output_{example_ct}.jpg")
            threeviews_log(feature, './results/figure/threeviews', f'feature_{example_ct}')
            threeviews_log(output, './results/figure/threeviews', f'output_{example_ct}')
            threeviews_log(vol, './results/figure/threeviews', f'volume_{example_ct}')

            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def train_log(cfg, loss, example_ct, epoch):
    if cfg.WANDB:
        wandb.log({"epoch": epoch, "loss": loss}, step = example_ct)
    print(f"Loss after  " + str(example_ct).zfill(5) + f"  examples:  {loss:.3f} ")
