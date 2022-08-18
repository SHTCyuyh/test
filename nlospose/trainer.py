from lzma import PRESET_DEFAULT
from lib.visualizer import threeviews_log, volume_log, joints_log, threeviews_log
import torch
import torch.nn as nn
import wandb
import time
import numpy as np

import matplotlib.pyplot as plt
from einops import rearrange, repeat
import sys
sys.path.append('./nlospose/models')
from models.loss import mpjpe, n_mpjpe, p_mpjpe


def train(model, dataloader, criterion, optimizer, cfg, epoch):
    

    model.train()
    total_ct = (cfg.TRAIN.END_EPOCH - cfg.TRAIN.BEGIN_EPOCH) * len(dataloader)
    time_begin = time.time()
    for step, (input, vol, target_joints, person_id) in enumerate(dataloader):
        
        
        example_ct = epoch * len(dataloader) + step

        input = input.to(cfg.DEVICE)  #torch.Size([1, 1, 256, 256, 256])
        
        preds = model(input) # out_a: torch.Size([2, 1, 24, 3])  out_b torch.Size([2, 3, 9, 24])
        # output_befor = rearrange(
        #     output_befor, 'b c f n -> b f n c').to(cfg.DEVICE)
        # target_joints = rearrange(
        #     target_joints, 'b f n d -> b (n d)').to(cfg.DEVICE)   #torch.Size([2, 9, 24, 3])
        # target_weights = torch.ones_like(target_joints).to(cfg.DEVICE)
        # loss = criterion(output, target_joints, target_weights)
        # frame_num = target_joints.shape[1]
        # mid = frame_num//2
        # mid_joints = target_joints[:,mid]
        target_joints = target_joints.to(cfg.DEVICE)
        # mid_joints = mid_joints.to(cfg.DEVICE)
        loss1 = criterion(preds,target_joints)
        
        # np.savetxt('./1.txt', mid_joints.cpu().numpy().reshape(24,-1))
        # output_after = output_after[:,0]
        # loss2 = criterion(output_after, mid_joints)
        loss = loss1 

        if example_ct % 5 == 0:
            train_log(cfg, loss, example_ct, epoch=epoch)

        if example_ct % 5 == 0 :
            used_time = time.time() - time_begin
            print(f'5 examples used {used_time}, finished {(example_ct * cfg.TRAIN.BATCH_SIZE / total_ct)*100}% ,'
                  + f"leave {used_time * total_ct / (5 * cfg.TRAIN.BATCH_SIZE) / 3600} h")
            time_begin = time.time()

        # if example_ct % 4 == 0:
            # volume_log(vol, './results/volume', f"volume_{person_id}", example_ct)
            # volume_log(output, './results/volume', f"output_{person_id}", example_ct)
            # volume_log(feature, './results/volume', f'feature_{person_id}', example_ct)

            # pred = softmax_integral_tensor(output, cfg.DATASET.NUM_JOINTS, True,
            #                                cfg.DATASET.HEATMAP_SIZE[0], cfg.DATASET.HEATMAP_SIZE[1], cfg.DATASET.HEATMAP_SIZE[2])
        # if epoch % 20 == 0:   
        if example_ct % 10 ==0 and epoch % 3 == 0: 
            for i in range(preds.shape[0]):
                pred1 = preds[i]
                # pred2 = output_after[1]
                joints_log(pred1.reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                           f'./results_{cfg.PROJECT_NAME}/figure/joints',
                           f"pred_joints_{person_id[i]}",
                           epoch)
                # joints_log(pred2.reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                #            './results/figure/joints',
                #            f"pred_joints_{person_id[mid][1]}",
                        #    epoch)
                gt1 = target_joints[i]
                # gt2 = mid_joints[1]
                joints_log(gt1.reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                           f'./results_{cfg.PROJECT_NAME}/figure/joints',
                           f"gt_joints_{person_id[i]}",
                           epoch)
                # joints_log(gt2.reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                #        './results/figure/joints',
                #        f"gt_joints_{person_id[mid][1]}",
                #        epoch)
    
                # threeviews_log(feature, './results/figure/threeviews',
                #                f'feature_{person_id}', example_ct)
                # threeviews_log(output, './results/figure/threeviews',
                #                f'output_{person_id}', example_ct)
                # threeviews_log(vol, './results/figure/threeviews',
                #                f'volume_{person_id}', example_ct)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_log(cfg, loss, example_ct, epoch):
    if cfg.WANDB:
        wandb.log({"epoch": epoch, "loss": loss}, step=example_ct, commit=True)
    print(f"Loss after  " + str(example_ct).zfill(5) +
          f"  examples:  {loss:.3f} ")
