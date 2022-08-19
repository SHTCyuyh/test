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


def train(model, lenth,prefetcher, criterion, optimizer, cfg, epoch):
    

    model.train()
    total_ct = (cfg.TRAIN.END_EPOCH - cfg.TRAIN.BEGIN_EPOCH) * lenth
    time_begin = time.time()
    input, _, target_joints, person_id = prefetcher.next()
    step = 0
    while input is not None:
        step += 1
        input, _, target_joints, person_id = prefetcher.next()
        example_ct = epoch * lenth + step
        preds = model(input) # out_a: torch.Size([2, 1, 24, 3])  out_b torch.Size([2, 3, 9, 24])
        loss1 = criterion(preds,target_joints)
        loss = loss1 

        if example_ct % 5 == 0:
            train_log(cfg, loss, example_ct * cfg.TRAIN.BATCH_SIZE, epoch=epoch)

        if example_ct % 5 == 0 :
            used_time = time.time() - time_begin
            print(f'5 examples used {used_time}, finished {(example_ct / total_ct)*100}% ,'
                  + f"leave {used_time * (total_ct - example_ct)  / 3600} h")
            time_begin = time.time()

        # if example_ct % 4 == 0:
            # volume_log(vol, './results/volume', f"volume_{person_id}", example_ct)
            # volume_log(output, './results/volume', f"output_{person_id}", example_ct)
            # volume_log(feature, './results/volume', f'feature_{person_id}', example_ct)

            # pred = softmax_integral_tensor(output, cfg.DATASET.NUM_JOINTS, True,
            #                                cfg.DATASET.HEATMAP_SIZE[0], cfg.DATASET.HEATMAP_SIZE[1], cfg.DATASET.HEATMAP_SIZE[2])
        # if epoch % 20 == 0:   
        if epoch % 2 == 0: 
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
