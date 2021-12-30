from lib.visualizer import threeviews_log, volume_log, joints_log, threeviews_log
from criterion import L2JointLocationLoss, softmax_integral_tensor
import torch
import torch.nn as nn
import wandb
import time
import numpy as np

import matplotlib.pyplot as plt
from einops import rearrange, repeat
import sys
sys.path.append('./nlospose/models')


def train(model, dataloader, criterion, optimizer, cfg, epoch):
    if cfg.WANDB:
        wandb.watch(model, log='all')

    model.train()
    total_ct = (cfg.TRAIN.END_EPOCH - cfg.TRAIN.BEGIN_EPOCH) * len(dataloader)
    time_begin = time.time()
    for step, (input, vol, target_joints, person_id) in enumerate(dataloader):
        np.savetxt('./1.txt', target_joints.cpu().numpy().reshape(24,-1))
        
        example_ct = epoch * len(dataloader) + step

        input = input.to(cfg.DEVICE)
        output, feature = model(input)
        target_joints = rearrange(
            target_joints, 'b n d -> b (n d)').to(cfg.DEVICE)
        target_weights = torch.ones_like(target_joints).to(cfg.DEVICE)
        loss = criterion(output, target_joints, target_weights)

        if example_ct % 5 == 0:
            train_log(cfg, loss, example_ct, epoch=epoch)

        if example_ct % 50 == 0:
            used_time = time.time() - time_begin
            print(f'50 examples used {used_time}, finished {(example_ct * cfg.TRAIN.BATCH_SIZE / total_ct)*100}% ,'
                  + f"leave {used_time * total_ct / (50 * cfg.TRAIN.BATCH_SIZE) / 3600} h")
            time_begin = time.time()

        if example_ct % 1 == 0:
            volume_log(vol, './results/volume', f"volume_{person_id}", example_ct)
            volume_log(output, './results/volume', f"output_{person_id}", example_ct)
            volume_log(feature, './results/volume', f'feature_{person_id}', example_ct)

            pred = softmax_integral_tensor(output, cfg.DATASET.NUM_JOINTS, True,
                                           cfg.DATASET.HEATMAP_SIZE[0], cfg.DATASET.HEATMAP_SIZE[1], cfg.DATASET.HEATMAP_SIZE[2])
            joints_log(pred.reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                       './results/figure/joints',
                       f"pred_joints_{person_id}",
                       example_ct)

            joints_log(target_joints.reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                       './results/figure/joints',
                       f"gt_joints_{person_id}",
                       example_ct)

            threeviews_log(feature, './results/figure/threeviews',
                           f'feature_{person_id}', example_ct)
            threeviews_log(output, './results/figure/threeviews',
                           f'output_{person_id}', example_ct)
            threeviews_log(vol, './results/figure/threeviews',
                           f'volume_{person_id}', example_ct)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_log(cfg, loss, example_ct, epoch):
    if cfg.WANDB:
        wandb.log({"epoch": epoch, "loss": loss}, step=example_ct, commit=True)
    print(f"Loss after  " + str(example_ct).zfill(5) +
          f"  examples:  {loss:.3f} ")
