from lzma import PRESET_DEFAULT
import torch
import torch.nn as nn
import wandb
from lib.metric import calc_MPJPE
from models.criterion import softmax_integral_tensor
from einops import rearrange



def eval(cfg, model, lenth,prefetcher,criterion, epoch):
    model.eval()

    total_MJPJE, total_loss = 0, 0

    with torch.no_grad():
        input, vol, target_joints, person_id = prefetcher.next()
        step = 0
        while input is not None:
            step += 1
            input, vol, target_joints, person_id = prefetcher.next()
            # example_ct = epoch * len(dataloader) + step
            preds = model(input) # out_a: torch.Size([2, 1, 24, 3])  out_b torch.Size([2, 3, 9, 24])
            loss = criterion(preds,target_joints)
            total_loss += loss.item()
        # for step, (images, vol, target_joints, person_id) in enumerate(eval_dataloader):
        #     images, target_joints = images.to(cfg.DEVICE), target_joints.to(cfg.DEVICE)
        #     preds = model(images)
        #     target_joints = target_joints.to(cfg.DEVICE)
        #     loss = criterion(preds, target_joints)
        #     total_loss += loss.item()


        total_data = lenth
        avg_loss = total_loss / total_data
        # avg_MJPJE = total_MJPJE / total_data
        if cfg.WANDB:
            wandb.log({"epoch" : epoch, "eval_avg_loss": avg_loss})
            # wandb.log({"epoch" : epoch, "eval_avg_MJPJE": avg_MJPJE})
    
    return avg_loss

