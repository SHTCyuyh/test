import torch
import torch.nn as nn
import wandb
from lib.metric import calc_MPJPE
from models.criterion import softmax_integral_tensor
from einops import rearrange


def eval(cfg, model, eval_dataloader, criterion, epoch):
    model.eval()

    total_MJPJE, total_loss = 0, 0

    with torch.no_grad():
        for step, (images, vol, labels, person_id) in enumerate(eval_dataloader):
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            output, feature = model(images)
            pred = softmax_integral_tensor(output, cfg.DATASET.NUM_JOINTS, True,
                                           cfg.DATASET.HEATMAP_SIZE[0], cfg.DATASET.HEATMAP_SIZE[1], cfg.DATASET.HEATMAP_SIZE[2])
            labels = rearrange(labels, 'b n d -> b (n d)').to(cfg.DEVICE)
            target_weights = torch.ones_like(labels).to(cfg.DEVICE)
            loss = criterion(output, labels, target_weights)
            total_loss += loss.item()
            MJPJE = calc_MPJPE(pred, labels)
            total_MJPJE += MJPJE

        total_data = len(eval_dataloader)
        avg_loss = total_loss / total_data
        avg_MJPJE = total_MJPJE / total_data
        wandb.log({"epoch" : epoch, "eval_avg_loss": avg_loss})
        wandb.log({"epoch" : epoch, "eval_avg_MJPJE": avg_MJPJE})
    
    return total_MJPJE

