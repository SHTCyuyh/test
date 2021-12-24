import torch
import torch.nn as nn
import wandb


def eval(cfg, model, eval_dataloader):
    model.eval()

    with torch.no_grad():
        total_MJPJE, total_loss = 0, 0
        for images, labels in eval_dataloader:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item()
            MJPJE = calc_MJPJE(output, labels)
            total_MJPJE += MJPJE

        total_data = len(eval_dataloader)
        avg_loss = total_loss / total_data
        avg_MJPJE = total_MJPJE / total_data
        wandb.log({"eval_avg_loss": avg_loss})
        wandb.log({"eval_avg_MJPJE": avg_MJPJE})
