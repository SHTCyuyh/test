import torch
import wandb
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from torch import optim

from models.dataset import NlosDataset
from torch.utils.data import DataLoader


from nlospose.models.config import _C as cfg
from models.model_video import Meas2Pose
# from models.nlosformer import nlosformer
from nlospose.trainer import train
from nlospose.eval import eval
from lib.vis_3view import vis_3view
from torchsummary import summary
from models.loss import mpjpe, n_mpjpe, p_mpjpe
# from mayavi import mlab
# mlab.options.offscreen = True


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make(cfg):

    train_dataset = NlosDataset(cfg, datapath=cfg.DATASET.TRAIN_PATH)
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=False)
    eval_dataset = NlosDataset(cfg, datapath=cfg.DATASET.EVAL_PATH)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS, pin_memory=False)
    # TODO change fixed parameters to cfg
    if cfg.TRAIN.BEGIN_EPOCH != 0: 
        model = Meas2Pose(cfg).to(cfg.DEVICE)
        # model = nlosformer().to(cfg.DEVICE)
        model.load_state_dict(torch.load(f"./checkpoint_{cfg.PROJECT_NAME}/{cfg.TRAIN.BEGIN_EPOCH}.pth"))
        if cfg.WANDB:
            wandb.log({"Train begin ": cfg.TRAIN.BEGIN_EPOCH}, commit=True)
            print(f"Train begin :{cfg.TRAIN.BEGIN_EPOCH} ")
    # model = model.to(cfg.DEVICE)
    else:
        model = Meas2Pose(cfg).to(cfg.DEVICE)
        # model = nlosformer().to(cfg.DEVICE)
    criterion = mpjpe
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.TRAIN.LR,
        weight_decay=0.1
    )

    return model, train_dataloader, criterion, optimizer, eval_dataloader


def run():
    if cfg.WANDB:
        wandb.login()

        wandb.init(project=cfg.PROJECT_NAME,
                   config=dict(cfg),
                   name="v2v_person2_mini")
                   
        # build_model_and_log(cfg, run)

    seed_everything(23333)

    model, train_dataloader, criterion, optimizer, eval_dataloader = make(cfg)
    if cfg.WANDB:
        wandb.watch(model, log='all')

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        cfg.TRAIN.LR_STEP,
        cfg.TRAIN.LR_FACTOR,
        last_epoch=-1,
    )
    # print(run)
    # wandb.log_artifact("/home/liuping/data/mini_test/", name='new_artifact', type='my_dataset')
    best_performance = np.finfo(np.float32).max
    for epoch in tqdm(range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH)):
        train(model, train_dataloader, criterion, optimizer, cfg, epoch)
        lr_scheduler.step()
        if epoch % 1 == 0 or epoch == cfg.TRAIN.END_EPOCH:
            os.makedirs(f'./checkpoint_{cfg.PROJECT_NAME}/', exist_ok=True)
            torch.save(model.state_dict(), f"./checkpoint_{cfg.PROJECT_NAME}/{epoch}.pth")
        performance = eval(cfg, model, eval_dataloader,
                           criterion, epoch)
        if performance < best_performance:
            os.makedirs(f'./results_{cfg.PROJECT_NAME}/trained_models/', exist_ok=True)
            torch.save(
                model.state_dict(), f'./results_{cfg.PROJECT_NAME}/trained_models/{cfg.PROJECT_NAME}.pth')
            best_performance = performance


    if cfg.WANDB:
        wandb.finish()

    print("finished")


if __name__ == '__main__':
    run()
