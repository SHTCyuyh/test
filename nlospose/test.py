from audioop import avg
import os
import sys
from tarfile import SOLARIS_XHDTYPE
from scipy.sparse import data
import torch
import numpy as np
from torch.tensor import Tensor
print(os.getcwd())
sys.path.append('./')
from nlospose.models.criterion import generate_3d_integral_preds_tensor, softmax_integral_tensor 
from torch import optim, nn
from nlospose.models.config import _C as cfg
from lzma import PRESET_DEFAULT
import torch
import torch.nn as nn
import wandb
from lib.metric import calc_MPJPE
from models.criterion import softmax_integral_tensor
from einops import rearrange
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from models.model_video import Meas2Pose
from nlospose.models.config import _C as cfg


def joints_log(joints, res_path, joint_name, index=0):
    os.makedirs(res_path, exist_ok=True)
    joints_txt_path = os.path.join(res_path, "txt")
    os.makedirs(joints_txt_path, exist_ok=True)
    np.savetxt(joints_txt_path + '/' +  f"{index}_" +joint_name + '.txt', joints)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection="3d")
    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(x_major_locator)
    ax.zaxis.set_major_locator(x_major_locator)

    # joints = (joints / 256.0 - 0.5) * 2.5  # TODO 将具体数字改成congfig中变量

    # xs = []
    # ys = []
    # zs = []

    ds = []
    hs = []
    ws = []

    for i in range(joints.shape[0]):
        # xs.append(joints[i, 0])
        # ys.append(joints[i, 1])
        # zs.append(-joints[i, 2])

        ds.append(joints[i, 0])
        hs.append(joints[i, 1])
        ws.append(joints[i, 2])


    # fig = plt.figure()
    # t = ys
    # ys = zs
    # zs = t
    # ys = ys
    ds, hs, ws = ws, ds, hs
    ax.scatter(ds, hs, ws)
    renderBones(ax, ds, hs, ws)

    ax.set_xlabel('D')
    ax.set_ylabel('H')
    ax.set_zlabel('W')
    ax.set_title(joint_name)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    ax.invert_zaxis()
    # ax.invert_yaxis()
    # ax.invert_xaxis()
    ax.figure.savefig(res_path + '/' + f"{index}_" +joint_name )
    # wandb.log({f"joints of {joint_name}": wandb.Image(ax.figure)}, commit=False) TODO
    plt.clf()

def renderBones(ax, xs, ys, zs):
    link = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8],
        [6, 9],
        [7, 10],
        [8, 11],
        [9, 12],
        [9, 13],
        [9, 14],
        [12, 15],
        [13, 16],
        [14, 17],
        [16, 18],
        [17, 19],
        [18, 20],
        [19, 21],
        [20, 22],
        [21, 23],
    ]
    for l in link:
        index1, index2 = l[0], l[1]
        ax.plot([xs[index1], xs[index2]], [ys[index1], ys[index2]],
                [zs[index1], zs[index2]], linewidth=1, label=r"$x=y=z$")

def eval(cfg, model, eval_dataloader, criterion, epoch):
    model.eval()

    total_MJPJE, total_loss = 0, 0

    with torch.no_grad():
        for step, (images, vol, target_joints, person_id) in enumerate(eval_dataloader):
            images, target_joints = images.to(cfg.DEVICE), target_joints.to(cfg.DEVICE)
            preds = model(images)
            target_joints = target_joints.to(cfg.DEVICE)
            loss = criterion(preds, target_joints)
            total_loss += loss.item()
            for i in range(preds.shape[0]):
                pred1 = preds[i]
                # pred2 = output_after[1]
                joints_log(pred1.reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                           f'./test_results_{cfg.PROJECT_NAME}/figure/joints',
                           f"pred_joints_{person_id[i]}",
                           epoch)
                # joints_log(pred2.reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                #            './results/figure/joints',
                #            f"pred_joints_{person_id[mid][1]}",
                        #    epoch)
                gt1 = target_joints[i]
                # gt2 = mid_joints[1]
                joints_log(gt1.reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                           f'./test_results_{cfg.PROJECT_NAME}/figure/joints',
                           f"gt_joints_{person_id[i]}",
                           epoch)
                print(f'rest {len(eval_dataloader)-step} ')


        total_data = len(eval_dataloader)
        avg_loss = total_loss / total_data
        # avg_MJPJE = total_MJPJE / total_data
        # if cfg.WANDB:
        #     wandb.log({"epoch" : epoch, "eval_avg_loss": avg_loss})
            # wandb.log({"epoch" : epoch, "eval_avg_MJPJE": avg_MJPJE})
    
    return avg_loss

def get_optimizer(cfg, model):
	optimizer = None
	if cfg.TRAIN.OPTIMIZER == 'sgd':
		optimizer = optim.SGD(
			model.parameters(),
			lr = cfg.TRAIN.LR,
			momentum = cfg.TRAIN.MOMENTUM,
			weight_decay = cfg.TRAIN.WD,
			nesterov = cfg.TRAIN.NESTEROV
		)
	elif cfg.TRAIN.OPTIMIZER == 'AdamW':
		optimizer = optim.AdamW(
			model.parameters(),
			lr = cfg.TRAIN.LR
		)
	return optimizer
if __name__ == "__main__":
    from models.dataset import NlosDataset
    from torch.utils.data import DataLoader
    from models.loss import mpjpe, n_mpjpe, p_mpjpe
    data_path = '/data2/og_data/person15'
    eval_dataset = NlosDataset(cfg, datapath=cfg.DATASET.TRAIN_PATH)
    eval_dataloader = DataLoader(
    eval_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
    num_workers=cfg.NUM_WORKERS, pin_memory=False)
    cfg['DEVICE'] = 1
    model = Meas2Pose(cfg)

    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch)
    checkpoint = torch.load("/home/yuyh/network_backbone_posenet3d/checkpoint_new_nlosformer_0810_lct+fusion_3person_train/4.pth")
    model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    model = model.to('cuda:1')
    avg_loss = eval(cfg, model,eval_dataloader,mpjpe,4)
    model



