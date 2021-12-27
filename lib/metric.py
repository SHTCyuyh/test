import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss 


def calc_MPJPE(pred_joints, gt_joints):
    fun = nn.MSELoss()
    return fun(pred_joints, gt_joints) * 3


if __name__ == "__main__":
    pred = torch.Tensor([[1,1,1],[2,2,2],[3,3,3]])
    gt = torch.Tensor([[1,0,1],[2,2,2],[3,3,3]])
    print(calc_MPJPE(pred, gt))