import torch
import torch.cuda.comm
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

import numpy as np

    
class L2JointLocationLoss(nn.Module):
    def __init__(self, output_3d, size_average=True, reduce=True):
        super(L2JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.output_3d = output_3d

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        num_joints = int(gt_joints_vis.shape[1] / 3)
        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        # hm_depth = preds.shape[-3] // num_joints if self.output_3d else 1
        hm_depth = preds.shape[-3]

        pred_jts = softmax_integral_tensor(preds, num_joints, self.output_3d, hm_width, hm_height, hm_depth,)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_l1_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average)


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim): # (d h w)
    assert isinstance(heatmaps, torch.Tensor)
    '''
    Parameter
    heatmaps: probility of location
    -----------------------------------
    Return 
    accu_x: mean location of x label

    '''

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)  # (1, 24, 5, 5, 5)
    accu_x = accu_x.sum(dim=2)
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(z_dim).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    return accu_z, accu_y, accu_x


def softmax_integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)
    # fn = nn.LogSoftmax(dim=2)
    # preds = fn(preds)
    # print(preds)

    # integrate heatmap into joint location
    if output_3d:
        x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
    else:
        assert 0, 'Not Implemented!'  # TODO: Not Implemented
    # x = x / float(hm_width) - 0.5
    # y = y / float(hm_height) - 0.5
    # z = z / float(hm_depth) - 0.5
    # writer.add_scalar('x0 location', x[0, 0, 0], global_step=global_iter_num)
    # writer.add_scalar('x10 location', x[0, 10, 0], global_step=global_iter_num)
    # writer.add_scalar('y0 location', y[0, 0, 0], global_step=global_iter_num)
    # writer.add_scalar('y10 location', y[0, 10, 0], global_step=global_iter_num)
    # writer.add_scalar('z0 location', z[0, 0, 0], global_step=global_iter_num)
    # writer.add_scalar('z10 location', z[0, 10, 0], global_step=global_iter_num)
    preds = torch.cat((x, y, z), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 3))
    return preds


def weighted_mse_loss(input, target, weights, size_average):
    out = (input - target) ** 2
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()


def weighted_l1_loss(input, target, weights, size_average):
    out = torch.abs(input - target)
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()




if __name__ == '__main__':

    test_L2JointLocationLoss = True # 3D Heatmap

    if test_L2JointLocationLoss:

        batch_size = 1
        num_joints = 24
        z_dim = 5
        y_dim = 5
        x_dim = 5

        input = torch.zeros(batch_size, num_joints, z_dim, y_dim, x_dim).cuda() - 1000
        # input[0, 0, 2, 2, 2] = 1.
        for idx in range(num_joints):
            input[0,idx,0,0,0] = 1 if idx != 0 else -1000
        input[0, 0, 1, 1, 1] = 1.

        print(input.shape)
        # test_a = generate_3d_integral_preds_tensor(input, num_joints, z_dim, y_dim, x_dim)
        # test_b = softmax_integral_tensor(input, 24, True, 5, 5, 5)

        Loss = L2JointLocationLoss(output_3d=True)
        gt_joints = torch.zeros(batch_size, num_joints, 3).cuda()
        # gt_joints[0, 1, 0] = 1
        gt_joints[0,0] = torch.Tensor([1,1,1])
        gt_joints_vis = torch.ones_like(gt_joints).cuda()
        # gt_joints = rearrange(gt_joints, 'b n d -> b (n d)')
        gt_joints = gt_joints.reshape((batch_size, num_joints * 3))
        gt_joints_vis = rearrange(gt_joints_vis, 'b n d -> b (n d)')
        print(gt_joints[0, 0])
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('./log')

        loss = Loss(input, gt_joints, gt_joints_vis, writer, 1)
        print(f'loss is {loss.item()}')
        # intput : [batch_size, num_joints, z_length, y_length, x_length]
        # gt_joints : [batch_size, num_joints * 3]  belong to [-0.5, 0.5]
        # gt_joints_vis : [batch_size, num_joints_vis]
        # writer : tensorboard SummaryWriter
        # global_iter_num : int

        print('finished')