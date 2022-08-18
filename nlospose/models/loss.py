# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from tkinter import Y
import torch
import numpy as np

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))


def mse_loss(pre, target):
    out = (pre - target) ** 2
    return out/len(pre)
# class L2JointLocationLoss(nn.Module):
# 	def __init__(self, output_3d, size_average=True, reduce=True):
# 		super(L2JointLocationLoss, self).__init__()
# 		self.size_average = size_average
# 		self.reduce = reduce
# 		self.output_3d = output_3d

# 	def forward(self, preds, *args):
# 		gt_joints = args[0]
# 		gt_joints_vis = args[1]

# 		num_joints = int(gt_joints_vis.shape[1] / 3)
# 		hm_width = preds.shape[-1]
# 		hm_height = preds.shape[-2]
# 		# hm_depth = preds.shape[-3] // num_joints if self.output_3d else 1
# 		hm_depth = preds.shape[-3]

# 		pred_jts = softmax_integral_tensor(preds, num_joints,  hm_width, hm_height, hm_depth)

# 		_assert_no_grad(gt_joints)
# 		_assert_no_grad(gt_joints_vis)
# 		return weighted_mse_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average)


def _assert_no_grad(tensor):
	assert not tensor.requires_grad, \
		"nn criterions don't compute the gradient w.r.t. targets - please " \
		"mark these tensors as not requiring gradients"


def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
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

	accu_x = accu_x * \
			 torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[
				 0]
	accu_y = accu_y * \
			 torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[
				 0]
	accu_z = accu_z * \
			 torch.cuda.comm.broadcast(torch.arange(z_dim).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[
				 0]

	accu_x = accu_x.sum(dim=2, keepdim=True)
	accu_y = accu_y.sum(dim=2, keepdim=True)
	accu_z = accu_z.sum(dim=2, keepdim=True)

	return accu_x, accu_y, accu_z


def softmax_integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth):
	# global soft max
	preds = preds.reshape((preds.shape[0], num_joints, -1))
	preds = F.softmax(preds, 2)

	# integrate heatmap into joint location
	if output_3d:
		x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
	else:
		assert 0, 'Not Implemented!'  # TODO: Not Implemented
	x = x / float(hm_width) - 0.5
	y = y / float(hm_height) - 0.5
	z = z / float(hm_depth) - 0.5
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

if __name__ == '__main__':
    x =np.zeros((2,1,24,3), dtype=np.float32) 
    x = torch.from_numpy(x)
    y =np.zeros((2,1,24,3), dtype=np.float32)
    y = torch.from_numpy(y)
    loss = p_mpjpe(x,y)
    print(loss)