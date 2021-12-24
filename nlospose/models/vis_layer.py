import torch
import torch.nn as nn 


def normalize(data_bxcxdxhxw):
	b, c, d, h, w = data_bxcxdxhxw.shape
	data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)

	data_min = data_bxcxk.min(2, keepdim = True)[0]
	data_zmean = data_bxcxk - data_min

	data_max = data_zmean.max(2, keepdim = True)[0]
	data_norm = data_zmean / (data_max + 1e-15)

	return data_norm.view(b, c, d, h, w)


class VisibleNet(nn.Module):
	def __init__(self, basedim, layernum=0):
		super().__init__()

		self.layernum = layernum

	def forward(self, x):
		x = nn.ReLU()(x)
		x = normalize(x)
		x = x * 1.e5

		x5 = x

		depdim = x5.shape[2]
		# print(depdim)
		# raw_pred_bxcxhxw, raw_dep_bxcxhxw = x5.max(dim=2)
		raw_pred_bxcxhxw, raw_dep_bxcxhxw = x5.topk(4, dim=2)
		
		raw_dep_bxcxhxw = depdim - 1 - raw_dep_bxcxhxw.float()
		raw_dep_bxcxhxw = raw_dep_bxcxhxw / (depdim - 1)

		xflatdep = torch.cat([raw_pred_bxcxhxw, raw_dep_bxcxhxw], dim=1)

		return xflatdep


if __name__ == '__main__':
	input = torch.ones((2,1,156,256,256))
	model = VisibleNet(basedim=3)
	output = model(input)
	print(f'input shape of VisibleNet is {input.shape} \noutput shape of VisibleNet is {output.shape}')
