import os
import sys
import torch
import numpy as np
from torch.tensor import Tensor
print(os.getcwd())
sys.path.append('./')
from nlospose.models.criterion import generate_3d_integral_preds_tensor, softmax_integral_tensor
from lib.visualizer import volume_log

def test_vol_axis(vol : torch.Tensor) -> torch.Tensor:
    path = './debug'
    os.makedirs(path, exist_ok=True)
    im = vol.sum(0)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plt.imshow(im)
    # plt.imshow(im)
    # plt.xlabel("X")
    # plt.ylabel("Y") 
    # filled = np.ones((100,100,100))
    # x, y, z = 0, 50, 80
    ax.voxels(vol)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    plt.savefig(path + '/vol.jpg')

    fig = plt.figure()
    plt.imshow(im)
    plt.xlabel("X")
    plt.ylabel("Y") 
    plt.show()
    plt.savefig(path+'/im_0.jpg')

    im = vol.sum(1)
    plt.imshow(im)
    plt.xlabel("X")
    plt.ylabel("Y") 
    plt.show()
    plt.savefig(path+'/im_1.jpg')

    im = vol.sum(2)
    plt.imshow(im)
    plt.xlabel("X")
    plt.ylabel("Y") 
    plt.show()
    plt.savefig(path+'/im_2.jpg')


if __name__ == "__main__":
    test_generate_3d_integral = True
    if test_generate_3d_integral:
        heatmap = torch.zeros(2, 24, 64, 64, 64).to("cuda")
        heatmap.fill_(torch.finfo(torch.float32).min)
        heatmap[:, :, 1, 25, 47] = 1
        preds = softmax_integral_tensor(heatmap, 24, True, 64, 64, 64)
        # generate_3d_integral_preds_tensor(heatmap, 24, 100, 100, 100)
    else:
        vol = torch.zeros(100,100,100)
        x = 0
        y = 60
        z = 90
        vol[x:x+2, y:y+2, z:z+2] += 100
        j = np.loadtxt('./1.txt')
        fig_3d = plt.figure(0)
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        fig_2d = plt.figure(1)
        ax_2d = fig_2d.add_subplot(111)

        ax_2d.set_xlim(0, 64)
        ax_2d.set_ylim(0, 64)
        ax_2d.set_aspect(1)

        ax_3d.set_xlim(0, 64)
        ax_3d.set_ylim(0, 64)
        ax_3d.set_zlim(0, 64)
        for i in range(24):
            x = j[i, 0]
            y = j[i, 1]
            z = j[i, 2]
            print(f"{i} : ({x}, {y})")

            ax_3d.scatter(x, y, z)
            ax_2d.scatter(x, y)
            ax_2d.set_xlabel("X")
            ax_2d.set_ylabel("Y")
            # ax_2d.set_oringin("lower")
            ax_3d.figure.savefig('./debug/joints.jpg')
            ax_2d.figure.savefig('./debug/joints_2d.jpg')
        # print(test_vol_axis(vol))

    print("finish")



