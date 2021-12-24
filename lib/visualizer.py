import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import tqdm
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import wandb


def volume_log(volume, res_path, name, index):
    volume_path = res_path + '/volume' + f'/{name}_{index}.pt'
    figure_path = res_path + '/figure' + f'/{name}_{index}.jpg'
    torch.save(volume[0,0], volume_path)
    im = volume[0,0].sum(0).detach().cpu().numpy()
    plt.imshow(im)
    wandb.log({f"projection of {name}": plt})
    plt.savefig(figure_path)


def joints_log(joints, path, joint_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x_major_locator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(x_major_locator)
    ax.zaxis.set_major_locator(x_major_locator)

    joints = (joints / 64.0 - 0.5) * 2.5 # TODO 将具体数字改成congfig中变量

    xs = []
    ys = []
    zs = []

    for i in range(joints.shape[0]):
        xs.append(joints[i, 0])
        ys.append(joints[i, 1])
        zs.append(-joints[i, 2])

    fig = plt.figure()
    t = ys
    ys = zs
    zs = t
    # ys = ys
    renderBones(ax, xs, ys, zs)
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(joint_name)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)
    ax.set_zlim(-0.75, 0.75)
    # ax.invert_zaxis()
    ax.invert_yaxis()
    # ax.invert_xaxis()
    ax.figure.savefig(path + joint_name)
    


def renderBones(ax, xs, ys, zs):
    link = [
             [ 0, 1 ],
             [ 0, 2 ],
             [ 0, 3 ],
             [ 1, 4 ],
             [ 2, 5 ],
             [ 3, 6 ],
             [ 4, 7 ],
             [ 5, 8 ],
             [ 6, 9 ],
             [ 7, 10],
             [ 8, 11],
             [ 9, 12],
             [ 9, 13],
             [ 9, 14],
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


def threeviews_log(re, path, name):
    volumn_MxNxN = re.detach().cpu().numpy()[0,-1]

    # get rid of bad points
    # if name == 'output':
    #     zdim = volumn_MxNxN.shape[0]
    # elif name == 'volume':
    zdim = volumn_MxNxN.shape[0]
    volumn_MxNxN = volumn_MxNxN[:zdim]
    print('volumn min, %f' % volumn_MxNxN.min())
    print('volumn max, %f' % volumn_MxNxN.max())
    # volumn_MxNxN[:5] = 0
    # volumn_MxNxN[-5:] = 0

    volumn_MxNxN[volumn_MxNxN < 0] = 0
    front_view = np.max(volumn_MxNxN, axis=0)
    plt.imshow(front_view / np.max(front_view))
    path = Path(path)
    Path.mkdir(path, exist_ok=True)
    plt.savefig(path + f'/{name}_front_view.jpg')

    top_view = np.max(volumn_MxNxN, axis=1)
    plt.imshow(np.rot90(top_view / np.max(top_view), 2))
    plt.savefig(path + f'/{name}_top_view.jpg')

    left_view = np.max(volumn_MxNxN, axis=2)
    plt.imshow(np.rot90(left_view / np.max(left_view), 3))
    plt.savefig(path + f'/{name}_left_view.jpg')


if __name__ == "__main__":
    res_path = Path("joint")
    file_list = []
    for fileName in res_path.iterdir():
        file_list.append(fileName)
        joints = np.loadtxt(fileName)
        joint_name = fileName.stem

        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x_major_locator = MultipleLocator(0.1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(x_major_locator)
        ax.zaxis.set_major_locator(x_major_locator)

        joints = (joints / 64.0 - 0.5) * 2.5

        xs = []
        ys = []
        zs = []

        for i in range(joints.shape[0]):
            xs.append(joints[i, 0])
            ys.append(joints[i, 1])
            zs.append(-joints[i, 2])

        fig = plt.figure()
        t = ys
        ys = zs
        zs = t
        # ys = ys
        renderBones(ax, xs, ys, zs)
        ax.scatter(xs, ys, zs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(joint_name)
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.75, 0.75)
        ax.set_zlim(-0.75, 0.75)
        # ax.invert_zaxis()
        ax.invert_yaxis()
        # ax.invert_xaxis()
        ax.figure.savefig('./results/joint_fig/' + joint_name)
    print("QwQ")


