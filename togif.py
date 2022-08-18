import imageio
import os
import glob


def png2gif(sources, gifname, time):
    # os.chdir(source) # os.chdir()：改变当前工作目录到指定的路径
    file_list = sources # os.listdir()：文件夹中的文件/文件夹的名字列表
    frames = [] #读入缓冲区
    for png in file_list:
        frames.append(imageio.imread(png))
    imageio.mimsave(gifname, frames, 'GIF', duration=time)

path = '/home/yuyh/network_backbone_posenet3d/results_nlos_poseformer_0524_onlymid_person16_newdataset/figure/joints'
gtfiles = glob.glob(path+'/9_gt*.png')
predfiles = glob.glob(path+'/9_pred*.png')
gtfiles = sorted(gtfiles)
gt1 = gtfiles[0:196]
gt2 = gtfiles[197:392]
gt3 = gtfiles[393:589]
gt4 = gtfiles[590:]
predfiles = sorted(predfiles)
# print(gtfiles[-20:])
# 196  392   589
png2gif(gt1, './fig/gt/5-27_person16_gt1gif.gif', 0.05)
png2gif(gt2, './fig/gt/5-27_person16_gt2gif.gif', 0.05)
png2gif(gt3, './fig/gt/5-27_person16_gt3gif.gif', 0.05)
png2gif(gt4, './fig/gt/5-27_person16_gt4gif.gif', 0.05)
# png2gif(predfiles, './fig/pred/5-26-9_person16_predgif.gif', 0.01)
print('done')