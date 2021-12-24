import torch
import wandb
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from torch import optim

from models.nlos_dataloader import NlosDataset
from torch.utils.data import DataLoader


from nlospose.models.config import _C as cfg
from models.model import Meas2Pose
from nlospose.trainer import train
from lib.vis_3view import vis_3view
from torchsummary import summary
from models.criterion import L2JointLocationLoss
# from mayavi import mlab
# mlab.options.offscreen = True


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def make(cfg):
    # train_dataset = NlosPoseDataset(cfg = cfg, datapath = cfg.DATASET.TRAIN_PATH)
    # train_dataloader = DataLoader(train_dataset, batch_size = cfg.TRAIN.BATCH_SIZE, shuffle = True, num_workers = cfg.NUM_WOKERS)
    # data = train_dataloader.next()
    # input = np.ones((512,256,256))
    
    train_dataset = NlosDataset(cfg, datapath = cfg.DATASET.TRAIN_PATH)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers = cfg.NUM_WORKERS, pin_memory=True)
    model = Meas2Pose(cfg).to(cfg.DEVICE) # TODO change fixed parameters to cfg
    # model = model.to(cfg.DEVICE)
    criterion = L2JointLocationLoss(output_3d=True)
    optimizer = optim.Adam(
        model.parameters(),
        lr = cfg.TRAIN.LR
    )
    
    return model, train_dataloader, criterion, optimizer


def eval(cfg, model, eval_loader, criterion):
    model.eval()

    with torch.no_grad():
        
        total, residual = 0, 0
        for step, (input, vol, joints, person_id) in enumerate(eval_loader):
            input = input.to(cfg.DEVICE)
            output,_ = model(input)
            gap = criterion(output, vol.to(cfg.DEVICE))
            total += vol.shape[0]
            residual += gap.sum().item()

        print(f"avg lresidual bewteen output and volume is {residual / total}")

        wandb.log({"eval residual" : residual / total})

    summary(model.unet3d, (1,256,256,256), batch_size=2, device="cuda")

    torch.save(model.stact_dict(), "./checkpoint/evaled.pth")
    wandb.save("./checkpoint/evaled.pth")



            
            



def build_model_and_log(cfg, run):

    model = Meas2Vol(cfg).to(cfg.DEVICE)
    # model.eval()
    # dummy_input = torch.ones((2,1,256,256,256)).to(cfg.DEVICE)
    # torch.onnx.export(model, dummy_input, 'model.onnx', export_params=True, verbose=False,
                    # input_names = ["input0"], output_names = ["output0"])

    # model = MobileFaceNet()
    # model.load_state_dict(torch.load('my_model.pth', map_location='cpu'))
    # model.eval()   先加载模型进入eval()模式
    # dummy_input = torch.randn(1, 3, 112, 112)  # 你模型的输入   NCHW
    # torch.onnx.export(model, dummy_input,'my_model.onnx', export_params=True,verbose=False,
    #                 input_names['input0'],output_names=['output0']) 
    # onnx_model = onnx.load('./my_model.onnx')  # load onnx model
    # onnx.checker.check_model(onnx_model)  # check onnx model
    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

    model_artifact = wandb.Artifact(
        "convnet", type = "model",
        description = "Test Model",
        metadata = dict(cfg)
    )
    model_dir = './'
    path = os.path.join(model_dir, 'model_1.pth')
    torch.save(model.cpu().state_dict(), path)
    wandb.save(path)
    model_artifact.add_file(path)
    
    feature = torch.load("results/obj/feature_2400.pt").cpu().detach().numpy()
    wandb.log({"feature": wandb.Image(feature.sum(0))})

    run.log_artifact(model_artifact)

    


def run():
    if cfg.WANDB:
        wandb.login()
        
        run = wandb.init(project=cfg.PROJECT_NAME, config=dict(cfg), name="meas2vol_testWandb")
        # build_model_and_log(cfg, run)
    
    seed_everything(23333)
    
    model, train_dataloader, criterion, optimizer = make(cfg)

    # eval(cfg, model, train_dataloader, criterion)
    
    # wandb.finish()
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        cfg.TRAIN.LR_STEP,
        cfg.TRAIN.LR_FACTOR,
        last_epoch = -1,
    )
    # print(run)
    # wandb.log_artifact("/home/liuping/data/mini_test/", name='new_artifact', type='my_dataset') 
    for epoch in tqdm(range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH)):
        train(model, train_dataloader, criterion, optimizer, cfg, epoch)
        lr_scheduler.step()
        torch.save(model, f"./checkpoint/{epoch}.pth")
          
        
    if cfg.WANDB:
        wandb.finish()
    
    print("finished")
    
    
if __name__=='__main__':
    run()