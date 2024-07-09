from __future__ import print_function
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose GPU
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from loss.loss import EventWarping
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import (
    load_model,
    create_model_dir,
    save_model,
    crop,
    registration,
    initialize_weights,
)
from dataloader import load_auto
from Networks.Hybrid import HybridNet
from Networks.Registration import RegistrationNet
from Networks.networks import UnetGenerator,UnetGenerator,ResUnetAttentionGenerator,MultiUnetGenerator,MultiResUnetAttentionGenerator,STN_UNetGenerator  #UnetGenerator ResUnetAttentionGenerator
from torch.optim import Adam


import torch.autograd as autograd
autograd.set_detect_anomaly(True)


def train(args):
    if not os.path.exists(args.path_models):
        os.makedirs(args.path_models)
    # 加载配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 损失函数
    loss_function = EventWarping(args, device)

    # 加载模型
    model_1 = nn.DataParallel(STN_UNetGenerator(60,60,3)).to(device)  #UnetGenerator ResUnetAttentionGenerator
    model_1 = load_model(args.registrationNet, model_1, device)
    model_1.train()

    model_2 = nn.DataParallel(HybridNet()).to(device)
    model_2 = load_model(args.reconstructNet, model_2, device)
    model_2.eval() #model_2.eval()  #model_2.train()

    print(model_1)
    print(model_2)
    # 训练模型的保存路径
    path_models = create_model_dir(args.path_models)


    # 数据加载器
    data = []
    for path in args.input_path:
        data.append(load_auto(path))
    data = ConcatDataset(data)


    dataloader = torch.utils.data.DataLoader(
        data, batch_size=args.batchsize, shuffle=False, num_workers=1
    )
    # 定义优化器
    optimizer = args.optimizer(model_1.parameters(), lr=args.lr)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)

    # 初始常数

    n_epochs = args.epochs
    B_per = 1
    B_pix = 32
    B_tv = 2.0e-4
    best_loss = 1.0e6

    Total_loss=torch.zeros(n_epochs)


    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0
        with tqdm(
            enumerate(dataloader), total=len(dataloader), ncols=100
        ) as loop:
            for i, (eData, diff_t, depth, fx, occ_free_aps) in loop:
                """
                eData_shape (step, channel, H, W)  (30,2,260,346)
                occ_free_aps_shape  (260, 346)
                """

                # crop data
                eData = eData.to(device)
                occ_free_aps = (
                    crop(occ_free_aps.unsqueeze(1), roiTL=(2, 45), size=(256, 256))
                    / 255
                )
                occ_free_aps = torch.tensor(occ_free_aps, dtype=torch.float32).to(
                    device
                )
                eData = crop(eData, roiTL=(2, 45), size=(256, 256))  # crop rois
                # registration
                reconInp = model_1(eData)

                b, s, c, h, w = eData.shape
                reconInp = reconInp.view(b, s ,c, h, w)


                #reconInp = crop(reconInp, roiTL=(2, 45), size=(256, 256))  # crop roi
                # forward pass
                time_win = reconInp.shape[1]
                outputs = model_2(reconInp, time_win)

                # loss and backward pass

                loss_smooth = loss_function.smooth(outputs).to(device)
                loss_pix = loss_function.pix(outputs, occ_free_aps).to(device)
                loss_per = loss_function.per(outputs, occ_free_aps).to(device)
                loss_template = B_tv * loss_smooth + B_pix * loss_pix + B_per * loss_per



                train_loss = train_loss + loss_template.item()
                loss_template.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_template = 0
                # 保存最优模型
                with torch.no_grad():
                    if train_loss/ (i + 1) < best_loss:
                        save_model(path_models, model_1)
                        best_loss = train_loss/ (i + 1)
                Total_loss[epoch] = train_loss / (i + 1)
                print(Total_loss[epoch])
            # print training info
                loop.set_description(f"Epoch: {epoch+1}/{n_epochs}") #,defaults:{optimizer.defaults}")
                loop.set_postfix(loss=train_loss / (i + 1), best_loss=best_loss)


        print("lr=",optimizer.param_groups[0]['lr'])
        #  保存最优模型
        # with torch.no_grad():
        #     if train_loss / (i + 1) < best_loss:
        #         save_model(path_models, model_1)
        #         best_loss = train_loss / (i + 1)
        #
        # # print training info
        # loop.set_description(f"Epoch: {epoch+1}/{n_epochs}")
        # loop.set_postfix(loss=train_loss / (i + 1), best_loss=best_loss)
        # print(f"Epoch: {epoch+1}/{n_epochs},loss:{train_loss}")
        scheduler.step()
    print("模型保存路径：", path_models)
    epochs = list(range(1, len(Total_loss) + 1))
    print(Total_loss)

    # 绘制折线图
    plt.plot(epochs, Total_loss, label='Training Loss')

    # 添加标题和标签
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()


if __name__ == "__main__":
    ## parameters
    parser = argparse.ArgumentParser(description="train REDIR-Net ")
    parser.add_argument(
        "--path_models", default="./train_models/", help="save_model_path"
    )
    parser.add_argument(
        "-rc",
        "--reconstructNet",
        default="./PreTraining/Hybrid.pth",
        help="pre-trained model to use as starting point",
    )
    parser.add_argument(
        "-rf",
        "--registrationNet",
        default="./train_models-US/model_2023_11_16_21_17_/DataParallel.pt",
        help="pre-trained model to use as starting point",
    )
    parser.add_argument(
        "-i", "--input_path", nargs="+", default="./Example_data/Processed-V/Train"
    )
    parser.add_argument("--batchsize", default=1)
    parser.add_argument("--optimizer", default=Adam)
    parser.add_argument("--lr", default=0.0003) #0.0003
    parser.add_argument("--epochs", default=800)

    args = parser.parse_args()
    ## train
    train(args)

