import os
import torch
import numpy as np
import datetime
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import vgg16
from torch.nn import functional as F


def mkdir(path):
    """
    This function is used to create directory

    Parameters:
        path: path of the desired directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_file_name(path, suffix):
    """
    This function is used to get file name with specific suffix

    Parameters:
        path: path of the parent directory
        suffix: specific suffix (in the form like '.png')
    """
    name_list = []
    file_list = os.listdir(path)
    for i in file_list:
        if os.path.splitext(i)[1] == suffix:
            name_list.append(i)
    name_list.sort()
    return name_list


def filter_events_by_key(key, x1, x2, x3, start, end):
    """
    This function is used to filter events by the key dimension (start inclusive and end exclusive)
    e.g., new_x,new_y,new_t,new_p = filter_events_by_key(x, y, t, p, start=0, end=128)
    returns the filted events with 0 <= x < 128

    Parameters:
        key: path of the parent directory
        suffix: specific suffix (in the form like '.png')#后缀
    """
    new_x1 = x1[key >= start]
    new_x2 = x2[key >= start]
    new_x3 = x3[key >= start]
    new_key = key[key >= start]

    new_x1 = new_x1[new_key < end]
    new_x2 = new_x2[new_key < end]
    new_x3 = new_x3[new_key < end]
    new_key = new_key[new_key < end]

    return new_key, new_x1, new_x2, new_x3


def crop(data, roiTL=(2, 45), size=(256, 256)):
    """
    This function is used to crop the region of interest (roi) from event frames or aps images

    Parameters:
        data: input data (either event frames or aps images)
        roiTL: coordinate of the top-left pixel in roi
        size: expected size of roi
    """
    Xrange = (roiTL[1], roiTL[1] + size[1])
    Yrange = (roiTL[0], roiTL[0] + size[0])
    if data.ndim == 2:
        out = data[Yrange[0] : Yrange[1], Xrange[0] : Xrange[1]]
    elif data.ndim == 3:
        out = data[:, Yrange[0] : Yrange[1], Xrange[0] : Xrange[1]]
    elif data.ndim == 4:
        out = data[:, :, Yrange[0] : Yrange[1], Xrange[0] : Xrange[1]]
    else:
        out = data[:, :, :, Yrange[0] : Yrange[1], Xrange[0] : Xrange[1]]
    return out


def refocus(data, psi, diff_t):
    """
    This function is used to refocus events with the predicted parameter psi

    Parameters:
        data: input unfocused event frames
        psi: refocusing parameter predicted by RefocusNet
        diff_t: time difference between the timestamps of event frames and the reference time
    """

    diff_t = diff_t.to(psi.device)
    refocused_data = torch.zeros_like(data).to(data.device)
    for i in range(data.shape[0]):  # batch
        current_diff_t = diff_t[i, :]  # (30)
        current_psi = psi[i, :]  # (2)
        theta = torch.zeros((data.shape[1], 2, 3), dtype=torch.float).to(
            psi.device
        )  # (step,2,3)  #仿射矩阵
        """
        [[1, 0, x_shift],
        [0, 1, y_shift]]
        """
        theta[:, 0, 0] = theta[:, 1, 1] = 1  # no zoom in/out
        theta[:, 0, 2] = current_psi[0] * current_diff_t  # x_shift
        theta[:, 1, 2] = current_psi[1] * current_diff_t  # y_shift
        #print(theta.shape)
        #print(data.shape)
        #print( data[i, :].shape)
        #print(theta,current_diff_t)
        grid = F.affine_grid(theta, data[i, :].squeeze().size())
        refocused_data[i, :] = F.grid_sample(data[i, :].squeeze(), grid)

    return refocused_data

def registration(data,psi):
    registrated_data = torch.zeros_like(data).to(data.device)
    for i in range(data.shape[0]):
        current_psi=psi[i,:,:]
        theta= torch.zeros((data.shape[1], 2, 3), dtype=torch.float).to(
            psi.device)
        theta[:, 0, 0] = theta[:, 1, 1] = 1  # no zoom in/out
        #print(current_psi[0,:].shape)
        theta[:, 0, 2] = current_psi[0,:]   # x_shift
        theta[:, 1, 2] = current_psi[1,:]   # y_shift
        grid = F.affine_grid(theta, data[i, :].squeeze().size())
        registrated_data[i, :] = F.grid_sample(data[i, :].squeeze(), grid)

    return registrated_data

def calculate_APSE(psi, diff_t, depth, width=346, v=0.177, fx=320.132621):
    """
    This function is used to calculate APSE in the horizontal direction

    Parameters:
        psi: refocusing parameter predicted by RefocusNet
        diff_t: time difference between the timestamps of event frames and the reference time
        depth: ground truth depth
        width: width of the event frame
        v: camera moving speed in the horizontal direction
        fx: parameter from the camera intrinsic matrix
    """

    psi_x = psi.squeeze()[0]
    pred_depth = (2 * fx * v) / (-1 * psi_x * width)

    mean_abs_diff_t = torch.mean(torch.abs(diff_t))
    mean_pix_shift_real = mean_abs_diff_t * fx * v / depth
    mean_pix_shift_pred = mean_abs_diff_t * fx * v / pred_depth
    APSE = np.abs(mean_pix_shift_real - mean_pix_shift_pred)

    return APSE


def load_model(model_dir, model, device):
    """
    Load model from file.
    :param model_dir: model directory
    :param model: instance of the model class to be loaded
    :param device: model device
    :return loaded model
    """
    if os.path.isfile(model_dir):  # 传入的model_dir是文件
        model.load_state_dict(torch.load(model_dir, map_location=device), strict=False)
        model.to(device)
        print("Model restored from " + model_dir + "\n")

    elif os.path.isdir(model_dir):  # 传入的model_dir是文件夹
        model_name = model_dir + model.__class__.__name__

        extensions = [
            ".pt",
            ".pth.tar",
            ".pwf",
            "_weights_min.pwf",
        ]  # backwards compatibility
        for ext in extensions:
            if os.path.isfile(model_name + ext):
                model_name += ext
                break

        if os.path.isfile(model_name):
            model_loaded = torch.load(model_name, map_location=device)
            if "state_dict" in model_loaded.keys():
                model_loaded = model_loaded["state_dict"]
            model.load_state_dict(model_loaded)
            print("Model restored from " + model_name + "\n")
        else:
            print("No model found at" + model_name + "\n")
    else:
        initialize_weights(model)
        print("新建初始化模型:{}".format(model.__class__.__name__))
    return model


def create_model_dir(path_models):
    """
    Create directory for storing model parameters.
    :param path_models: path in which the model should be stored
    :return path to generated model directory
    """

    now = datetime.datetime.now()

    path_models += "model_"
    path_models += "%04d_%02d_%02d" % (now.year, now.month, now.day)
    path_models += "_%02d_%02d_" % (now.hour, now.minute)
    path_models += "/"
    if not os.path.exists(path_models):
        os.makedirs(path_models)
    print("Weights stored at " + path_models + "\n")
    return path_models


def save_model(path_models, model):
    """
    Overwrite previously saved model with new parameters.
    :param path_models: model directory
    :param model: instance of the model class to be saved
    """
    model_name = path_models + model.__class__.__name__ + ".pt"
    torch.save(model.state_dict(), model_name)


def save_models(path_models, models):
    """
    Overwrite previously saved model with new parameters.
    :param path_models: model directory
    :param model: instance of the model class to be saved
    """
    for model in models:

        model_name = path_models + model.__class__.__name__ + ".pt"
        torch.save(model.state_dict(), model_name)


def vgg16_loss(feature_module, loss_func, outputs, occ_free_aps):
    out = feature_module(outputs)
    out_ = feature_module(occ_free_aps)
    loss = loss_func(out, out_)
    return loss


# 获取指定的特征提取模块
def get_feature_module(layer_index, device=None):
    vgg = vgg16(pretrained=True, progress=True).features
    vgg.eval()

    # 冻结参数
    for parm in vgg.parameters():
        parm.requires_grad = False

    vgg = vgg.to(device)
    feature_module = vgg[0 : layer_index + 1]
    feature_module.to(device)
    return feature_module


def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
