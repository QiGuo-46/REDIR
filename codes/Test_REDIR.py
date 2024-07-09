from __future__ import print_function
import sys

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose GPU
import torch

import torch.nn as nn
import cv2
import numpy as np
from utils import crop, refocus, calculate_APSE, mkdir, load_model
from Networks.Hybrid import HybridNet
from Networks.networks import STN_UNetGenerator
from Event_Dataset import TestSet_AutoRefocus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(opt):
    mkdir(os.path.join(opt.save_path, "Test"))
    mkdir(os.path.join(opt.save_path, "True"))

    ## load reconNet, use HybridNet here
    model_1 = nn.DataParallel(STN_UNetGenerator(60, 60, 3))  # ResUnetAttentionGenerator  UnetGenerator
    model_1.load_state_dict(
        torch.load(opt.registrationNet, map_location={"cuda:0": "cpu"}), False
    )
    model_1.to(device)
    model_1 = model_1.eval()

    model_2 = nn.DataParallel(HybridNet())
    model_2.load_state_dict(torch.load(opt.reconstructNet, map_location={"cuda:0": "cpu"}), False)
    model_2.to(device)
    model_2 = model_2.eval()

    ## prepare dataset
    testDataset = TestSet_AutoRefocus(opt.input_path)
    testLoader = torch.utils.data.DataLoader(
        testDataset, batch_size=1, shuffle=False, num_workers=1
    )

    f = open(opt.save_path + "APSE.txt", "w")  # open a txt to save APSE results
    APSEs = []
    with torch.no_grad():
        for i, (eData, diff_t, depth, fx, occ_free_aps) in enumerate(testLoader):
            print("Processing data %d ..." % i)
            eData = eData.to(device)

            eData = crop(eData, roiTL=(2, 45), size=(256, 256))  # crop roi
            # registration
            reconInp = model_1(eData)
            b, s, c, h, w = eData.shape
            reconInp = reconInp.view(b, s, c, h, w)

            timeWin = reconInp.shape[1]
            outputs = model_2(reconInp, timeWin)
            print(outputs)

            name = os.path.join(opt.save_path, "Test", "%04d" % i + ".png")
            img = (outputs[0, :].permute(1, 2, 0) * 255).cpu().numpy()
            cv2.imwrite(name, img)

            name = os.path.join(opt.save_path, "True", "%04d" % i + ".png")

            # occ_free_aps = occ_free_aps.squeeze().cpu().numpy()
            occ_free_aps = (
                crop(occ_free_aps, roiTL=(2, 45), size=(256, 256)).squeeze().numpy()
            )
            cv2.imwrite(name, occ_free_aps)

    f.write("Mean APSE = %.5f " % (np.array(APSEs).mean()))
    f.close()
    print("Completed !")


if __name__ == "__main__":
    ## parameters
    parser = argparse.ArgumentParser(description="Test REDIR-Net")
    parser.add_argument(
        "--reconstructNet",
        default=" ",
        help="pre-trained model to use as starting point",
    )
    parser.add_argument(
        "--registrationNet",
        default=" ",
        help="pre-trained model to use as starting point",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./Example_data/Processed-unf/",
        help="input data path",
    )
    parser.add_argument(
        "--save_path", type=str, default="./Results-unf/", help="saving path"
    )

    opt = parser.parse_args()
    test(opt)

