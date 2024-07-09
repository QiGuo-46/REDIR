import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import math
import numpy as np
from utils import get_file_name, filter_events_by_key
import cv2
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch
import random


def pack_with_manual_refocus(opt, event_path, save_name):
    """
    This function is used to manually refocus events and pack them to event frames
    
    Parameters:
        opt: basic parameters
        event_path: path of the target event data (in the form of 'npy')
        aps_path: path of the corresponding ground truth image
        save_name: saving path of the processed data    去掉后缀npy的文件名
    """
    
    ## load raw data and initialize parameters ------
    data = np.load(event_path,allow_pickle=True).item()
    eventData = data.get('events')     #  事件数据 unfocused events
    reference_time = data.get('occ_free_aps_ts')  # 参考帧 timestamp at the reference camera pose, which is equivalent to the timestamp of occlusion-free APS
    fx = data.get('fx') # 焦距 parameter from camera intrinsic matrix
    d = data.get('depth') # 深度 depth
    v = data.get('v') # 相机速度 camera speed (m/s)
    img_size = data.get('size') # 相机尺寸 image size
    occ_free_aps = data.get('occ_free_aps') # 图片帧 occlusion-free aps
    
    ## manual event refocusing ------手工聚焦
    x = eventData['x'].astype(float)
    y = eventData['y'].astype(float)
    t = eventData['t'].astype(float)
    p = eventData['p'].astype(float)
    shift_x = np.round((t - reference_time) * fx * v / d)#四舍五入到整数
    valid_ind = (x+shift_x >= 0) * (x+shift_x < img_size[1]) #有效数据对应x的索引
    x[valid_ind] += shift_x[valid_ind]
    
    ## pack events to event frames ------
    minT = t.min()
    maxT = t.max()
    t -= minT
    interval = (maxT - minT) / opt.time_step#time_step表示将事件分为多少份 
                                            #interval 每一份多少时间
    # filter events
    Xrange = (opt.roiTL[1], opt.roiTL[1] + opt.roi_size[1]) # 感兴趣区域
    Yrange = (opt.roiTL[0], opt.roiTL[0] + opt.roi_size[0]) #图片x、y的范围
    x,y,t,p = filter_events_by_key(x,y,t,p, Xrange[0], Xrange[1])   #过滤
    y,x,t,p = filter_events_by_key(y,x,t,p, Yrange[0], Yrange[1])
    
    # convert events to event frames
    pos = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    neg = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    T,H,W = pos.shape
    pos = pos.ravel()#展平为一维数组
    neg = neg.ravel()
    ind = (t / interval).astype(int)#时间完全归一化到[0,N]范围
    ind[ind == T] -= 1  #这步冗余
    x = (x - Xrange[0]).astype(int) #X,Y归一化
    y = (y - Yrange[0]).astype(int)
    pos_ind = p == 1    #正负事件的索引
    neg_ind = p == 0
    
    np.add.at(pos, x[pos_ind] + y[pos_ind]*W + ind[pos_ind]*W*H, 1) #正负事件赋值
    np.add.at(neg, x[neg_ind] + y[neg_ind]*W + ind[neg_ind]*W*H, 1)
    pos = np.reshape(pos, (T,H,W))
    neg = np.reshape(neg, (T,H,W))

    # show data

    for i in range(30):
        cv2.imshow("pos", pos[i, :, :])
        key = cv2.waitKey(0)

    for i in range(30):
        cv2.imshow("neg", neg[i, :, :])
        key = cv2.waitKey(0)

    print(np.min(pos[1, :, :]), np.max(pos[1, :, :]))
    print(np.min(neg[1, :, :]), np.max(neg[1, :, :]))
    data = pos * 32 - neg * 32 + 128
    print(np.min(data[1, :, :]), np.max(data[1, :, :]))
    cm = plt.colormaps['jet']
    plt.imshow(data[1, :, :], cmap=cm)
    plt.colorbar()
    plt.show()
                
    # crop occlusion-free aps according to roi 
    occ_free_aps = occ_free_aps[Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
    
    ## save processed data -------       
    processed_data = dict()
    processed_data['Pos'] = pos.astype(np.int16) # positive event frame 
    processed_data['Neg'] = neg.astype(np.int16) # negative event frame
    processed_data['size'] = opt.roi_size # frame size
    processed_data['Inter'] = interval # time interval of one event frame
    processed_data['depth'] = d # target depth
    processed_data['occ_free_aps'] = occ_free_aps # croped occlusion-free aps
    np.save(opt.save_path+save_name+'.npy', processed_data)
    
    
def pack_without_refocus(opt, event_path, save_name):
    """
    This function is used to pack unfocused events to event frames
    
    Parameters:
        opt: basic parameters
        event_path: path of the target event data (in the form of 'npy')
        aps_path: path of the corresponding ground truth image
        save_name: saving path of the processed data
    """
    
    ## load raw data and initialize parameters ------
    data = np.load(event_path,allow_pickle=True).item()
    eventData = data.get('events')  # unfocused events
    d = data.get('depth') # depth
    fx = data.get('fx') # parameter from camera intrinsic matrix
    ref_t = data.get('occ_free_aps_ts') # timestamp at the reference camera pose, which is equivalent to the timestamp of occlusion-free APS
    occ_free_aps = data.get('occ_free_aps') # occlusion-free aps

    ## pack events ------
    x = eventData['x']
    y = eventData['y']
    t = eventData['t']
    p = eventData['p']
    pos = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    neg = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    minT = t.min()
    maxT = t.max()
    ref_t -= minT
    t -= minT
    interval = (maxT - minT) / opt.time_step
    index_t = []
    for i in range(opt.time_step):
        index_t.append(i*interval)
    index_t = np.array(index_t)
    
    # filter events
    Xrange = (opt.roiTL[1], opt.roiTL[1] + opt.roi_size[1])
    Yrange = (opt.roiTL[0], opt.roiTL[0] + opt.roi_size[0])
    x,y,t,p = filter_events_by_key(x,y,t,p, Xrange[0], Xrange[1])
    y,x,t,p = filter_events_by_key(y,x,t,p, Yrange[0], Yrange[1])
    
    # convert events to event frames
    pos = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    neg = np.zeros((opt.time_step, opt.roi_size[0], opt.roi_size[1]))
    T,H,W = pos.shape
    pos = pos.ravel()
    neg = neg.ravel()
    ind = (t / interval).astype(int)
    ind[ind == T] -= 1
    x = (x - Xrange[0]).astype(int)
    y = (y - Yrange[0]).astype(int)
    pos_ind = p == 1
    neg_ind = p == 0
    np.add.at(pos, x[pos_ind] + y[pos_ind]*W + ind[pos_ind]*W*H, 1)
    np.add.at(neg, x[neg_ind] + y[neg_ind]*W + ind[neg_ind]*W*H, 1)
    pos = np.reshape(pos, (T,H,W))
    neg = np.reshape(neg, (T,H,W))

    #Translate transformation
    Lhor= random.uniform(-0.005, 0.005)
    Lver= random.uniform(-0.005, 0.005)
    pos1 = torch.from_numpy(pos).float()
    Translate_pos1 = np.zeros(pos1.shape)
    neg1 = torch.from_numpy(neg).float()
    Translate_neg1 = np.zeros(neg1.shape)
    occ_free_aps1=torch.from_numpy(occ_free_aps).float()
    for i in range(30):
        theta = torch.Tensor([[1, 0, i * Lhor], [0, 1, i * Lver]]).unsqueeze(0)
        grid = F.affine_grid(theta, pos1[i, :, :].unsqueeze(0).unsqueeze(0).size())
        Translate_pos1[i] = F.grid_sample(pos1[i, :, :].unsqueeze(0).unsqueeze(0), grid)
        grid = F.affine_grid(theta, neg1[i, :, :].unsqueeze(0).unsqueeze(0).size())
        Translate_neg1[i] = F.grid_sample(neg1[i, :, :].unsqueeze(0).unsqueeze(0), grid)

    theta = torch.Tensor([[1, 0, 15 * Lhor], [0, 1, 15*Lver]]).unsqueeze(0)
    grid = F.affine_grid(theta, occ_free_aps1.unsqueeze(0).unsqueeze(0).size())
    Translate_occ_free_aps1 = F.grid_sample(occ_free_aps1.unsqueeze(0).unsqueeze(0), grid, padding_mode='reflection')

    #Scale transformation
    Fs=random.uniform(-0.001, 0.001)
    pos2 = torch.from_numpy(pos).float()
    Translate_pos2 = np.zeros(pos2.shape)
    neg2 = torch.from_numpy(neg).float()
    Translate_neg2 = np.zeros(neg2.shape)
    occ_free_aps2=torch.from_numpy(occ_free_aps).float()
    for i in range(30):
        theta = torch.Tensor([[1+i*Fs, 0, 0], [0, 1+i*Fs, 0]]).unsqueeze(0)
        grid = F.affine_grid(theta, pos2[i, :, :].unsqueeze(0).unsqueeze(0).size())
        Translate_pos2[i] = F.grid_sample(pos2[i, :, :].unsqueeze(0).unsqueeze(0), grid)
        grid = F.affine_grid(theta, neg2[i, :, :].unsqueeze(0).unsqueeze(0).size())
        Translate_neg2[i] = F.grid_sample(neg2[i, :, :].unsqueeze(0).unsqueeze(0), grid)

    theta = torch.Tensor([[1+15*Fs, 0, 0], [0, 1+15*Fs, 0]]).unsqueeze(0)
    grid = F.affine_grid(theta, occ_free_aps2.unsqueeze(0).unsqueeze(0).size())
    Translate_occ_free_aps2 = F.grid_sample(occ_free_aps2.unsqueeze(0).unsqueeze(0), grid, padding_mode='reflection')



    #Rotate transformation
    theta0 = random.uniform(-30, 30)
    pos3 = torch.from_numpy(pos).float()
    Translate_pos3 = np.zeros(pos3.shape)
    neg3 = torch.from_numpy(neg).float()
    Translate_neg3 = np.zeros(neg3.shape)
    occ_free_aps3 = torch.from_numpy(occ_free_aps).float()
    for i in range(30):
        theta = torch.Tensor([[math.cos(-10/theta0/180*math.pi*i) , math.sin(-10/theta0/180*math.pi*i), 0], [-math.sin(-10/theta0/180*math.pi*i), math.cos(-10/theta0/180*math.pi*i) , 0]]).unsqueeze(0)
        grid = F.affine_grid(theta, pos3[i, :, :].unsqueeze(0).unsqueeze(0).size())
        Translate_pos3[i] = F.grid_sample(pos3[i, :, :].unsqueeze(0).unsqueeze(0), grid)
        grid = F.affine_grid(theta, neg3[i, :, :].unsqueeze(0).unsqueeze(0).size())
        Translate_neg3[i] = F.grid_sample(neg3[i, :, :].unsqueeze(0).unsqueeze(0), grid)

    theta = torch.Tensor([[math.cos(-10/theta0/180*math.pi*15) , math.sin(-10/theta0/180*math.pi*15), 0], [-math.sin(-10/theta0/180*math.pi*15), math.cos(-10/theta0/180*math.pi*15) , 0]]).unsqueeze(0)
    grid = F.affine_grid(theta, occ_free_aps3.unsqueeze(0).unsqueeze(0).size())
    Translate_occ_free_aps3 = F.grid_sample(occ_free_aps3.unsqueeze(0).unsqueeze(0), grid, padding_mode='reflection')


    # crop occlusion-free aps according to roi 
    occ_free_aps = occ_free_aps[Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
    Translate_occ_free_aps1 = Translate_occ_free_aps1[Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
    Translate_occ_free_aps2 = Translate_occ_free_aps2[Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
    Translate_occ_free_aps3 = Translate_occ_free_aps3[Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]

    Translate_occ_free_aps1=torch.squeeze(torch.squeeze(Translate_occ_free_aps1, 0), 0).numpy().astype(np.uint8)
    Translate_occ_free_aps2=torch.squeeze(torch.squeeze(Translate_occ_free_aps2, 0), 0).numpy().astype(np.uint8)
    Translate_occ_free_aps3=torch.squeeze(torch.squeeze(Translate_occ_free_aps3, 0), 0).numpy().astype(np.uint8)

    ## save event data -------
    processed_data = dict()
    processed_data['Pos'] = pos.astype(np.int16) # positive event frame 
    processed_data['Neg'] = neg.astype(np.int16) # negative event frame
    processed_data['size'] = opt.roi_size # frame size
    processed_data['depth'] = d # target depth
    processed_data['Inter'] = interval # time interval of one event frame
    processed_data['ref_t'] = ref_t # timestamp of reference camera position
    processed_data['fx'] = fx # camera intrinsic parameter
    processed_data['roiTL'] = opt.roiTL # top-left coordinate of roi
    processed_data['index_t'] = index_t # timestamp of each event frame
    processed_data['occ_free_aps'] = occ_free_aps # croped occlusion-free aps
    np.save(opt.save_path+save_name+'.npy', processed_data)

    processed_data = dict()
    processed_data['Pos'] = Translate_pos1.astype(np.int16)  # positive event frame
    processed_data['Neg'] = Translate_neg1.astype(np.int16)  # negative event frame
    processed_data['size'] = opt.roi_size  # frame size
    processed_data['depth'] = d  # target depth
    processed_data['Inter'] = interval  # time interval of one event frame
    processed_data['ref_t'] = ref_t  # timestamp of reference camera position
    processed_data['fx'] = fx  # camera intrinsic parameter
    processed_data['roiTL'] = opt.roiTL  # top-left coordinate of roi
    processed_data['index_t'] = index_t  # timestamp of each event frame
    processed_data['occ_free_aps'] = Translate_occ_free_aps1  # croped occlusion-free aps
    np.save(opt.save_path + save_name + '_T1.npy', processed_data)

    processed_data = dict()
    processed_data['Pos'] = Translate_pos2.astype(np.int16)  # positive event frame
    processed_data['Neg'] = Translate_neg2.astype(np.int16)  # negative event frame
    processed_data['size'] = opt.roi_size  # frame size
    processed_data['depth'] = d  # target depth
    processed_data['Inter'] = interval  # time interval of one event frame
    processed_data['ref_t'] = ref_t  # timestamp of reference camera position
    processed_data['fx'] = fx  # camera intrinsic parameter
    processed_data['roiTL'] = opt.roiTL  # top-left coordinate of roi
    processed_data['index_t'] = index_t  # timestamp of each event frame
    processed_data['occ_free_aps'] = Translate_occ_free_aps2  # croped occlusion-free aps
    np.save(opt.save_path + save_name + '_T2.npy', processed_data)

    processed_data = dict()
    processed_data['Pos'] = Translate_pos3.astype(np.int16)  # positive event frame
    processed_data['Neg'] = Translate_neg3.astype(np.int16)  # negative event frame
    processed_data['size'] = opt.roi_size  # frame size
    processed_data['depth'] = d  # target depth
    processed_data['Inter'] = interval  # time interval of one event frame
    processed_data['ref_t'] = ref_t  # timestamp of reference camera position
    processed_data['fx'] = fx  # camera intrinsic parameter
    processed_data['roiTL'] = opt.roiTL  # top-left coordinate of roi
    processed_data['index_t'] = index_t  # timestamp of each event frame
    processed_data['occ_free_aps'] = Translate_occ_free_aps3  # croped occlusion-free aps
    np.save(opt.save_path + save_name + '_T3.npy', processed_data)



if __name__ == '__main__':
    ## parameters
    parser = argparse.ArgumentParser(description="preprocess data + event refocusing")
    parser.add_argument("--input_path", type=str, default="./Example_data/Raw/Event/", help="input data path")
    parser.add_argument("--save_path", type=str, default="./Example_data/Processed/Event/", help="saving data path")
    parser.add_argument("--time_step", type=int, default=30, help="the number of event frames ('N' in paper)")
    parser.add_argument("--roiTL", type=tuple, default=(0,0), help="coordinate of the top-left pixel in roi")
    parser.add_argument("--roi_size", type=tuple, default=(260,346), help="size of roi")
    opt = parser.parse_args()
    
    ## obtain input file names 
    input_data_name = get_file_name(opt.input_path,'.npy')  #返回列表：包含文件名
    input_data_name.sort()
    
    ## create directory
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    
    ## preprocessing
    for i in range(len(input_data_name)):
        print('processing event ' + input_data_name[i] + '...')
        current_data_path = os.path.join(opt.input_path, input_data_name[i])
        save_name = input_data_name[i][:-4]
        pack_without_refocus(opt, current_data_path, save_name)
    print('Completed!')



"""
python codes/Preprocess_Transformation.py --input_path=E:/SAIDataset/PackedData/Outdoor/Scene/Train --save_path=E:/V-ESAI_Dataset/Outdoor/Scene/Train/Train

"""
