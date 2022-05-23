import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
from datetime import datetime
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import pandas as pd 
import cv2
import csv
import math
import nibabel as nib
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions
from skimage.transform import resize
import configparser

# Faster RCNN module
sys.path.insert(0, '/tf/jacky831006/faster-rcnn.pytorch-0.4/lib/')
from model.utils.config_3d import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils_3d import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from model.faster_rcnn.resnet_3d import resnet
from model.rpn.bbox_transform_3d import bbox_transform_inv, clip_boxes

# Data augmnetation module (based on MONAI)

from monai.apps import download_and_extract
from monai.data import CacheDataset, DataLoader, Dataset
from monai.config import print_config
from monai.utils import first, set_determinism
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadNifti,
    LoadNiftid,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    CenterSpatialCropd,
    RandCropByPosNegLabeld,
    Resized,
    ToTensord
)

# Need change the config input  
cfgpath ='/tf/jacky831006/faster-rcnn.pytorch-0.4/config/standard_config_new_data_onelabel_2.ini'
# Select the fold of data, if only one fold then fold is 0
fold = 0
print(f'\n Select config:{cfgpath} & fold:{fold}')
conf = configparser.ConfigParser()
conf.read(cfgpath)

# Augmentation
num_samples = conf.getint('Augmentation','num_sample')
size = eval(conf.get('Augmentation','size'))
prob = conf.getfloat('RandAffined','prob')
translate_range = eval(conf.get('RandAffined','translate_range'))
rotate_range = eval(conf.get('RandAffined','rotate_range'))
scale_range = eval(conf.get('RandAffined','scale_range'))

# Data_setting
pre_train = conf.getboolean('Data_Setting','pretrained')
gpu_number = conf.get('Data_Setting','gpu')
seed = conf.getint('Data_Setting','seed')
cross_kfold = conf.getint('Data_Setting','cross_kfold')
epoch = conf.getint('Data_Setting','epoch')
early_stop = conf.getint('Data_Setting','early_stop')
traning_batch_size = conf.getint('Data_Setting','traning_batch_size')
valid_batch_size = conf.getint('Data_Setting','valid_batch_size')
testing_batch_size = conf.getint('Data_Setting','testing_batch_size')
data_split_ratio = eval(conf.get('Data_Setting','data_split_ratio'))
dataloader_num_workers = conf.getint('Data_Setting','dataloader_num_workers')
mGPU = conf.getboolean('Data_Setting','mGPU')
init_lr = conf.getfloat('Data_Setting','init_lr')
optimizer = conf.get('Data_Setting','optimizer')
lr_decay_rate = conf.getfloat('Data_Setting','lr_decay_rate')
lr_decay_epoch = conf.getint('Data_Setting','lr_decay_epoch')

class Annotate(object):
    '''
    transform mask to bounding box label after augmentation
    check the image shape to know scale_x_y, scale_z 
    input: keys=["image", "label"]
    output dictionary add 
        im_info: [x,y,z,scale_x_y,scale_z]
        num_box: 1 (All is one in our data)
    '''
    def __init__(self,keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        #image = d[self.keys[0]]
        #label = d[self.keys[1]]
        image = d['image']
        label = d['label']
        label = label.squeeze(0)
        annotations = np.zeros((1, 7))
        annotation = mask2boundingbox(label)
        if annotation == 0:
            annotation = annotations
            raise ValueError('Dataloader data no annotations')
            #print("Dataloader data no annotations")
        else:
            # add class label
            class_dic = d['all_class']
            cls = d['class']
            annotation = np.array(annotation)
            annotation = np.append(annotation,class_dic[cls])
            annotation = np.expand_dims(annotation,0)
        #print(annotation.shape)
        #print(image.shape)
        d['label'] = annotation
        shape = np.array(image.shape)
        d['im_info'] = np.delete(shape,0,axis=0)
        d['num_box'] = len(class_dic) # all class inculde background
        return d

def mask2boundingbox(label):
    if torch.is_tensor(label):
        label = label.numpy()   
    sk_mask = sk_label(label) 
    regions = sk_regions(label.astype(np.uint8))
    #global top, left, low, bottom, right, height 
    #print(regions)
    # check regions is empty
    if not regions:
        return 0

    for region in regions:
        # print('[INFO]bbox: ', region.bbox)
        # region.bbox (x1,y1,z1,x2,y2,z2)
        # top, left, low, bottom, right, height = region.bbox
        y1, x1, z1, y2, x2, z2 = region.bbox
   # return left, top, right, bottom, low, height
    return x1, y1, x2, y2, z1, z2

    
# data augmentation

inference_transforms = Compose(
    [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Resized(keys=['image', 'label'], spatial_size = size),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        Annotate(keys=["image", "label"]),
        ToTensord(keys=["image", "label"])
    ]
)

def calc_iou(a, b): 
    # a,b (x1,y1,x2,y2,z1,z2)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) * (b[:, 5] - b[:, 4])
    #   area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1) area 不確定要不要+1

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    idp = torch.min(torch.unsqueeze(a[:, 5], dim=1), b[:, 5]) - torch.max(torch.unsqueeze(a[:, 4], 1), b[:, 4])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    idp = torch.clamp(idp, min=0) 

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) * (a[:, 5] - a[:, 4]), dim=1) + area - iw * ih *idp 

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih * idp

    IoU = intersection / ua

    '''
    medicaldetectiontoolkit 的方法
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)
    '''
    return IoU

def calc_diou(bboxes1, bboxes2):
    # 確認bboxes1和bboxes2維度
    # bboxes1 : (N,6)  bboxes2:(M,6)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:#
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True
    # xmin,ymin,xmax,ymax,zmin,zmax -> [:,0],[:,1],[:,2],[:,3],[:,4],[:,5]
    # (N,M)
    w1 = (bboxes1[:, 2] - bboxes1[:, 0]).unsqueeze(1).expand(rows,cols)
    h1 = (bboxes1[:, 3] - bboxes1[:, 1]).unsqueeze(1).expand(rows,cols) 
    d1 = (bboxes1[:, 5] - bboxes1[:, 4]).unsqueeze(1).expand(rows,cols)
    w2 = (bboxes2[:, 2] - bboxes2[:, 0]).unsqueeze(0).expand(rows,cols)
    h2 = (bboxes2[:, 3] - bboxes2[:, 1]).unsqueeze(0).expand(rows,cols)
    d2 = (bboxes2[:, 5] - bboxes2[:, 4]).unsqueeze(0).expand(rows,cols)
    
    area1 = w1 * h1 * d1 # (N,M)
    area2 = w2 * h2 * d2 # (N,M)
    
    # (N,M)
    center_x1 = ((bboxes1[:, 2] + bboxes1[:, 0]) / 2).expand(rows,cols) 
    center_y1 = ((bboxes1[:, 3] + bboxes1[:, 1]) / 2).expand(rows,cols) 
    center_z1 = ((bboxes1[:, 4] + bboxes1[:, 5]) / 2).expand(rows,cols) 
    center_x2 = ((bboxes2[:, 2] + bboxes2[:, 0]) / 2).expand(rows,cols) 
    center_y2 = ((bboxes2[:, 3] + bboxes2[:, 1]) / 2).expand(rows,cols) 
    center_z2 = ((bboxes2[:, 4] + bboxes2[:, 5]) / 2).expand(rows,cols) 
    
    # inter 
    # 避免bboxes1與bboxes2維度不同
    # iw, ih ,idp 維度為(N,M)
    iw = torch.min(torch.unsqueeze(bboxes1[:, 2], dim=1), bboxes2[:, 2]) - torch.max(torch.unsqueeze(bboxes1[:, 0], 1), bboxes2[:, 0])
    ih = torch.min(torch.unsqueeze(bboxes1[:, 3], dim=1), bboxes2[:, 3]) - torch.max(torch.unsqueeze(bboxes1[:, 1], 1), bboxes2[:, 1])
    idp = torch.min(torch.unsqueeze(bboxes1[:, 5], dim=1), bboxes2[:, 5]) - torch.max(torch.unsqueeze(bboxes1[:, 4], 1), bboxes2[:, 4])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    idp = torch.clamp(idp, min=0) 
    inter_area = iw * ih * idp # (N,M)
    
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2 + (center_z2 - center_z1)**2
    #print(f'inter_diag:{inter_diag.size()}')
    #inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:]) 
    #inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2]) 
    #inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    #inter_area = inter[:, 0] * inter[:, 1]
    #inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    
    # Outer
    # 避免bboxes1與bboxes2維度不同
    # ow, oh ,odp 維度為(N,M)
    ow = torch.max(torch.unsqueeze(bboxes1[:, 2], dim=1), bboxes2[:, 2]) - torch.min(torch.unsqueeze(bboxes1[:, 0], 1), bboxes2[:, 0])
    oh = torch.max(torch.unsqueeze(bboxes1[:, 3], dim=1), bboxes2[:, 3]) - torch.min(torch.unsqueeze(bboxes1[:, 1], 1), bboxes2[:, 1])
    odp = torch.max(torch.unsqueeze(bboxes1[:, 5], dim=1), bboxes2[:, 5]) - torch.min(torch.unsqueeze(bboxes1[:, 4], 1), bboxes2[:, 4])
    ow = torch.clamp(ow, min=0)
    oh = torch.clamp(oh, min=0)
    odp = torch.clamp(odp, min=0) 
    #o_max_x = torch.max(bboxes1[:,2],bboxes2[:,2]) # x2
    #o_max_y = torch.max(bboxes1[:,3],bboxes2[:,3]) # y2
    #o_max_z = torch.max(bboxes1[:,5],bboxes2[:,5]) # z2
    #o_min_x = torch.min(bboxes1[:,0],bboxes2[:,0]) # x1
    #o_min_y = torch.min(bboxes1[:,1],bboxes2[:,1]) # y1
    #o_min_z = torch.min(bboxes1[:,4],bboxes2[:,4]) # z1
    
    outer_diag = ow ** 2 + oh ** 2 + odp ** 2 + 1e-7
    #print(f'outer_diag:{outer_diag.size()}')
    #out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:]) 
    #out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])
    #outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    #outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    
    union = area1 + area2 - inter_area + 1e-7
    #print(f'union size: {union.size()}')
    #print(f'test :{((inter_diag) / outer_diag).size()}')
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    return dious

def Weighted_cluster_nms(boxes, scores, NMS_threshold=0.7):
    '''
    Arguments:
        boxes (Tensor[N, 6])
        scores (Tensor[N, 1])
    Returns:
        Fast NMS results
    '''
    #scores, idx = scores.sort(1, descending=True)
    #scores, idx = scores.sort(dim=0,descending=True)
    #print(type(scores))
    #print(scores)
    #scores = scores.unsqueeze(dim=1)
    #print(f'boxes shape:{boxes.shape}') [N, 6]
    #print(f'scores shape:{scores.shape}') [N, 1]
    scores, idx = scores.sort(dim=0,descending=True)
    #print(f'idx:{idx.shape}') #[N, 1] 
    # idx 為排序完的順序為原本的哪個位置
    idx = idx.squeeze(dim=1)
    # print(f'scores shape:{scores.shape}')
    #print(f'boxes:{boxes.shape}')
    boxes = boxes[idx]   # 对框按得分降序排列
    #print(f'boxes shape:{boxes.shape}')
    
    scores = scores
    iou = calc_diou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
    #print(f'iou:{iou}')
    C = iou
    # print(f'C:{C}')
    for i in range(200):    
        A=C
        maxA = A.max(dim=0)[0]   # 列最大值向量
        E = (maxA < NMS_threshold).float().unsqueeze(1).expand_as(A)   # 对角矩阵E的替代
        C = iou.mul(E)     # 按元素相乘
        if A.equal(C)==True:     # 终止条件
            break
    keep = maxA < NMS_threshold  # 列最大值向量，二值化
    # print(f'keep:{keep}')

    n = len(scores)
    weights = (C*(C>NMS_threshold).float() + torch.eye(n).cuda()) * (scores.reshape((1,n)))
    xx1 = boxes[:,0].expand(n,n)
    yy1 = boxes[:,1].expand(n,n)
    xx2 = boxes[:,2].expand(n,n)
    yy2 = boxes[:,3].expand(n,n)
    zz1 = boxes[:,4].expand(n,n)
    zz2 = boxes[:,5].expand(n,n)


    weightsum=weights.sum(dim=1)         # 坐标加权平均
    xx1 = (xx1*weights).sum(dim=1)/(weightsum)
    yy1 = (yy1*weights).sum(dim=1)/(weightsum)
    xx2 = (xx2*weights).sum(dim=1)/(weightsum)
    yy2 = (yy2*weights).sum(dim=1)/(weightsum)
    zz1 = (zz1*weights).sum(dim=1)/(weightsum)
    zz2 = (zz2*weights).sum(dim=1)/(weightsum)

    boxes = torch.stack([xx1, yy1, xx2, yy2, zz1, zz2], 1)
    #print(boxes)
    # torch.IntTensor(keep)
    # 原本為 keep
    return boxes[keep], keep, scores[keep]


def resize_box_ori(predict_box, ratio, injury_label):
    # 等比例放大
    center_x = (predict_box[:,0] + predict_box[:,2]) / 2
    center_y = (predict_box[:,1] + predict_box[:,3]) / 2
    center_z = (predict_box[:,4] + predict_box[:,5]) / 2

    scale_x = predict_box[:,2] - predict_box[:,0]
    scale_y = predict_box[:,3] - predict_box[:,1]
    scale_z = predict_box[:,5] - predict_box[:,4]
    
 
    
    out_box = torch.zeros_like(predict_box)
    out_box[:,0]  = center_x - 1/2 * scale_x * ratio
    out_box[:,1]  = center_y - 1/2 * scale_y * ratio
    out_box[:,2]  = center_x + 1/2 * scale_x * ratio 
    out_box[:,3]  = center_y + 1/2 * scale_y * ratio 
    out_box[:,4]  = center_z - 1/2 * scale_z * ratio
    out_box[:,5]  = center_z + 1/2 * scale_z * ratio 
    # 將BBoX的最後一欄 從spleen posibility改成 spleen injury label
    out_box[:,6] = injury_label
    #print(predict_box)
    return out_box


def resize_box(predict_box, ratio, injury_label):
    # 等比例放大
    center_x = (predict_box[:,0] + predict_box[:,2]) / 2
    center_y = (predict_box[:,1] + predict_box[:,3]) / 2
    center_z = (predict_box[:,4] + predict_box[:,5]) / 2

    scale_x = predict_box[:,2] - predict_box[:,0]
    scale_y = predict_box[:,3] - predict_box[:,1]
    scale_z = predict_box[:,5] - predict_box[:,4]
    
    # Z axis 前面增加(變小) 後面減少(變小)， X axis 右邊增加(變大)
    
    out_box = torch.zeros_like(predict_box)
    out_box[:,0]  = center_x - 1/2 * scale_x * ratio
    out_box[:,1]  = center_y - 1/2 * scale_y * ratio 
    out_box[:,2]  = center_x + 1/2 * scale_x * ratio * 1.3/1.2
    out_box[:,3]  = center_y + 1/2 * scale_y * ratio * 1.4/1.2
    # 若 Z軸本身就取很多了則不增加取的數量 (> 90)
    out_box[:,4]  = center_z - 1/2 * torch.where(scale_z*ratio>=90, scale_z * ratio, scale_z * ratio * 1.4/1.2)
    out_box[:,5]  = center_z + 1/2 * scale_z * ratio * 1.3/1.2
    # 將BBoX的最後一欄 從spleen posibility改成 spleen injury label
    out_box[:,6] = injury_label
    #print(predict_box)
    return out_box

def vis_show(image, label):
    for j in range(label.shape[0]):
        left, bottom, right, top, low, height = int(label[j,0]), int(label[j,1]), int(label[j,2]), int(label[j,3]), int(label[j,4]), int(label[j,5])
        print(left,bottom,right,top,low,height)
        for i in range(height-low+1):
            # low -1 往前一個畫框框
            image_show = image[0,0,:,:,low-1+i]
            fig, (ax1) = plt.subplots(1, 1, figsize = (20, 5))
            #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20, 5))
            print ('Image Phase', low-1+i)
            image_label = image_show.copy()
            #print('Found bbox', prop.bbox)
            cv2.rectangle(image_label, (left, bottom), (right, top), (255, 0, 0), 1)

            ax1.set_title('Image with derived bounding box')

            ax1.imshow(image_show,cmap="gray")
            ax1.imshow(image_label,cmap="gray", alpha=0.4)

            plt.show()

def vis_save(image, label, save_path):
    # 已經篩選到只剩一個box
    left, bottom, right, top, low, height = int(label[0]), int(label[1]), int(label[2]), int(label[3]), int(label[4]), int(label[5])
    #print(left,bottom,right,top,low,height)
        
    for i in range(height-low+1):
        # low -1 往前一個畫框框
        image_show = image[0,0,:,:,low-1+i]
        fig, ax = plt.subplots()
        plt.axis('off')
        label_image = image_show.copy()
        # bounding box 
        cv2.rectangle(label_image, (left, bottom), (right, top), (255, 0, 0), 1)
        # 水平翻轉跟順時鐘旋轉 (原本為RAS)
        image_show = cv2.flip(image_show, 1)
        label_image = cv2.flip(label_image, 1)
        image_show=cv2.rotate(image_show, cv2.cv2.ROTATE_90_CLOCKWISE) 
        label_image=cv2.rotate(label_image, cv2.cv2.ROTATE_90_CLOCKWISE)

        ax.imshow(image_show,cmap="gray")
        ax.imshow(label_image,cmap="gray", alpha=0.4)

        plt.savefig(f"{save_path}/{i:03}.png",bbox_inches='tight',pad_inches = 0)
        # plt.show()
        # 關閉減少記憶體
        plt.close()
    print(f'{save_path} is done!')

# batch size為1的測試
def inference(model, inference_loader, resize_ori, data_dic, class_list, vis):
    # Settings
    model.eval()
    all_boxes = [[[] for _ in range(len(data_dic))]
            for _ in range(len(class_list))]
    overlap_list = [[[[] for _ in range(6)] for _ in range(len(data_dic))] for _ in range(len(class_list))]
    
    #df_urls = pd.DataFrame(columns = ['newID', 'file', 'Path', "BBox", "Posibility", "Gap_with_GT", "Aug_Shape"])
    df_urls = pd.DataFrame(columns = ['ID', 'file', 'Path', "BBox", "Posibility", "Aug_Shape", "Gap_with_GT", "DIOU", "Box_cover"])
    # Test validation data
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        i = -1
        
        for test_data in inference_loader:
            i += 1
            #print(f'start {i}')
            # initilize the tensor holder here.
            num_boxes = torch.LongTensor(1).cuda()
            gt_boxes = torch.FloatTensor(1).cuda()
            gt_boxes.resize_(1, 1, 7).zero_()
            num_boxes.resize_(1).zero_()
            
            # Faster RCNN predict
            im_data, gt_boxes, im_info, num_boxes  = test_data['image'].cuda(), test_data['label'].cuda(), \
                                                     test_data['im_info'], test_data['num_box'].cuda()
            ori_shape = test_data['image_meta_dict']['spatial_shape'].cuda()
            #png_file_name = test_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].replace(".nii.gz","")
            
            file_name = test_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]

            
            '''
            # old formate
            # 要根據file path格式決定ID
            if 'classification_negative_data' in test_data['image_meta_dict']['filename_or_obj'][0]:
                ID = test_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('@')[0]
            elif 'SDY' in test_data['image_meta_dict']['filename_or_obj'][0]:
                ID = test_data['image_meta_dict']['filename_or_obj'][0].split('/')[-3]
            else:
                ID = test_data['image_meta_dict']['filename_or_obj'][0].split('/')[-2]
            ''' 
            ID = test_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('@')[0] 
            path = test_data['image_meta_dict']['filename_or_obj'][0]
            det_tic = time.time()
            #val_outputs = model(val_images)
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = model(im_data, im_info, gt_boxes, num_boxes)
            
            scores = cls_prob.data
            boxes = rois.data[:, :, 1:7]

            if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                    #if args.class_agnostic:
                    #    box_deltas = box_deltas.view(-1, 6) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    #               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    #    box_deltas = box_deltas.view(1, -1, 6)
                    #else:
                    box_deltas = box_deltas.view(-1, 6) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(inference_loader.batch_size, -1, 6 * 2)
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
                pred_boxes = _.cuda() # if args.cuda > 0 else _

            # 將predict box resize回原本大小(經過augmentation後的) 
            # X座標對應Y軸 Y座標對應X軸
            # 0-5是bg 6-11是spleen
            '''
            resize_size = test_data['foreground_start_coord']
            resize_image = test_data['foreground_end_coord']
            resize_x = int(resize_image[0]-resize_size[0])
            resize_y = int(resize_image[1]-resize_size[1])
            resize_z = int(resize_image[2]-resize_size[2])
            scale_y = 128 / resize_x
            scale_x = 128 / resize_y
            scale_z = 128 / resize_z
            
            pred_boxes[:,:,6] = pred_boxes[:,:,6] / scale_x
            pred_boxes[:,:,7] = pred_boxes[:,:,7] / scale_y
            pred_boxes[:,:,8] = pred_boxes[:,:,8] / scale_x
            pred_boxes[:,:,9] = pred_boxes[:,:,9] / scale_y
            pred_boxes[:,:,10] = pred_boxes[:,:,10] / scale_z
            pred_boxes[:,:,11] = pred_boxes[:,:,11] / scale_z
            '''
            #print(f'After resize:{pred_boxes}')
            #print(pred_boxes.shape)
            #print(pred_boxes[0,0,:])
            scores = scores.squeeze()
            
            pred_boxes = pred_boxes.squeeze()
            
            # Caculate gt box and box IOU
            gt_boxes = gt_boxes.squeeze(0)
            #print(gt_boxes)
            injury_label = int(gt_boxes[0][6])
            
            det_toc = time.time()
            detect_time = det_toc - det_tic
            #print(f'detect_time:{detect_time}')
            misc_tic = time.time()
            # check output is same as label if same return ture else false
            #if vis:
            #    im = cv2.imread(imdb.image_path_at(i))
            #    im2show = np.copy(im)
            #print(f'pred_boxes last:{pred_boxes}')
            for j in range(1, 2):
                inds = torch.nonzero(scores[:,j]>thresh, as_tuple=False).view(-1)
                # if there is det
                if inds.numel() > 0:  
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    #if args.class_agnostic:
                    #    cls_boxes = pred_boxes[inds, :]
                    #else:
                    # 0:6 是 bg
                    cls_boxes = pred_boxes[inds][:, j * 6:(j + 1) * 6]
                    
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    #keep = nms(cls_dets, cfg.TEST.NMS)
                    #print(f'cls_score:{cls_dets[:,-1].shape}')
                    anchorBoxes_f, keep, scores_f = Weighted_cluster_nms(cls_dets[:,:6], cls_dets[:,-1].view(-1,1), 0.5)
                    cls_dets_last = torch.cat((anchorBoxes_f, scores_f), 1)
                    
                    # 計算原始的diou
                    diou = calc_diou(gt_boxes[:,:6].float(),cls_dets_last[:,:6])

                    # 將predict box resize至overlap gt box , ratio為放大比例
                    if resize_ori: 
                        overlap_box = resize_box_ori(cls_dets_last[:,:7], ratio=ratio, injury_label=injury_label)
                    else:
                        overlap_box = resize_box(cls_dets_last[:,:7], ratio=ratio, injury_label=injury_label)
                    posibility = cls_dets_last[:,6].tolist()

                    # 確保 resize 的 box 在 image 範圍內
                    #print(f'Before crop:{overlap_box}')
                    # data_check size is 128,128,128
                    resize_x,resize_y,resize_z = 128,128,128
                    overlap_box = torch.clamp(overlap_box, min=0.0)
                    overlap_box[:,2] = torch.clamp(overlap_box[:,2], max=resize_x)
                    overlap_box[:,3] = torch.clamp(overlap_box[:,3], max=resize_y)
                    overlap_box[:,5] = torch.clamp(overlap_box[:,5], max=resize_z)
                    #print(f'After crop:{overlap_box}')
                    gt_boxes = gt_boxes.to(torch.int)
                    overlap_box = overlap_box.to(torch.int)
                    overlap_list[j][i][0] = (gt_boxes[:,0] - overlap_box[:,0]).tolist()
                    overlap_list[j][i][1] = (gt_boxes[:,1] - overlap_box[:,1]).tolist()
                    overlap_list[j][i][4] = (gt_boxes[:,4] - overlap_box[:,4]).tolist() 
                    overlap_list[j][i][2] = (overlap_box[:,2] - gt_boxes[:,2]).tolist()
                    overlap_list[j][i][3] = (overlap_box[:,3] - gt_boxes[:,3]).tolist()
                    overlap_list[j][i][5] = (overlap_box[:,5] - gt_boxes[:,5]).tolist() 
                    
                    
                    if vis: 
                        final_image = resize(im_data.cpu().numpy(),[1,1,resize_x,resize_y,resize_z]) 
                        vis_show(final_image, overlap_box[:,:6].cpu().numpy())
                    '''
                    # file path 
                    gt_path = f'/tf/jacky831006/faster-rcnn.pytorch-0.4/object_test/{png_file_name}/GT'
                    if not os.path.isdir(gt_path):
                        os.makedirs(gt_path)
                    label_path = f'/tf/jacky831006/faster-rcnn.pytorch-0.4/object_test/{png_file_name}/label'
                    if not os.path.isdir(label_path):
                        os.makedirs(label_path)
                    vis_detections(im_data.cpu().numpy(), gt_boxes[:,:6].cpu().numpy(), gt_path)
                    vis_detections(im_data.cpu().numpy(), cls_dets_last[:,:6].cpu().numpy(), label_path)
                    '''
                    # TP (1,num)
                    # AP 計算的方法在只有一個gt box時相對沒有意義
                    #TP = iou.ge(0.5).cpu().numpy()
                    #FP = iou.le(0.5).cpu().numpy()
                    #Precision = TP.sum()/ TP.sum() + FP.sum()
                    #print(f"TP,FP:{(TP.shape,FP.shape)}")        
                    #all_boxes[j][i] = overlap_box.cpu().numpy()
                    
                else:
                    overlap_box = [0]*6
                    overlap_box.append(injury_label)
                    posibility = 0
                    
                    # all_boxes[j][i] = empty_array
            # tensor to list
            overlap_box = overlap_box.tolist()
            # reshape the overlap_list [t1,t2],[t1,t2] -> [t1,t1] , [t2,t2]
            overlap_list_out = np.array(overlap_list[1][i]).T.tolist()
            print(overlap_list_out)
            # 計算resize後的bounding box是否完全覆蓋label

            # 取Posibility最高的結果
            box_cover = [ True if i >= 0 else False for i in overlap_list_out[0]]
            misc_toc = time.time()
            nms_time = misc_toc - misc_tic
            # Colname 'newID', 'file', 'Path', "BBox", "posibility", "Gap_with_GT" ,"Aug_Shape"
            #df_series = pd.Series([newID, file_name, path, overlap_box, posibility, overlap_list_out, (resize_x,resize_y,resize_z)], index=df_urls.columns)
            df_series = pd.Series([ID, file_name, path, overlap_box, posibility, (resize_x,resize_y,resize_z), overlap_list_out, diou, all(box_cover)], index=df_urls.columns) 
            df_urls = df_urls.append(df_series, ignore_index=True)
            
    return df_urls


# Running 
# valid or test 
valid = False
test_df = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_test_20220310.csv')
valid_df = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_valid_20220310.csv')
all_cls = ['__background__','spleen']
all_cls_label = [i for i in range(len(all_cls))]
all_cls_dic = dict(zip(all_cls,all_cls_label))


def data_progress(df, dicts):
    dicts = []
    for index, row in df.iterrows():
    
        if row['spleen_injury'] == 0:
            image = row['source']
            label = row['source'].replace('image','label')
        else:
            image = f'/tf/jacky831006/object_detect_data_new/pos/image/{row["chartNo"]}@venous_phase.nii.gz'
            label = f'/tf/jacky831006/object_detect_data_new/pos/label/{row["chartNo"]}@venous_phase.nii.gz'
        cls = 'spleen'
        all_cls = all_cls_dic
        dicts.append({'image':image,'label':label,'class':cls,'all_class':all_cls})
    return dicts

test_data_dicts = data_progress(test_df,'test_data_dicts')
valid_data_dicts = data_progress(valid_df,'valid_data_dicts')

# set augmentation seed 
set_determinism(seed=0)
if valid:
    inference_ds = CacheDataset(data=valid_data_dicts, transform=inference_transforms, cache_rate=1.0, num_workers=4)
else:
    inference_ds = CacheDataset(data=test_data_dicts, transform=inference_transforms, cache_rate=1.0, num_workers=4)

inference_data = DataLoader(inference_ds, batch_size=1, num_workers=4)

fasterRCNN = resnet(all_cls, 101, pretrained=False, class_agnostic=False)
fasterRCNN.create_architecture()
fasterRCNN.cuda()
# multiple GPU setting
mGPUs = False
#mGPUs = True
#fasterRCNN = nn.DataParallel(fasterRCNN)
#all_boxes = [[[] for _ in xrange(num_images)]
#               for _ in xrange(imdb.num_classes)]

init_lr = 0.0001
optimizer = torch.optim.Adam(fasterRCNN.parameters(), init_lr)
#if vis:
#    thresh = 0.05
#else:
thresh = 0.0
max_per_image = 10

data_file = eval(conf.get('Data output','data file name'))[fold]
data_diou = eval(conf.get('Data output','best valid diou'))[fold]

data_weight = f'/tf/jacky831006/faster-rcnn.pytorch-0.4/training_checkpoints/{data_file}/{data_diou}.pth'

fasterRCNN.load_state_dict(torch.load(data_weight))
thresh = 0.0
max_per_image = 10
# resize ratio
ratio = 1.2
resize_ori = False

#(model, inference_loader, resize_ori, data_dic, class_list, vis)
if valid:
    df_out = inference(fasterRCNN, inference_data , resize_ori, valid_data_dicts, all_cls, vis = False)
else:
    df_out = inference(fasterRCNN, inference_data , resize_ori, test_data_dicts, all_cls, vis = False)

# Merge all data
if valid:
    valid_pos = valid_df[valid_df.spleen_injury == 1].iloc[:,4:]
    valid_pos.chartNo = valid_pos.chartNo.astype('str')
    all_pos = pd.merge(valid_pos, df_out, left_on='chartNo',right_on = 'ID')

    all_neg = pd.merge( valid_df[valid_df.spleen_injury == 0].iloc[:,4:], df_out, left_on='source',right_on = 'Path')
    df_out_new = pd.concat([all_pos,all_neg])
else:
    test_pos = test_df[test_df.spleen_injury == 1].iloc[:,4:]
    test_pos.chartNo = test_pos.chartNo.astype('str')
    all_pos = pd.merge(test_pos, df_out, left_on='chartNo',right_on = 'ID')

    all_neg = pd.merge( test_df[test_df.spleen_injury == 0].iloc[:,4:], df_out, left_on='source',right_on = 'Path')
    df_out_new = pd.concat([all_pos,all_neg])

# Select non cover data
# x,y 誤差小於3即可，z軸誤差要小於1 
df_out_false = df_out_new[df_out_new.Box_cover == False]

error_list = []
for index , row in df_out_false.iterrows():
    # 選取最大值的DIOU
    ind = int(torch.argmax(row['DIOU']))
    error_list.append([1 if i>=0 else 0 for i in row['Gap_with_GT'][ind]])

pos_error = []
for li in error_list:
    pos_error.extend([i for i, x in enumerate(li) if x == 0])

pos_error_total = []
for i in range(6):
    pos_error_total.append(pos_error.count(i))

print(f'\n Non cover position x1,y1,x2,y2,z1,z2:{pos_error_total}')

print(f'False label, na is negative: \n{df_out_false.spleen_injury_grade.value_counts(dropna=False)}')

# Show high, midle, low diou bounding box image (each level show most 5)
new_diou = []
for i in df_out_new.DIOU:
    ind = int(torch.argmax(i))
    new_diou.append(float(i[0][ind]))

df_out_new['new_diou'] = np.array(new_diou)

df_sel = df_out_new[df_out_new.new_diou >=0.5].sort_values(by=['new_diou'])
midle_level = df_sel[:5]
high_level = df_sel[-5:]
low_level = df_out_new.sort_values(by=['new_diou'])[:5]

def plot_diou(level, transform):
    inference_data_dict = data_progress(eval(f'{level}_level'),'inference_data_dict')

    # set augmentation seed 
    set_determinism(seed=0)

    inference_ds = CacheDataset(data=inference_data_dict, transform=transform, cache_rate=1.0, num_workers=4)
    inference_data = DataLoader(inference_ds, batch_size=1, num_workers=4)

    for data in inference_data:
        file_name = data['image_meta_dict']['filename_or_obj'][0]
        newID = file_name.split('/')[-1].split('@')[0]
        file_dir = f'/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_{ratio}_{data_weight.split("/")[-2]}_{data_weight.split("/")[-1][:6]}/{level}/{newID}/'
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        vis_save(data['image'].numpy(), data['label'].cpu().numpy()[0][0],file_dir)

diou_list= ['high','midle','low']
for i in diou_list:
    plot_diou(i,inference_transforms)


# Save bouding box data
if valid:
    if resize_ori:
        output_file_name = f'bouding_box_{ratio}_{data_weight.split("/")[-2]}_{data_weight.split("/")[-1][:6]}_new_data_valid_compare_gt.csv'
    else:
        output_file_name = f'bouding_box_{ratio}_{data_weight.split("/")[-2]}_{data_weight.split("/")[-1][:6]}_resize_new_data_valid_compare_gt.csv'
else:
    if resize_ori:
        output_file_name = f'bouding_box_{ratio}_{data_weight.split("/")[-2]}_{data_weight.split("/")[-1][:6]}_new_data_test_compare_gt.csv'
    else:
        output_file_name = f'bouding_box_{ratio}_{data_weight.split("/")[-2]}_{data_weight.split("/")[-1][:6]}_resize_new_data_test_compare_gt.csv'

df_out_new.to_csv(f'/tf/jacky831006/faster-rcnn.pytorch-0.4/{output_file_name}')

print('All is done!')