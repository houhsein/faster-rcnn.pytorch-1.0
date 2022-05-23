import numpy as np
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions
import torch
import torch.nn as nn
import torch.optim as optim


# Based on monai(0.4.0), include the image mask to label (x1,y1,x2,y2,z1,z2) after all augmentation (rescale and transform)

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

# Dulicated dataset by num_samples
class Dulicated(object):
    '''
    Dulicated data for augmnetation
    '''
    def __init__(self,
                 keys,
                 num_samples: int = 1):
        self.keys = keys
        self.num_samples = num_samples

    def __call__(self, data):
        d = dict(data)
        image = d['image']
        label = d['label']
        results: List[Dict[Hashable, np.ndarray]] = [dict(data) for _ in range(self.num_samples)]
            
        for key in data.keys():            
            for i in range(self.num_samples):
                results[i][key] = data[key]
        return results

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


# inference annotate is only get zero array, im_info and number box
class Annotate_inference(object):
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
        image = d[self.keys[0]]
        annotation = np.zeros((1, 7))
        shape = np.array(image.shape)
        d['im_info'] = np.delete(shape,0,axis=0)
        d['num_box'] = 2 # spleen & background
        return d


#  Weighted_cluster_nms for valid data select, and two function for 

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

def Weighted_cluster_nms(boxes, scores, NMS_threshold=0.7, iou_class="iou"):
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
    if iou_class == "iou":
        iou = calc_iou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
    elif iou_class == "diou":
        iou = calc_diou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
    else:
        print('Only iou and diou can be used!')
    C = iou
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

def resize_box(predict_box, ratio, injury_label):
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
    out_box[:,3]  = center_y + 1/2 * scale_x * ratio
    out_box[:,4]  = center_z - 1/2 * scale_z * ratio
    out_box[:,5]  = center_z + 1/2 * scale_x * ratio
    # 將BBoX的最後一欄 從spleen posibility改成 spleen injury label
    
    out_box[:,6] = injury_label
    #print(predict_box)
    return out_box
