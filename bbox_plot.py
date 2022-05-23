import pandas as pd
import numpy as np
import torch
import os

from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2

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

def vis_save_middle(image, label, save_path, id, predict):
    if predict:
        # Select first label
        label =label[0]
    else:
        label = label
    left, bottom, right, top, low, height = int(label[0]), int(label[1]), int(label[2]), int(label[3]), int(label[4]), int(label[5])
    print(left,bottom,right,top,low,height)
    medle_z =  int((low+height)/2-1)  
    # low -1 往前一個畫框框
    image_show = image[0,0,:,:,medle_z]
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

    plt.savefig(f'{save_path}/{id}.png', bbox_inches='tight',pad_inches = 0)
    # plt.show()
    # 關閉減少記憶體
    plt.close()
    print(f'{save_path}/{id}.png is done!')


def vis_save(image, label, save_path, predict):
    if predict:
        # Select first label
        label = label[0]
    else:
        label = label
    left, bottom, right, top, low, height = int(label[0]), int(label[1]), int(label[2]), int(label[3]), int(label[4]), int(label[5])
    print(left,bottom,right,top,low,height)
        
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

def vis_save_percen(image, label, save_path, predict):
    if predict:
        # Select first label
        label =label[0]
    else:
        label = label
    left, bottom, right, top, low, height = int(label[0]), int(label[1]), int(label[2]), int(label[3]), int(label[4]), int(label[5])
    print(left,bottom,right,top,low,height)
    if low == 0:
        low = 1
    percen_list = [low-1, int((low*3+height)/4)-1, int((low+height)/2)-1, int((low+height*3)/4)-1, height-1]
    for i in range(len(percen_list)):
        # low -1 往前一個畫框框
        image_show = image[0,0,:,:,percen_list[i]]
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

        plt.savefig(f"{save_path}/{i:01}.png", bbox_inches='tight',pad_inches = 0)
        # plt.show()
        # 關閉減少記憶體
        plt.close()
        print(f'{save_path} is done!')

def data_progress(df, dicts, predict):
    dicts = []
    if predict:
        for index, row in df.iterrows():
            if row['spleen_injury'] == 0:
                image = row['source']
                bbox = row['BBox']
            else:
                image = f'/tf/jacky831006/object_detect_data_new/pos/image/{row["chartNo"]}@venous_phase.nii.gz'
                bbox = row['BBox']
            cls = 'spleen'
            all_cls = all_cls_dic
            dicts.append({'image':image,'bbox':bbox,'class':cls,'all_class':all_cls})
    else:    
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

class Annotate_predict(object):
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
        shape = np.array(image.shape)
        d['im_info'] = np.delete(shape,0,axis=0)
        d['num_box'] = len(all_cls_dic) # all class inculde background
        return d

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

def plot_png(df, transform, predict, type, data_type):
    inference_data_dict = data_progress(df,'inference_data_dict', predict)

    # set augmentation seed 
    set_determinism(seed=0)
    inference_ds = CacheDataset(data=inference_data_dict, transform=transform, cache_rate=1.0, num_workers=4)
    inference_data = DataLoader(inference_ds, batch_size=1, num_workers=4)

    for data in inference_data:
        file_name = data['image_meta_dict']['filename_or_obj'][0]
        newID = file_name.split('/')[-1].split('@')[0]

        if type == 'middle':
            if predict:
                save_path = f'/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636/{data_type}/middle_predict_type'
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                vis_save_middle(data['image'].numpy(), eval(data['bbox'][0]), newID, save_path, True)
            else:
                save_path = f'/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636/{data_type}/middle_gt_type'
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                vis_save_middle(data['image'].numpy(), data['label'][0][0], newID, save_path, False)
        elif type == 'all':
            if predict:
                save_path = f'/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636/{data_type}/all_predict_type/{newID}'
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                vis_save(data['image'].numpy(), eval(data['bbox'][0]), save_path, True)
            else:
                save_path = f'/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636/{data_type}/all_gt_type/{newID}'
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                vis_save(data['image'].numpy(), data['label'][0][0], save_path, False)
        else:
            # 0,25,50,75,100% image
            if predict:
                save_path = f'/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636/{data_type}/percen_predict_type/{newID}'
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                vis_save_percen(data['image'].numpy(), eval(data['bbox'][0]), save_path, True)
            else:
                save_path = f'/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636/{data_type}/percen_gt_type/{newID}'
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                vis_save_percen(data['image'].numpy(), data['label'][0][0], save_path, False)
    return save_path

def plot_vedio(path, img_name, predict):
    #path = '/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636/problem/21179295'
    img = cv2.imread(f'{path}/{img_name}/000.png')
    size = (img.shape[1],img.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    if predict:
        dir_path = f'{path}/vedio_predict'  
    else:
        dir_path = f'{path}/vedio_gt'
    video = cv2.VideoWriter(f'{dir_path}/{img_name}.avi', fourcc, 20, size) # 檔名, 編碼格式, 偵數, 影片大小(圖片大小)

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        
    files = os.listdir(f'{path}/{img_name}')
    files.sort()
    for i in files:
        file_name = f'{path}/{img_name}/{i}'
        img = cv2.imread(file_name)
        video.write(img)



if __name__ == '__main__':

    # bbox_file 
    # input data
    #bbox_file = pd.read_csv('/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_1.2_20220316081421_0.5636_new_data_final.csv')
    #bbox_file = pd.read_csv('/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_1.2_20220316081421_0.5636_new_data_valid_final.csv')
    #bbox_file = pd.read_csv('/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_1.2_20220316081421_0.5636_resize_new_data_valid_final.csv')
    #bbox_file = pd.read_csv('/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_1.2_20220316081421_0.5636_resize_new_data_test_final.csv')
    #bbox_file = pd.read_csv('/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_1.2_20220316081421_0.5636_new_data_valid_latest.csv')
    #bbox_file = pd.read_csv('/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_1.2_20220316081421_0.5636_resize_new_data_valid_final_old.csv')
    #bbox_file = pd.read_csv('/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_1.2_20220316081421_0.5636_resize_new_data_test_final.csv')
    bbox_file = pd.read_csv('/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636_resize_new_data_test_final_by_val.csv')
    '''
    # GT 
    train_file = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_train_20220310.csv')
    valid_file = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_valid_20220310.csv')
    test_file = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_test_20220310.csv')
    bbox_file = pd.concat([train_file, valid_file, test_file])
    '''
    # predict or true label
    predict = True
    # png formate all, percentage(0,25,50,75,100%), middle
    png_type = 'all'
    # data type all, valid, test ori or resize
    data_type = 'test_resize_by_val'
  

    all_cls = ['__background__','spleen']
    all_cls_label = [i for i in range(len(all_cls))]
    all_cls_dic = dict(zip(all_cls,all_cls_label))
    if predict:
        inference_transforms = Compose(
        [
            LoadNiftid(keys=["image"]),
            AddChanneld(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
            Orientationd(keys=["image"], axcodes="RAS"),
            CropForegroundd(keys=["image"], source_key="image"),
            #Resized(keys=['image'], spatial_size = (128, 128, 128)),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
            ),
            Annotate_predict(keys=["image"]),
            ToTensord(keys=["image"])
        ]
        )
    else:
        inference_transforms = Compose(
        [
            LoadNiftid(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Resized(keys=['image', 'label'], spatial_size = (128,128,128)),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
            ),
            #CropForegroundd(keys=["image", "label"], source_key="image"),
            Annotate(keys=["image", "label"]),
            ToTensord(keys=["image", "label"])
        ]
        )

    save_path = plot_png(bbox_file, inference_transforms, predict, png_type, data_type)

    # zip dictionary
    if 'type' in save_path.split('/')[-1]:
        os.system(f'zip -r {save_path}.zip {save_path}')
    else:
        file_name = '/'.join(save_path.split('/')[:-1])
        os.system(f'zip -r {file_name}.zip {file_name}')

    if png_type == 'all':
        if predict:
            path = f'/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636/{data_type}/all_predict_type'
        else:
            path = f'/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636/{data_type}/all_gt_type'
        file_name = os.listdir(path)
        for i in file_name:
            plot_vedio(path, i, predict)
            print(f'{i} vedio is finish.')

        # check vedio is not loss
        if predict:
            dir_path = f'{path}/vedio_predict'  
        else:
            dir_path = f'{path}/vedio_gt'
        file_name_check = os.listdir(dir_path)
        file_name_check = [i.replace('.avi','') for i in file_name_check]
        vedio_loss = [i for i in file_name if i not in file_name_check and 'vedio' not in i]
        if not vedio_loss:
            print("All vedios are done")
        else:
            for i in vedio_loss:
                plot_vedio(path, i, predict)
    print('All is done!')