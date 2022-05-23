import pandas as pd
import os
import shutil
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import csv
import glob

# Based on MITK v2021.02

# 將bounding box label和image位置合併成dataset讀取格式
# bounding box label處理: offset相減後乘rever_matrix再將偏移量加上
# image 留下原本的 affine，以利之後轉向
label_path = '/tf/jacky831006/object_detect_data_new/neg/label_untrans'
list_= os.listdir(label_path)
parent_dirs = '/tf/jacky831006/object_detect_data_new/neg/image'
outpath = '/tf/jacky831006/object_detect_data_new/neg/label'
table = [['image','annotation','label']]
#write_dic = {}
for i in list_:
    file = i.replace(" Bounding Shape.mitkgeometry","")
    tgt_url = f"{parent_dirs}/{file}.nii.gz"
    nib_image = nib.load(tgt_url)
    header = nib_image.header
    arr_image = nib_image.get_data()
    affine = header.get_best_affine()
    offx, offy, offz = header['qoffset_x'],header['qoffset_y'],header['qoffset_z']
    path = f'{label_path}/{i}'
    out_path = f'{outpath}/{file}.nii.gz'
    print(path)
    #print(out_path)
    with open(path, 'r') as f:
        content = f.read() # 讀取檔案內容
        # (-?\d+\.?\d*) 取正負整數、小數
        # <IndexToWorld type="Matrix3x3" m_0_0="0.78200000524520874" m_0_1="-0" m_0_2="-0" m_1_0="-0" m_1_1="0.78200000524520874" m_1_2="-0" m_2_0="0" m_2_1="0" m_2_2="5"/>
        pattern = re.compile(r'<IndexToWorld type="Matrix3x3" m_0_0="(-?\d+\.?\d*)" m_0_1="(-?\d+\.?\d*)" m_0_2="(-?\d+\.?\d*)" m_1_0="(-?\d+\.?\d*)" m_1_1="(-?\d+\.?\d*)" m_1_2="(-?\d+\.?\d*)" m_2_0="(-?\d+\.?\d*)" m_2_1="(-?\d+\.?\d*)" m_2_2="(-?\d+\.?\d*)"/>')
        matrix = pattern.search(content)
        #<Offset type="Vector3D" x="-196.28900146484375" y="199.40669250488281" z="-358.5"/>
        offset_pattern = re.compile(r'<Offset type="Vector3D" x="(-?\d+\.?\d*)" y="(-?\d+\.?\d*)" z="(-?\d+\.?\d*)"/>')
        off = offset_pattern.search(content)
        #<Min type="Vector3D" x="306.99999999999977" y="93.999999999999886" z="68"/>
        min_pattern = re.compile(r'<Min type="Vector3D" x="(\d+\.?\d*)" y="(-?\d+\.?\d*)" z="(-?\d+\.?\d*)"/>')
        min_ = min_pattern.search(content)
        #<Max type="Vector3D" x="443.99999999999983" y="241.9999999999994" z="92"/>
        max_pattern = re.compile(r'<Max type="Vector3D" x="(\d+\.?\d*)" y="(-?\d+\.?\d*)" z="(-?\d+\.?\d*)"/>')
        max_ = max_pattern.search(content)

        # matrix.group 裡面配對的第幾個內容 0 為全部內容
        index2world_matrix = np.array([
                                      [matrix.group(1),matrix.group(2),matrix.group(3)],
                                      [matrix.group(4),matrix.group(5),matrix.group(6)],
                                      [matrix.group(7),matrix.group(8),matrix.group(9)]]).astype(np.float)
        # check image and label affine
        if index2world_matrix[0, 0] * affine[0, 0] < 0:
            offx = -offx
        if index2world_matrix[1, 1] * affine[1, 1] < 0:
            offy = -offy
        if index2world_matrix[2, 2] * affine[2, 2] < 0:
            offz = -offz
            
        rever_matrix = np.linalg.inv(index2world_matrix)
        offset = np.array([off.group(1),off.group(2),off.group(3)]).astype(np.float)
        min_matrix = np.array([min_.group(1),min_.group(2),min_.group(3)]).astype(np.float)
        max_matrix = np.array([max_.group(1),max_.group(2),max_.group(3)]).astype(np.float)
        # offset相減後乘rever_matrix再將偏移量加上
        off_test = np.array(offset - [offx,offy,offz])
        test = off_test.dot(rever_matrix)
        min_matrix = min_matrix + test
        max_matrix = max_matrix + test
        
        low = int(min_matrix[2])
        height = int(max_matrix[2])
        right = int(max_matrix[1])
        left = int(min_matrix[1])
        top = int(max_matrix[0])
        bottom = int(min_matrix[0])
        # left,bottom,right,top,low,height
        # x1,y1,x2,y2,z1,z2
        table.append([tgt_url,f"({left},{bottom},{right},{top},{low},{height})",out_path])
        
        # save the label box nifti to path
        inti = np.full(arr_image.shape, 0)
        #print(left,bottom,right,top,low,height)
        point = np.ones((top-bottom, right-left, height-low))
        inti[bottom:top, left:right, low:height] = point
        out = nib.Nifti1Image(inti, affine=affine, header=header)
        nib.save(out, out_path)
                
# 將lable 寫入CSV
print("=================== write csv ===================")
with open('/tf/jacky831006/object_detect_data_new/spleen_neg_annotation.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(table)