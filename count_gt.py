import cv2
import numpy as np
import os
import pandas as pd
import random


#file_str_list = []
#train_file = open('/data_1/data/formatted_trainval/shanghaitech_part_A_patches_300/formatted_trainval/shanghaitech_part_A_patches_300/val.txt','w')
#test_file = open('/home/pengshanzhen/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/training-augment/data/L_test.txt','w')
#folder_root = '/home/pengshanzhen/high-quality-densitymap/GCE/data/'
out_path ='/data_1/data/formatted_trainval/shanghaitech_part_A_patches_100/class-level/'
data_path ='/data_1/data/formatted_trainval/shanghaitech_part_A_patches_100/train'
label_path ='/data_1/data/formatted_trainval/shanghaitech_part_A_patches_100/train_den'

filelist = os.listdir(data_path)
#f = open('/home/pengshanzhen/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_A_patches_9/2.txt','w')
i =0
for fname in filelist:

  img_path =os.path.join(data_path,fname)
  img_y = cv2.imread(img_path)
  img = cv2.imread(img_path,0)
  #img = img.astype(np.float32, copy=False)
  #index = fname.strip().rfind('X')
  #refname = fname[:index]
  #crefname = refname + '.csv'
  #print(crefname)
  #exit()
  den = pd.read_csv(os.path.join(label_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()
  #den = pd.read_csv(os.path.join(label_path,crefname), sep=',',header=None).as_matrix()
  den = den.astype(np.float32, copy=False)
  ht = img.shape[0]
  wd = img.shape[1]
  ht_1 = (ht/4)*4
  wd_1 = (wd/4)*4
  img = cv2.resize(img,(wd_1,ht_1))
  #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #img = np.transpose(img,(2,0,1))
  wd_2 = wd_1/4
  ht_2 = ht_1/4
  den = cv2.resize(den,(wd_2,ht_2))                
  den = den * ((wd*ht)/(wd_2*ht_2))
  gt_data = den.reshape((1,1,den.shape[0],den.shape[1]))  
  
  gt_count = np.sum(gt_data)
  
  if gt_count <= 10:
    out_img_path = out_path +'ex-lo' +'/' + fname
    cv2.imwrite(out_img_path,img_y)
  elif gt_count <= 50:
    out_img_path = out_path +'lo' +'/' + fname
    cv2.imwrite(out_img_path,img_y)
  elif gt_count <= 150:
    out_img_path = out_path +'med' +'/' + fname
    cv2.imwrite(out_img_path,img_y)
  elif gt_count <= 400:
    out_img_path = out_path +'hi' +'/' + fname
    cv2.imwrite(out_img_path,img_y)
  else:
    out_img_path = out_path +'ex-hi' +'/' + fname
    cv2.imwrite(out_img_path,img_y)
  
    





