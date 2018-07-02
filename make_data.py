import cv2
import numpy as np
import os
import pandas as pd
import random


file_str_list = []
train_file = open('/data_1/data/formatted_trainval/shanghaitech_part_A_patches_100/train.txt','w')
test_file = open('/data_1/data/formatted_trainval/shanghaitech_part_A_patches_100/val.txt','w')
data_path ='/data_1/data/formatted_trainval/shanghaitech_part_A_patches_100/density_level'

filelist = os.listdir(data_path)

for fname in filelist:

  img_path =os.path.join(data_path,fname)
  img_filelist = os.listdir(img_path)
  for img_fname in img_filelist:
    every_img_path = os.path.join(img_path,img_fname)
    dst_str = every_img_path +' '+fname+'\n'
    file_str_list.append(dst_str)
    
random.shuffle(file_str_list)
test_list = file_str_list[:len(file_str_list)/5]
train_list = file_str_list[len(file_str_list)/5:]
for txt_str in test_list:
    test_file.write(txt_str)
for txt_str in file_str_list:
    train_file.write(txt_str)








