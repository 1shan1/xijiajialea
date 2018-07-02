import numpy as np
import os
import random


img_path = '/data_1/data/formatted_trainval/shanghaitech_part_A_patches_100/val'
den_path = '/data_1/data/formatted_trainval/shanghaitech_part_A_patches_100/val_den'



train_file = open('/data_1/data/formatted_trainval/shanghaitech_part_A_patches_100/val_img_den.txt','w')
file_str_list = []

filelist = os.listdir(img_path)

for fname in filelist:
  every_img_path =os.path.join(img_path,fname)
  every_label_path = os.path.join(den_path,os.path.splitext(fname)[0] + '.csv')
  dst_str = every_img_path +' '+every_label_path+'\n'
  file_str_list.append(dst_str)



random.shuffle(file_str_list)

for txt_str in file_str_list:
    train_file.write(txt_str)
