from text_dataset import TextDataSet
import cv2
import numpy as np
import re
import random

class fangxiang(TextDataSet):
    def __init__(self, file_path):
        p = TextDataSet.default_param
        p['filename'] = file_path
        p['batch_size'] = 64
        p['thread_num'] = 3
        TextDataSet.__init__(self, p)

    def _parse_line(self, line):
        label_index = line.rfind(' ')
        label_str = line[label_index+1:]
        pic_path = line[:label_index]
        li = []
        li.append(pic_path)
        li.append(label_str)
        return li

    def _record_process(self, record):
        im=cv2.imread(record[0],0)
        h,w = im.shape
        im = cv2.resize(im,(128,128))
        im = (im -127.5)*0.0078125
        im = im.reshape((1,128,128))
        label=int(record[1])
        return (im, label)

    def _compose(self, list_single):
        im=np.zeros((self.batch_size, 1, 128, 128))
        labels=np.zeros((self.batch_size,1))
        for id, s in enumerate(list_single):
            im[id]=s[0]
            labels[id]=s[1]
        return (im, labels)


class fangxiang_turn(TextDataSet):
    def __init__(self, file_path):
        p = TextDataSet.default_param
        p['filename'] = file_path
        p['batch_size'] = 64
        p['thread_num'] = 2
        TextDataSet.__init__(self, p)

    def _parse_line(self, line):
        label_index = line.rfind(' ')
        label_str = line[label_index+1:]
        pic_path = line[:label_index]
        li = []
        li.append(pic_path)
        li.append(label_str)
        return li

    def _record_process(self, record):
        im=cv2.imread(record[0])
        im=cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        if random.random() > 0.5:
            im=cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im=cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        h,w,c = im.shape
        im=(np.transpose(cv2.resize(im, (100, 100)), [2,0,1])-127.5)*0.0078125
        label=int(record[1])
        return (im, label)

    def _compose(self, list_single):
        im=np.zeros((self.batch_size, 3, 100, 100))
        labels=np.zeros((self.batch_size,1))
        for id, s in enumerate(list_single):
            im[id]=s[0]
            labels[id]=s[1]
        return (im, labels)

class faceclass(TextDataSet):
    def __init__(self):
        p = TextDataSet.default_param
        p['filename'] = '/media/f/jiakao/train/direction_turn_free1_noface_slice/train0909_face.txt'
        p['batch_size'] = 64
        p['thread_num'] = 2
        TextDataSet.__init__(self, p)

    def _parse_line(self, line):
        label_index = line.rfind(' ')
        label_str = line[label_index+1:]
        pic_path = line[:label_index]
        li = []
        li.append(pic_path)
        li.append(label_str)
        return li

    def _record_process(self, record):
        im=cv2.imread(record[0])
        im=cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        if random.random() > 0.5:
            im=cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im=cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        im=(np.transpose(cv2.resize(im, (100, 100)), [2,1,0])-127.5)*0.0078125
        label=int(record[1])
        return (im, label)

    def _compose(self, list_single):
        im=np.zeros((self.batch_size, 3, 100, 100))
        labels=np.zeros((self.batch_size,1))
        for id, s in enumerate(list_single):
            im[id]=s[0]
            labels[id]=s[1]
        return (im, labels)
