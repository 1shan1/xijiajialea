import sys
import cv2
import caffe
import numpy as np
import random 
import gc
from fangxiang_dataset import fangxiang
from fangxiang_dataset import fangxiang_turn
from fangxiang_dataset import faceclass

point_num = [10,8,4]
label_num = 3

class data_Layer(caffe.Layer):
    def setup(self,bottom,top):
        params = eval(self.param_str)
        file_path = params['source']
        self.dataset=fangxiang(file_path)
    def reshape(self,bottom,top):
        top[0].reshape(self.dataset.batch_size, 1, 128, 128)
        top[1].reshape(self.dataset.batch_size, 1)
              
    def forward(self,bottom,top):
        top[0].data[...], top[1].data[...]=self.dataset.batch()
        
    def backward(self,top,propagate_down,bottom):
        pass

class turn_data_Layer(caffe.Layer):
    def setup(self,bottom,top):
        params = eval(self.param_str)
        file_path = params['source']
        self.dataset=fangxiang_turn(file_path)
    def reshape(self,bottom,top):
        top[0].reshape(self.dataset.batch_size, 3, 100, 100)
        top[1].reshape(self.dataset.batch_size, 1)
              
    def forward(self,bottom,top):
        top[0].data[...], top[1].data[...]=self.dataset.batch()
        
    def backward(self,top,propagate_down,bottom):
        pass

class face_class_data_Layer(caffe.Layer):
    def setup(self,bottom,top):
        params = eval(self.param_str)
        file_path = params['source']
        self.faceset=faceclass(file_path)
    def reshape(self,bottom,top):
        top[0].reshape(self.faceset.batch_size, 3, 100, 100)
        top[1].reshape(self.faceset.batch_size, 1)
              
    def forward(self,bottom,top):
        top[0].data[...], top[1].data[...]=self.faceset.batch()
        
    def backward(self,top,propagate_down,bottom):
        pass
        
class pts_loss_Layer(caffe.Layer):
    def setup(self,bottom,top):
        self.batchsz=bottom[0].data.shape[0]
        self.pointsz=bottom[0].data.shape[1]
        
    def reshape(self,bottom,top):
        self.score=np.where(bottom[2].data>0.5, bottom[2].data, 0.0)
        #print self.score
        self.diff_coor=np.zeros((self.batchsz, self.pointsz, 2))
        top[0].reshape(1)

    def forward(self,bottom,top):
        self.diff_coor=bottom[0].data-bottom[1].data
        loss=np.sum(0.5*(self.diff_coor ** 2), axis=2)
        loss=loss*self.score
        top[0].data[0]=np.mean(loss)

    def backward(self,top,propagate_down,bottom):
        shape=[self.batchsz, self.pointsz, 1]
        x=self.score
        #bottom[0].diff[...]=self.diff_coor*x.reshape(shape)/self.batchsz#/self.pointsz
        #print top[0].diff
        bottom[0].diff[...]=top[0].diff[...]*self.diff_coor*x.reshape(shape)/self.batchsz
        #print bottom[0].diff
        bottom[1].diff[...]=0
        bottom[2].diff[...]=0

class dir_pick_Layer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 1:
            raise Exception("Need 1 Inputs")

    def reshape(self,bottom,top):
        self.valid_index = np.array(range(64))
        self.N = len(self.valid_index)
        if self.N != 0:
            top[0].reshape(self.N, 8)

    def forward(self,bottom,top):
        top[0].data[...][...] = 0
        if self.N != 0:
            top[0].data[0:self.N] = bottom[0].data[self.valid_index]
            
    def backward(self,top,propagate_down,bottom):
        bottom[0].diff[...]=0
        if not propagate_down[0] or self.N==0:
            pass
        bottom[0].diff[self.valid_index] = top[0].diff[...]

class face_pick_Layer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 1:
            raise Exception("Need 1 Inputs")
    def reshape(self,bottom,top):
        self.valid_index = np.array(range(64,128))
        self.N = len(self.valid_index)
        if self.N != 0:
            top[0].reshape(self.N, 2)

    def forward(self,bottom,top):
        top[0].data[...][...] = 0
        if self.N != 0:
            top[0].data[0:self.N] = bottom[0].data[self.valid_index]
            
    def backward(self,top,propagate_down,bottom):
        bottom[0].diff[...]=0
        if not propagate_down[0] or self.N==0:
            pass
        bottom[0].diff[self.valid_index] = top[0].diff[...]

class lr_pick_Layer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 1:
            raise Exception("Need 1 Inputs")

    def reshape(self,bottom,top):
        self.valid_index = np.array(range(64))
        self.N = len(self.valid_index)
        if self.N != 0:
            top[0].reshape(self.N, 5)

    def forward(self,bottom,top):
        top[0].data[...][...] = 0
        if self.N != 0:
            top[0].data[0:self.N] = bottom[0].data[self.valid_index]
        
    def backward(self,top,propagate_down,bottom):
        bottom[0].diff[...]=0
        if not propagate_down[0] or self.N==0:
            pass
        bottom[0].diff[self.valid_index] = top[0].diff[...]
