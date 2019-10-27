import preprocessing
import evaluate
import tensorflow as tf
import config
import numpy as np
import os
from model import SRCNN
import time
import h5py

def test(self,sess):
    nx,ny = input_up(sess)
    print(nx,ny)
    data_dir = os.path.join(os.getcwd(), "checkpoint\\test.h5")
    test_data, test_label = preprocessing.read_data(data_dir)
    if SRCNN.load(self,config.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
    print("Testing...")
    #312*21
    result = SRCNN.model(self).eval({self.images:test_data,self.labels:test_label})
    result = merge(result,[nx,ny])
    result = result.squeeze() # 除去size为1的维度
    # result= exposure.adjust_gamma(result, 1.07)#调暗一些
    image_path = os.path.join(os.getcwd(), "sample")
    image_path = os.path.join(image_path, "MySRCNN.bmp")
    preprocessing.imsave( image_path,result)




def input_up(sess):
    data = preprocessing.prepare_data(dataset='Test')
    print(len(data))
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) // 2  # 6
    input_,label_ = preprocessing.preprocess(data[0],config.scale)
    if len(input_.shape) == 3:
        h, w, _ = input_.shape
    else:
        h, w = input_.shape
    nx = 0  # 后注释
    ny = 0  # 后注释
    # 自图需要进行合并操作
    for x in range(0, h - config.image_size + 1, config.stride):  # x从0到h-33+1 步长stride(21)
        nx += 1
        ny = 0
        for y in range(0, w - config.image_size + 1, config.stride):  # y从0到w-33+1 步长stride(21)
            ny += 1
            # 分块sub_input=input_[x:x+33,y:y+33]  sub_label=label_[x+6,x+6+21, y+6,y+6+21]
            sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
            sub_label = label_[x + padding:x + padding + config.label_size,
                        y + padding:y + padding + config.label_size]  # [21 x 21]
            sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
            sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
            sub_input_sequence.append(sub_input)
            sub_label_sequence.append(sub_label)
    # 上面的部分和训练是一样的
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]
    make_data(arrdata, arrlabel)  # 存成h5格式
    return nx,ny

def merge(images, size):
  print(images.shape)
  h, w = images.shape[1], images.shape[2] #觉得下标应该是0,1
  #h, w = images.shape[0], images.shape[1]
  img = np.zeros([h*size[0], w*size[1], 1])
  print(img.shape)
  j = 0
  k = 0
  for i in range(13):
        while(j<24):
            img[(i*21):((i + 1) * 21), (j*21):((j + 1) * 21), :] = images[k, 0:21, 0:21, :]
            j += 1
            k += 1
        if(j == 24):
            j = 0
  print(k)
  return img

def make_data(data, label):
    savepath = os.path.join(os.getcwd(),'checkpoint\\test.h5')
    with h5py.File(savepath,'w')as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)

if __name__ == '__main__':
    with tf.Session() as sess:
        srcnn = SRCNN(sess)
        test(srcnn,sess)
