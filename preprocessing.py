import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import scipy
import scipy.misc
import scipy.ndimage
from PIL import Image
import os
import glob
import config
import h5py
#先写训练
# 灰度图像为例glob.glob得到所有训练集的图片先取出3的整数倍的像素值，
# 以进行下面的图像模糊过程
# input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)
# 通过 把图片先缩小 再放大 ，得到模糊图像（后面要用的input），原始图像就是label，
# 然后把所有的图像裁剪成33*33,和21*21 分别添加到两个list中， 把两个list分别转成array，得到n*33*33*1 和n*21*21*1，然后存成.h5文件格式，
# 再读出注意 只按照块作为一个训练数据，不是一个图像作为一个数据

#image_size = 33 图像使用尺寸,
# label_size = 21 label_制作的尺寸,
# batch_size = 128
# c_dim = 1 图像维度
#tf.session as sess
#train:input_setup ,data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5"),train_data, train_label = read_data(data_dir)  # 读取.h5文件(由测试和训练决定)
def input_setup():
    #训练
    data = prepare_data(dataset = "Train")
    print(len(data))
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) // 2  # 6
    for i in range(len(data)):
        input_,label_ = preprocess(data[i],i,config.scale)
        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        for x in range(0,h-config.image_size+1,config.stride):
            for y in range(0,w-config.image_size+1,config.stride):
                sub_input = input_[x:x+config.image_size,y:y+config.image_size]#33*33
                sub_label = label_[x+padding:x+padding+config.label_size,y+padding:y+padding+config.label_size]#21*21
                sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  # 按image size大小重排 因此 imgae_size应为33 而label_size应为21
                sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
                sub_input_sequence.append(sub_input)  # 在sub_input_sequence末尾加sub_input中元素 但考虑为空
                sub_label_sequence.append(sub_label)
    arrdata = np.asarray(sub_input_sequence)# [?, 33, 33, 1]

    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]

    make_data(arrdata, arrlabel)
    print("ok")# 存成h5格式



def prepare_data(dataset):
    #训练
    filenames = os.listdir(dataset)#输出dataset下的所有文件名
    print(filenames)
    data_dir = os.path.join(os.getcwd(),dataset)
    print(data_dir)
    data = glob.glob(os.path.join(data_dir,"*.bmp"))
    return data

def preprocess(path,i,scale = 3):
    image = scipy.misc.imread(path,mode = 'YCbCr').astype(np.float32)
    image = modcrop(image,scale)
    label_ = image[:,:,0]
    image = image/255.
    label_ = label_/255.
    input_ = scipy.ndimage.interpolation.zoom(label_, (1./(scale)),mode='wrap',prefilter=False)
    input_ = scipy.ndimage.interpolation.zoom(input_,  ((scale)/1.),mode='wrap',prefilter=False)
    label_small = modcrop_small(label_)
    input_small = modcrop_small(input_)
    imsave( "C:\\Users\\zzy\\Desktop\\learn\\sample\\bicubic\\bicubic{}.bmp".format(i), input_small)  # 保存插值图像
    imsave("C:\\Users\\zzy\\Desktop\\learn\\sample\\orign\\origin{}.bmp".format(i), label_small)  # 保存原始图像
    imsave("C:\\Users\\zzy\\Desktop\\learn\\sample\\input\\input_{}.bmp".format(i),input_)  # 保存input_图像
    imsave("C:\\Users\\zzy\\Desktop\\learn\\sample\\label\\label_{}.bmp".format(i),label_,)  # 保存label_图像
    return input_, label_




#裁剪图片，保证图片大小可为scale的倍数
def modcrop(image,scale = 3):
    h,w,_ = image.shape
    h = h - np.mod(h,scale)
    w = w - np.mod(h,scale)
    image = image[0:h,0:w,:]
    return image

def imsave(path,image):
    return  scipy.misc.imsave(path, image)

#这个是输出图片的大小，直接将原图裁剪至输出大小作为真实图片
def modcrop_small(image):
  #6来自padding = abs(config.image_size - config.label_size) // 2
  #21来自label_size
  #33来自image_size
  #不知道这个公式怎么来的
  padding2 = 6
  #padding2 = 0
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = (h-33+1)//21*21+21+padding2
    w =(w-33+1)//21*21+21+padding2
    image1 = image[padding2:h, padding2:w, :]#6
  else:
    h, w = image.shape
    h = (h - 33 + 1) // 21 * 21 + 21 + padding2
    w = (w - 33 + 1) // 21 * 21 + 21 + padding2
    image1 = image[padding2:h, padding2:w]
  return image1

def make_data(data, label):
    savepath = os.path.join(os.getcwd(),'checkpoint\\train.h5')
    with h5py.File(savepath,'w')as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)

def read_data(path):
  """
  读取h5格式数据文件,用于训练或者测试
  参数:
    路径: 文件
    data.h5 包含训练输入
    label.h5 包含训练输出
  """
  with h5py.File(path, 'r') as hf:  #读取h5格式数据文件(用于训练或测试)
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label


