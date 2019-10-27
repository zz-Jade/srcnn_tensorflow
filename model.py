import tensorflow as tf
import numpy as np
import os
import time
import config
from preprocessing import(
    read_data,
    input_setup,
    imsave,
    #merge
)
import evaluate

class SRCNN(object):
    def __init__(self,sess):
        self.sess = sess
        self.image_size = config.image_size
        self.is_grayscale = (config.c_dim == 1)
        self.label_size = config.label_size
        self.batch_size = config.batch_size
        self.c_dim = config.c_dim
        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.build_model()

    def train(self):
        input_setup()
        data_dir = os.path.join(os.getcwd(), "checkpoint\\train.h5")
        train_data,train_label = read_data(data_dir)
        glob_step = tf.Variable(0)
        learning_rate_exp = tf.train.exponential_decay(config.learning_rate,glob_step,1480,0.98,
                                                       staircase=True)# 每1个Epoch 学习率*0.98
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate_exp).minimize(self.loss,
                                                                                      global_step = glob_step)
        tf.global_variables_initializer().run()
        counter = 0
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        print("Training...")
        for ep in range(config.epoch):
            batch_indx = len(train_data)//config.batch_size
            for idx in range(0,batch_indx):
                batch_images = train_data[idx * config.batch_size: (idx + 1) * config.batch_size]
                batch_labels = train_label[idx * config.batch_size: (idx + 1) * config.batch_size]
                counter += 1
                _,err = self.sess.run([self.train_op,self.loss],
                                      feed_dict = {self.images:batch_images,self.labels:batch_labels})
                if counter % 10 == 0:  # 10的倍数step显示
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep + 1), counter, time.time() - start_time, err))
                if counter % 500 == 0:  # 500的倍数step存储
                    self.save(config.checkpoint_dir, counter)

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name="image")
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name="label")
        # 第一层CNN：对输入图片的特征提取。（9 x 9 x 64卷积核）
        # 第二层CNN：对第一层提取的特征的非线性映射（1 x 1 x 32卷积核）
        # 第三层CNN：对映射后的特征进行重建，生成高分辨率图像（5 x 5 x 1卷积核）
        # 权重
        self.weights = {
            'w1':tf.Variable(tf.random_normal([9,9,1,64],stddev=1e-3),name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([1]), name='b3')
        }
        self.pred = self.model()
        # 以MSE作为损耗函数
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.saver = tf.train.Saver()

    def model(self):
        conv1 = tf.nn.relu(
            tf.nn.conv2d(self.images,self.weights['w1'],strides = [1,1,1,1],padding='VALID')+self.biases['b1'])
        conv2 = tf.nn.relu(
            tf.nn.conv2d(conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b3']
        return conv3

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        # 加载路径下的模型(.meta文件保存当前图的结构; .index文件保存当前参数名; .data文件保存当前参数值)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir,
                                                       ckpt_name))  # saver.restore()函数给出model.-n路径后会自动寻找参数名-值文件进行加载
            return True
        else:
            return False

    def save(self,checkpoint_dir,step):
        model_name = 'SCRNN.model'
        model_dir = "%s_%s" % ("srcnn",self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir,model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),  # 文件名为SRCNN.model-迭代次数
                        global_step=step)



if __name__ == '__main__':
    with tf.Session() as sess:
        srcnn = SRCNN(sess)
        srcnn.train()