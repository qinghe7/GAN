from keras import models
from keras.engine import input_layer, training
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

import numpy as np

class GAN():
    def __init__(self):
        self.batch_size = 100
        self.original_dim = 28*28
        self.latent_dim = 2 # 潜在空间维度
        self.intermediate_dim = 256 # 中间维度
        self.nb_epoch = 5
        self.epsilon = 1.0

    def _encoder(self): #编码器
        x = Input(shape=(self.original_dim,),name = "input") # 编码器输入
        h = Dense(self.intermediate_dim, activation='relu', name = 'encoding')(x) # 中间层
        z_mean = Dense(self.latent_dim, name = 'mean')(h) # 定义潜在空间
        z_log_var = Dense(self.latent_dim, name = "log-variance")(h)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))(z_mean, z_log_var) # 注意output_shape不是一定要用tensorlfow后端
        self.encoder =Model(x, [z_mean, z_log_var, z],name = 'encoder')

    # 定义采样辅助函数
    def sampling (self,args):  #生成标准正态分布
        z_mean ,z_log_var = args 
        epsilon = K.random_normal(shape=(self.batch_size,self.latent_dim), mean=0.)
        return z_mean +K.exp(z_log_var / 2) * epsilon

    
    def _decoder(self): #解码器
        input_decoder = Input(shape=(self.latent_dim,), name="decoder_input") # 解码器的输入
        decoder_h = Dense(self.intermediate_dim , activation="rule", name = "decoder_h")(input_decoder) # 潜在空间转化为中间维度
        x_decoded = Dense(self.original_dim, activation='sigmoid', name = 'flat_decoded')(decoder_h) #得到原始维度的平均值
        self.decoder = Model(input_decoder, x_decoded ,name = 'decoder') # 将解码器定义为一个keras模型


    def train(self):
        output_combined = self.decoder(self.encoder(self.x)[2]) # 抓取数据
        vae = Model(self.x ,output_combined) # 变分自编码器
        vae.summary() # 模型的信息
        
        def vae_loss(x, x_decoded_mean, z_log_var, z_mean, original_dim = self.original_dim):
            xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean) # binary_crossentropy二值交叉熵
            kl_loss = -0.5*K.sum(
                1 + z_log_var - K.square(z_mean) - K.square(z_mean) - K.exp(z_log_var), axis=-1
            )
        
        vae.compile(optimizer="rmsprop", loss=vae_loss)

        (x_train,y_train),(x_test,y_test)  = self._split_data()

        vae.fit(x_train,x_train,
                shuffle= True,
                nb_epoch = self.nb_epoch, # 这里nb_epoch已经替换成epoch了
                batch_size= self.batch_size,
                validation_data= (x_test,x_test),
                verbose=1) # 由于编码器的输入和解码器的输出都是mnist数据集字母，所以data和label都是x_train
                            # verbose：日志显示
                            # verbose = 0 为不在标准输出流输出日志信息
                            # verbose = 1 为输出进度条记录
                            # verbose = 2 为每个epoch输出一行记录
                            # 注意： 默认为 1


    def _split_data(self):
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        
        x_train = x_train.astype("float32") / 255.
        x_test = x_test.astype("float32") / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # np.prod 
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1,:])))
        return (x_train,y_train),(x_test,y_test)

    
