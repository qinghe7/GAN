from _typeshed import Self
from keras import models
from keras.engine import input_layer
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

    def _encoder(self):
        x = Input(shape=(self.original_dim,),name = "input") # 编码器输入
        h = Dense(self.intermediate_dim, activation='relu', name = 'encoding')(x) # 中间层
        z_mean = Dense(self.latent_dim, name = 'mean')(h) # 定义潜在空间
        z_log_var = Dense(self.latent_dim, name = "log-variance")(h)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))(z_mean, z_log_var) # 注意output_shape不是一定要用tensorlfow后端
        self.encoder =Model(x, [z_mean, z_log_var, z],name = 'encoder')

    # 定义采样辅助函数
    def sampling (self,args):
        z_mean ,z_log_var = args 
        epsilon = K.random_normal(shape=(self.batch_size,self.latent_dim), mean=0.)
        return z_mean +K.exp(z_log_var / 2) * epsilon

    
    def _decoder(self):
        input_decoder = Input(shape=(self.latent_dim,), name="decoder_input") # 解码器的输入
        decoder_h = Dense(self.intermediate_dim , activation="rule", name = "decoder_h")(input_decoder) # 潜在空间转化为中间维度
        x_decoded = Dense(self.original_dim, activation='sigmoid', name = 'flat_decoded')(decoder_h) #得到原始维度的平均值
        self.decoder = Model(input_decoder, x_decoded ,name = 'decoder') # 将解码器定义为一个keras模型


    def train(self):
        output_combined = self.decoder(self.encoder(self.x)[2])

        
