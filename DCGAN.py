#深度卷积生成对抗网络

# 架构设计规则：
# 使用卷积层代替池化层 
# 除去全连接层FCN 
# 使用批归一化 （batch bormalization） 
# 使用恰当的激活函数

# 使用keras
import numpy as np

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv1D, UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam



class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels =1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        
        optimizer = Adam(0.0002, 0.5)

        #Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                    optimizer=optimizer,
                                    metrics = ['accuracy'])
        
        self.generator = self.build_generator()

        z = Input(shape=(100,))
        img = self.generator(z)

        self.discriminator.trainable = False
        Valid = self.discriminator(img)

        self.combined = Model(z, Valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)



        '''
        构建生成器
        采用上采样加卷积层来代替池化层
        中间不包括全连接层(最开始有Dense（全连接层）)
        加入批量归一化(BN)
        生成器中激活函数使用Relu函数，输出层使用tanh函数
        '''
    def build_generator(self):
            
        model = Sequential()
            
        model.add(Dense(128*7*7, activation='relu', input_dim= self.latent_dim)) # 稠密连接 即FCN
        model.add(Reshape((7, 7, 128))) #reshape层
        model.add(UpSampling2D()) #上采样
        model.add(Conv2D(128, kernel_size=3, padding="same")) #输出维度128 padding ="same"  https://zhuanlan.zhihu.com/p/62760780
        model.add(BatchNormalization(momentum=0.8)) #批量归一化 动量=0.8 ？
        model.add(Activation('relu')) # 激活函数
        model.add(UpSampling2D()) # 上采样
        model.add(Conv2D(64, kernel_size=3, padding='same'))  
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh")) # tanh激活函数

        model.summary()# 打印模型信息

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    '''
    构建判别器：
    使用步长为2的卷积层来代替池化层（stride = 2）
    中间不含全连接层
    添加批量归一化
    激活函数使用LeakyReLU 斜率为0.2
    '''
    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape = self.img_shape, padding="same")) # "same"padding后图像尺寸是原图像尺寸除以stride
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2 , padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1)))) #就是padding 0 查看官方文档
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2)) # 激活函数LeakyReLU 斜率为0.2
        model.add(Dropout(0.25)) #概率为0.25
        model.add(Conv2D(128, kernel_size =3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size =3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten()) # 展平层，把二维矩阵flatten为一维向量
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)


    def train(self, epochs, batch_size=128, save_interval=50):
        
        (X_train,_), (_,_) = mnist.load_data()

        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.combined.train_on_batch(noise, valid)
            print(d_loss,g_loss)


if __name__ == "__main__":
    gan = DCGAN()
    gan.train(epochs=10)
