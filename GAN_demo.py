from re import M, X
import numpy as np
import tensorflow as tf
from scipy.stats.stats import mode, norm
from matplotlib import pyplot as plt


# 真实数据样本的分布
class DataDistribution(object):
    def __init__(self):
        self.mu = 3
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


# 生成器的初始化分布(平均分布)
class GeneratorDistribution(object):
    def __init__(self, range):
        self.range =range
    
    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01


# 线性运算函数
def linear(input, output_dim, scope=None, stddev = 1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_creator_scope(scope or "linear"):
        w = tf.compat.v1.get_variable('w',[input.get_shape()[1], output_dim], initializer=norm)
        b = tf.compat.v1.get_variable("b", [output_dim], initializer= const)
        return tf.matmul(input, w)+b

def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input ,h_dim,"g0"))
    h1 =linear(h0, 1, "g1")
    return h1

def discriminator(input, h_dim):
    h_0 = tf.tanh(linear(input, h_dim * 2, "d0"))
    h_1 = tf.tanh(linear(h_0, h_dim *2,"d1"))
    h_2 = tf.tanh(linear(h_1, h_dim *2,"d2"))
    h3 = tf.sigmoid(linear(h_2, 1, "d3"))
    return h3

# 设置优化器 学习率衰减的梯度下降
def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.9
    num_decay_steps = 150
    batch= tf.Variable(0)
    learning_rate = tf.compat.v1.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list = var_list
    )
    return optimizer

# GAN类

class GAN(object):
    def __init__(self, data, gen, num_steps, batch_size, log_every):
        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.mlp_hidden_size = 4
        self.learning_rate = 0.3
        self._create_model()

# 创建模型，预判别器D_pre、生成器Generator和判别器，按照之前的公式定义生成器和判别器的损失函数los s g 与loss_d 以及它们的优化器opt_g 与opt_d ，其中Dl 与D2 分别代表真实数据与生成数据的判别。
    def _create_model(self):

        with tf.compat.v1.variable_scope('D_pre'):
            self.pre_input = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.pre_labels = tf.compat.v1.placeholder(tf.float32, shape =(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre -self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        with tf.compat.v1.variable_scope("Generator"):
            self.z = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.G = generator(self.z, self.mlp_hidden_size)
        
        with tf.compat.v1.variable_scope("Discriminator") as scope:
            self.x = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.D1 = discriminator(self.x, self.mlp_hidden_size)
            scope.reuse_variables()
            self.D2 = discriminator(self.G, self.mlp_hidden_size)

        self.loss_d =  tf.reduce_mean(-tf.log(self.D1) - tf.log(1 -self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        self.d_pre_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope="D_pre")
        self.d_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
        self.g_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
        
        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)


    def train(self):
        with tf.compat.v1.Session() as session:
            tf .compat.v1.global_variables_initializer().run()

            #pretraining discriminator
            num_pretrain_steps = 1000
            for step in range(num_pretrain_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10.0
                labels = norm.pdf(d, loc=self.data.mu, scale =self.data.sigma)
                pretrain_loss,_ =  session.run([self.preloss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (self.batch_size, 1)),
                    self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                })
            self.weightsD = session.run(self.d_pre_params)
            for i,v in enumerate(self.d_params):
                session.run(v.assigin(self.weightsD[i]))

            for step in range(self.num_steps):
                #update discriminator
                X = self.data.sample(self.batch_size)
                z = self.gen.sample(self.batch_size)
                loss_d,_ = session.run([self.loss_d, self, self.opt_d], {
                    self.X:np.reshape(X, (self.batch_size, 1)),
                    self.z:np.reshape(z, (self.batch_size, 1))
                })

                #update generator
                z = self.gen.sample(self.batch_size)
                loss_g,_ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                if step % self.log_every == 0:
                    print('{}:{}\t{}'.format(step, loss_d, loss_g))
                if step % 100 == 0 or step == 0 or step == self.num_steps -1:
                    self._plot_distribution(session)
                
    def _samples(self, session, num_points = 10000, num_bins = 100):
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range ,num_bins)

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points,1))
        for i in range (num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size *(i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins ,density=True)
        return pd ,pg

    def _plot_distribution(self, session):
        pd, pg = self._samples(session)
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='Real Data')
        plt.plot(p_x, pg, label='Generated Data')
        plt.title('GAN Visualization')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()

def main (arg):
    model = GAN(
        DataDistribution(),
        GeneratorDistribution(range=8),
        1200, #num_steps
        12, #batch_size
        10, #log_every
    )
    model.train()