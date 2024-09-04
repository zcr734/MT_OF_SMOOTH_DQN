# 定义智能体网络
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


class dqn(tf.keras.Model):
    def __init__(self, act_dim, lr=0.01, e_greed=0.1):
        super(dqn, self).__init__()
        self.act_dim = act_dim  # 动作维数
        self.e_greed = e_greed
        self.lr = lr
        self.model = self.build_model()
        self.loss_fun = tf.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
    def build_model(self):
        # NOTE ------------------ build evaluate_net -----------------
        model = models.Sequential()
        model.add(layers.Reshape((115, 1)))
        model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv1'))
        model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv2'))
        model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv3'))
        model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv4'))
        model.add(layers.Flatten())

        model.add(layers.Dense(256, activation='relu', name='l0'))
        model.add(layers.Dense(256, activation='relu', name='l1'))
        model.add(layers.Dense(256, activation='relu', name='l2'))
        model.add(layers.Dense(256, activation='relu', name='l3'))
        model.add(layers.Dense(self.act_dim, name='l4'))
        return model

    def call(self, obs, *args, **kwargs):  # *args就是就是传递一个可变参数列表给函数实参，这个参数列表的数目未知，甚至长度可以为0。
        # ** kwargs则是将一个可变的关键字参数的字典传给函数实参，同样参数列表长度可以为0或为其他值。
        act = self.model(obs)
        return act

    def test(self, obs, training=False):
        obs = tf.expand_dims(obs, axis=0)
        Q = self.call(obs)
        action_index = tf.argmax(Q, axis=1)  # 返回Q最大值的索引(代表第几个动作)
        action_index = action_index.numpy()[0]
        if training:
            sample = np.random.rand()  # 产生0~1之间的小数
            if sample < self.e_greed:
                action_index = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择，随机选择的动作概率为0.1
            else:
                self.e_greed = 0.1
        Q_max = np.mean(Q.numpy()[0])
        return action_index, Q_max  # 返回选取动作的索引和当前的Q值

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)