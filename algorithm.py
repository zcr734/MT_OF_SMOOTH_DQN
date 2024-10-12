# 定义智能体的决策(学习，权重更新)
import tensorflow as tf
import numpy as np
from replay_memory import ReplayMemory

def MT1D(T, rho, h):
    mu = 4e-7 * np.pi  # 真空磁导率
    k = np.zeros((len(rho), len(T)), dtype=complex)
    for i in range(len(rho)):
        k[i] = np.sqrt((-2j * np.pi * mu) / (T * rho[i]))
    m = len(rho)  # 层数
    Z = -(2j * mu * np.pi) / (T * k[m - 1])  # python 的索引从 0 开始，Z为地球表面阻抗
    layer = np.arange(m - 1, 0, -1)  # np.arange() 不包含 stop 参数
    for n in layer:
        A = -(1j * mu * 2 * np.pi) / (T * k[n - 1])
        B = np.exp(-2 * k[n - 1] * h[n - 1])
        Z = A * (A * (1 - B) + Z * (1 + B)) / (A * (1 + B) + Z * (1 - B))
    rho_a = (T / (mu * 2 * np.pi)) * (np.abs(Z) ** 2)
    phase = -np.arctan(Z.imag / Z.real) * 180 / np.pi
    return [rho_a, phase]


class dqn_algorithm():
    def __init__(self,MEMORY_WARMUP_SIZE,constraint_network):
        self.global_step = 0
        self.gamma = 0.95
        self.update_target_steps = constraint_network
        self.loss_fun = tf.losses.MeanSquaredError()
        self.memory = ReplayMemory(MEMORY_WARMUP_SIZE)
        self.memory_counter = 0
    def learn(self, obs, action, reward, next_obs, terminal, model, target_model):
        """ 使用DQN算法更新self.model的value网络
        """
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.replace_target(model, target_model)

        next_q_value = model(next_obs).numpy()
        action_index = np.argmax(next_q_value, axis=1)  # 返回Q最大值的索引(代表第几个动作)
        enum_action = list(enumerate(action_index))
        Qnextmax = tf.gather_nd(target_model(next_obs), indices=enum_action)

        # 计算目标
        target = reward + self.gamma * Qnextmax * (1 - terminal)
        # 训练模型
        with tf.GradientTape() as tape:
            # 计算 Q(s,a) 与 target_Q的均方差，得到loss
            # obs_feature = stepmodel(obs)
            predictions = model(obs, training=True)
            enum_action = list(enumerate(action))
            pred_action_value = tf.gather_nd(predictions, indices=enum_action)
            loss = self.loss_fun(target, pred_action_value)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


        self.global_step += 1
        return loss
    def replace_target(self, model, target_model):
        '''预测模型权重更新到target模型权重'''

        target_model.layers[0].set_weights(model.layers[0].get_weights())
     