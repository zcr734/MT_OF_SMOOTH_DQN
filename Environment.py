import numpy as np
import random


class environment:
    def __init__(self, mes, mes1, guang=True):
        self.T = np.array(mes[0])
        self.true_result = mes[1]
        # self.struct_thi = np.array([500, 600, 720, 865, 1035, 1245, 1495, 1795, 2150])
        self.struct_thi =  np.array([600,672,752,842,944,1057,1184,1326,1485,1663,1863,2087,2337,2618,2932,3284,3678,4119,4613])
        self.struct_model_log10 = np.zeros((len(self.struct_thi) + 1,)) + 2  # 初始100

        self.struct_model = np.zeros((len(self.struct_thi) + 1,)) + 100
        self.pro = 0
        self.guang = guang
        self.regu = mes1['regularization_factor']
        self.zaoshen = mes1['noise_factor']
        self.obs, self.data_loss, self.restraint_loss= self.init_obs()  # 初始化环境和误差


    # 返回状态，奖励
    def state(self, action):
        if action == 0:
            pass
        elif action == 1:
            self.struct_model_log10[self.pro % len(self.struct_model_log10)] = self.struct_model_log10[
                                                                                   self.pro % len(
                                                                                       self.struct_model_log10)] - 0.05
        elif action == 2:
            self.struct_model_log10[self.pro % len(self.struct_model_log10)] = self.struct_model_log10[
                                                                                   self.pro % len(
                                                                                       self.struct_model_log10)] + 0.05

        if self.struct_model_log10[self.pro % len(self.struct_model_log10)] < 0:
            self.struct_model_log10[self.pro % len(self.struct_model_log10)] = 0
        if self.struct_model_log10[self.pro % len(self.struct_model_log10)] > 4:
            self.struct_model_log10[self.pro % len(self.struct_model_log10)] = 4

        self.obs, data_loss, restraint_loss = self.ret_obs()  # 返回下一次的obs和误差
        done = 0

        if data_loss < self.data_loss:
            data_reward = 1
        elif data_loss == self.data_loss:
            data_reward = 0
        else:
            data_reward = -1

        if restraint_loss < self.restraint_loss:
            model_reward = np.tanh(0.3*(data_loss-1))*0.5
        elif restraint_loss > self.restraint_loss:
            model_reward = - np.tanh(0.3*(data_loss-1))*0.5
        else:
            model_reward = 0


        sumreward=data_reward+model_reward
        if (data_loss) < 1:
            done = 1
            self.pro = 0
        elif (self.pro == len(self.struct_model_log10) - 1):
            done = 1
            self.pro = 0
        else:
            self.pro = self.pro + 1
        self.data_loss = data_loss
        self.restraint_loss = restraint_loss
        return self.obs, sumreward,data_reward,model_reward, done, data_loss

    def init_obs(self):
        for i in range(len(self.struct_model_log10)):
            self.struct_model[i] = 10 ** self.struct_model_log10[i]
        ypre = self.MT1D(self.T, self.struct_model, self.struct_thi)
        fitting_error = sum(abs((ypre[0] - self.true_result[0])) / (self.true_result[0] * self.zaoshen)) / len(
            self.T) + sum(abs((ypre[1] - self.true_result[1])) / (self.true_result[1] * self.zaoshen)) / len(self.T)

        loss_guanghua = 0
        for i in range(len(self.struct_model_log10) - 2):
            loss_guanghua = loss_guanghua + abs(
                self.struct_model_log10[i] - 2 * self.struct_model_log10[i + 1] + self.struct_model_log10[i + 2])
        dangqiancen = np.zeros((20,))
        dangqiancen[self.pro] = 1  # 下一状态obs_next
        struct_rho_log10 = self.struct_model_log10.copy()
        obs = np.concatenate(
            [[self.true_result[0]], [ypre[0]],[self.true_result[0]], [self.true_result[1]], [ypre[1]], [struct_rho_log10], [dangqiancen]], 1)
        obs = np.reshape(obs, (115,))

        return obs, fitting_error, loss_guanghua

    # 计算误差并构造环境
    def ret_obs(self):
        for i in range(len(self.struct_model_log10)):
            self.struct_model[i] = 10 ** self.struct_model_log10[i]
        ypre = self.MT1D(self.T, self.struct_model, self.struct_thi)
        fitting_error = sum(abs((ypre[0] - self.true_result[0])) / (self.true_result[0] * self.zaoshen)) / len(
            self.T) + sum(abs((ypre[1] - self.true_result[1])) / (self.true_result[1] * self.zaoshen)) / len(self.T)

        loss_guanghua = 0
        for i in range(len(self.struct_model_log10) - 2):
            loss_guanghua = loss_guanghua + abs(
                self.struct_model_log10[i] - 2 * self.struct_model_log10[i + 1] + self.struct_model_log10[i + 2])
        dangqiancen = np.zeros((20,))
        dangqiancen[(self.pro + 1) % len(self.struct_model_log10)] = 1  # 下一状态obs_next
        struct_rho_log10 = self.struct_model_log10.copy()
        obs = np.concatenate(
            [[self.true_result[0]], [ypre[0]],[self.true_result[0]], [self.true_result[1]], [ypre[1]], [struct_rho_log10], [dangqiancen]], 1)
        obs = np.reshape(obs, (115,))

        return obs, fitting_error, loss_guanghua
        # 计算误差并构造环境

    def ret_obs_spilt(self, selected_minima):
        for i in range(len(self.struct_model_log10)):
            self.struct_model[i] = 10 ** self.struct_model_log10[i]
        ypre = self.MT1D(self.T, self.struct_model, self.struct_thi)
        fitting_error = sum(abs((ypre[0] - self.true_result[0])) / (self.true_result[0] * self.zaoshen)) / len(
            self.T) + sum(abs((ypre[1] - self.true_result[1])) / (self.true_result[1] * self.zaoshen)) / len(self.T)

        loss_guanghua = 0
        for j in range(len(self.struct_model_log10) - 2):
            if j not in selected_minima or j + 1 or j + 2 not in selected_minima:
                loss_guanghua = loss_guanghua + abs(
                    self.struct_model_log10[j] - 2 * self.struct_model_log10[j + 1] + self.struct_model_log10[j + 2])

        dangqiancen = np.zeros((20,))
        dangqiancen[self.pro % len(self.struct_model_log10)] = 1
        struct_rho_log10 = self.struct_model_log10.copy()
        obs = np.concatenate(
            [[self.true_result[0]], [ypre[0]], [self.true_result[1]], [ypre[1]], [struct_rho_log10], [dangqiancen]], 1)
        obs = np.reshape(obs, (100,))

        return obs, fitting_error, loss_guanghua

    # 正演代码
    def MT1D(self, T, rho, h):
        mu = (4e-7) * np.pi
        k = np.zeros((len(rho), len(T)), dtype=complex)
        for i in range(len(rho)):
            k[i] = np.sqrt((-1j * 2 * np.pi * mu) / (T * rho[i]))
        m = len(rho)  # 层数
        Z = -(2j * mu * np.pi) / (T * k[m - 1])  # python 的索引从 0 开始
        layer = np.arange(m - 1, 0, -1)  # np.arange() 不包含 stop 参数
        for n in layer:
            A = -(1j * mu * 2 * np.pi) / (T * k[n - 1])
            B = np.exp(-2 * k[n - 1] * h[n - 1])
            Z = A * (A * (1 - B) + Z * (1 + B)) / (A * (1 + B) + Z * (1 - B))
        rho_a = (T / (mu * 2 * np.pi)) * (np.abs(Z) ** 2)
        phase = -np.arctan(Z.imag / Z.real) * 180 / np.pi
        return [rho_a, phase]  # rho_a视电阻率 phase相位
