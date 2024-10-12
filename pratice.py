#coding=UTF-8
from matplotlib import pyplot as plt
from DQN import dqn
from Environment import *
from algorithm import dqn_algorithm as alg
from algorithm import MT1D
import numpy as np
from tools import *
import time
import tensorflow as tf
from tqdm import tqdm,trange
from config import DefaultConfig
import os
gpulist=tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpulist[0],True)

def pratice():



    plt.ioff()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    pbar = tqdm(total=300)
    updatenum= model_updates=model_updates1 = num = sumloss = sumQ = sumreward  = 0
    sum_data_reward=0
    sum_model_reward=0
    sum_data_reward_list=[]
    sum_model_reward_list=[]

    while (num < 300):

        if num == 0:
            timebegin = time.time()
            print("the current time:", timebegin)
        if num % 50 == 0 and num != 0:
            timeend = time.time()
            timelist.append(timeend - timebegin)
            timebegin = time.time()

        T = np.logspace(-3, 3, 15)
        guance = MT1D(T, rho, thi)
        for i in range(len(guance[0])):
            guance[0][i]=guance[0][i]+random.gauss(0,0.05)
            guance[1][i] = guance[1][i] + random.gauss(0, 0.05)
        # guance[0] = [2466.20865723, 2034.09939687, 1835.5380766, 1386.72899556,
        #              815.20797979, 500.85140707, 307.85804617, 162.37306102,
        #              78.8960647, 41.21235713, 32.03383206, 32.88581052,
        #              31.4810154, 22.90690236, 13.99949944]
        # guance[1] = [50.6001223, 52.50210452, 54.81156599, 62.4083678, 67.0342564,
        #              67.19829381, 70.57495631, 73.3688767, 73.04974058, 64.70386014,
        #              53.55401454, 49.85834526, 54.71019415, 62.18453316, 65.26232766]
        guance = np.array(guance)
        save_data(guance, "{}/guance".format(save_path))
        Mes = [T, guance]  # 厚度，分辨率,约束
        Mes1 = {"regularization_factor":opt.regularization_factor, "noise_factor": opt.noise_factor}
        step = environment(Mes, Mes1)
        res_log=step.struct_model_log10
        per=0
        while(True):

            step.struct_model_log10 = res_log
            step.obs, step.data_loss, step.restraint_loss = step.init_obs()  # 初始化环境和误差
            done = loss = 0  # 迭代次数
            obs = step.obs  # 初始环境
            while done == 0:
                action, Q = model.test(obs, training=True)  # 得到最大Q值的索引，对应一个动作
                next_obs, reward,reward1,reward2, done, last_error = step.state(action)  # 动作带入环境
                alg.memory.append((obs, action, reward, next_obs, done))  # 放入经验池
                alg.memory_counter=alg.memory_counter+1
                sumreward = sumreward + reward
                sum_data_reward=sum_data_reward+reward1
                sum_model_reward=sum_model_reward+reward2
                sumQ = sumQ + Q
                if (alg.memory_counter >= MEMORY_WARMUP_SIZE) and (updatenum % LEARN_FREQ == 0):  # 经验池到一定程度开始batchsize学习
                    batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = alg.memory.sample(BATCH_SIZE)
                    loss = alg.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done, model,
                                     targetmodel)  # 学
                    model_updates=model_updates+1
                    model_updates1=model_updates1+1
                    sumloss=sumloss+loss
                    if(model_updates1%500==0):
                        trainlosslist.append(loss)
                        sumloss=0
                obs = next_obs
                updatenum=updatenum+1
            res_log = np.copy(step.struct_model_log10)
            if last_error < 1:
                model.model.save("{}/{}".format(save_path, "10_5_m.keras"))
                targetmodel.model.save("{}/{}".format(save_path, "10_5_t.keras"))
                model_updates_list.append(model_updates)
                mean_action_Q_list.append(sumQ/model_updates)
                lasterrlist.append(last_error)
                reslist.append(step.struct_model_log10)
                mean_reward_list.append(sumreward/model_updates)
                sum_data_reward_list.append(sum_data_reward/model_updates)
                sum_model_reward_list.append(sum_model_reward/model_updates)

                save_data(sum_data_reward_list, "res/{}/R_data".format(save_path))
                save_data(sum_model_reward_list, "res/{}/R_model".format(save_path))
                save_data(model_updates_list, "res/{}/updates".format(save_path))
                save_data(reslist, "res/{}/res".format(save_path))
                save_data(lasterrlist, "res/{}/lasterror".format(save_path))
                save_data(mean_action_Q_list, "res/{}/meanQ".format(save_path))
                save_data(mean_reward_list, "res/{}/meanR".format(save_path))
                save_data(trainlosslist, "res/{}/trainlosslist".format(save_path))
                sumQ = 0
                sumreward = 0
                model_updates=0
                sum_data_reward = 0
                sum_model_reward = 0
                break

        num = num + 1
        pbar.update(1)
        if num == 300:
            timeend = time.time()
            timelist.append(timeend - timebegin)
            save_data(timelist, "res/{}/time".format(save_path))
    print("finished time:",time.localtime())




if __name__ == "__main__":
    set_gpu("0")
    """
    set the paremeters of DQN
    """
    opt=DefaultConfig()
    LEARN_FREQ = opt.learn_frequency
    MEMORY_SIZE = opt.memory_size
    MEMORY_WARMUP_SIZE = opt.memory_warmup_size
    constraint_network = opt.constraint_network
    BATCH_SIZE = opt.batch_size
    LEARNING_RATE = opt.learning_rate
    GAMMA = opt.gamma
    save_path = opt.result_path
    model_path=opt.model_path
    """
    initialise the replay memory，double Q network and variable 
    """
    rho =opt.resisty
    thi =opt.thickness
    model = dqn(3, lr=LEARNING_RATE, e_greed=0.1)
    targetmodel = dqn(3, lr=LEARNING_RATE, e_greed=0.1)
    model_updates_list = []  # 每次反演模型更新次数记录
    mean_action_Q_list = []  # 每次反演均值Q记录
    mean_action_V_list = []  # 每次反演均值Q记录
    mean_reward_list = []  # 每次反演均值Q记录
    losslist = []  # 每次反演最后的损失记录
    trainlosslist = []
    lasterrlist = []
    reslist = []  # 每次反演的结果记录
    timelist = []
    model.model.build(input_shape=(None, 115,1))
    model.model.summary()
    targetmodel.model.build(input_shape=(None, 115,1))
    alg = alg(MEMORY_SIZE,constraint_network)  # 导入DQN算法类
    """
    begin to pratice
    """
    pratice()