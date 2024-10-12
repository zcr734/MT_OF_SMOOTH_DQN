from matplotlib import pyplot as plt
from DQN import dqn
from Environment import *
from replay_memory import ReplayMemory
from algorithm import dqn_algorithm as alg
from algorithm import MT1D
import numpy as np
from tools import *
import time
# 引入 FigureCanvasAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
# 引入 Image
import PIL.Image as Image


"""
构建训练集
"""

if __name__ == "__main__":
    set_gpu("0")
    from pylab import mpl
    rho = [200, 20, 100, 10, 100]  # 电阻率模型参数
    thi = [1820, 1900, 4535, 2150]  # 厚度模型参数
    res = []
    for i in range(len(rho) - 1):
        for j in range(int(thi[i])):
            res.append(rho[i])
    for i in range(2000):
        res.append(rho[-1])

    T = np.logspace(-3, 3, 15)
    guance = MT1D(T, rho, thi)
    Mes = [T, guance]  # 厚度，分辨率,约束
    Mes1 = {"zhenze": 0.01, "zaoshen": 0.02}
    step = environment(Mes, Mes1)
    mpl.rcParams['font.sans-serif'] = ['SimHei']


    R_P = np.load('res/learining_0.0013_pra_1000_rho20cen/R_P.npy', allow_pickle=True)
    plt.plot(range(len(R_P)), R_P)
    plt.show()
    meanR = np.load('res/learining_0.0013_pra_1000_rho20cen/meanR.npy', allow_pickle=True)
    meanQ = np.load('res/learining_0.0013_pra_1000_rho20cen/updates.npy')
    res= np.load('res/learining_0.0013_pra_1000_rho20cen/res.npy', allow_pickle=True)[80:100]
    plt.plot(range(len(meanR)), meanR)
    plt.figure()
    plt.plot(range(len(meanQ)), meanQ)
    plt.show()
    plt.figure()


    preres = []
    for i in range(len(rho) - 1):
        for j in range(int(thi[i])):
            preres.append(rho[i])
    for i in range(2000):
        preres.append(rho[-1])
    plt.semilogx(preres, range(len(preres)),alpha=0.5)  # 预测电阻率

    reslist=[]
    for i in range(len(res)):
        res1=[]
        rho=10**res[i][0]
        thi=10**res[i][1]
        for j in range(len(rho) - 1):
            for k in range(int(thi[j])):
                res1.append(rho[j])
        for j in range(2000):
                res1.append(rho[-1])
        reslist.append(res1)
        plt.semilogx(res1, range(len(res1)),alpha=0.01)  # 真实电阻率
   # plt.show()
    lenmax=max(reslist,key=lambda v: len(v))
    meanmodel=[]
    for j in range(len(lenmax)):
        lislist = {}
        for i in range(len(reslist)):
            rho = reslist[i]
            try:
                lislist[int(rho[j])] = lislist.get(int(rho[j]), 0) + 1
            except:
                pass
        zhong = max(lislist, key=lislist.get)
        meanmodel.append(zhong)
    plt.semilogx(meanmodel, range(len(meanmodel)), alpha=1)

    ax = plt.gca()
    ax.set_ylim((0, 20000))
    ax.invert_yaxis()  # y轴反向
    plt.xlim((0, 1000))
    ax.set_yticks([0, 5000, 10000, 15000, 20000], [0, 5, 10, 15, 20])
    plt.show()