"""
    构建训练集
"""
import matplotlib.pyplot as plt
import numpy as np
def MT1D(T, rho, h):
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
def plt_struct_data(rho, thi):
    res = []
    for i in range(len(rho) - 1):
        for j in range(int(thi[i] * 0.05)):
            res.append(rho[i])
    for i in range(int(2000 * 0.1)):
        res.append(rho[-1])
    return res
def drawres(rho, axs):
    thi = [600,672,752,842,944,1057,1184,1326,1485,1663,1863,2087,2337,2618,2932,3284,3678,4119,4613]
    preres = plt_struct_data(rho, thi)
    axs.semilogx(preres, range(len(preres)), alpha=1)

T = np.logspace(-3, 3, 15)  # 频点（0.001到1000等间隔采样15个点）
rethi = [600,672,752,842,944,1057,1184,1326,1485,1663,1863,2087,2337,2618,2932,3284,3678,4119,4613]


threelayers=[[[800],[5]],[[80],[2]]]
fivelayers=[[[500,100,600],[2,2,2]],[[100,600,200],[2,2,2]]]
eighlayers=[[[300,500,800,200,300,500],[2,2,2,2,2,2]],[[800,500,100,200,600,500],[2,2,2,2,2,2]]]
#定义层数和类别，按照类别来抽
#对比实验的类型都学过。补充常见类型

trainset=[]




index=0
while(index<20):
    for i in range(len(threelayers)):
        rho = []
        thi = []
        for j in range(20):
            print(len(threelayers[i][1])*2+index)
            if j>=index and j<len(threelayers[i][1])*2+index:
                rho.append(threelayers[i][0][int((j-index)/2)])
            else:
                rho.append(200)
    trainset.append([rho,rethi])
    index=index+1


index=0
while(index<20):
    for i in range(len(fivelayers)):
        rho = []
        thi = []
        for j in range(20):
            print(len(fivelayers[i][1])*2+index)
            if j>=index and j<len(fivelayers[i][1])*2+index:
                rho.append(fivelayers[i][0][int((j-index)/2)])
            else:
                rho.append(200)
    trainset.append([rho,rethi])
    index=index+1
print(trainset)


index=0
while(index<20):
    for i in range(len(eighlayers)):
        rho = []
        thi = []
        for j in range(20):
            print(len(eighlayers[i][1])*2+index)
            if j>=index and j<len(eighlayers[i][1])*2+index:
                rho.append(eighlayers[i][0][int((j-index)/2)])
            else:
                rho.append(200)
    trainset.append([rho,rethi])
    index=index+1
print(trainset)
trainset = np.array(trainset, dtype='object')
np.save('trainset.npy', trainset)
rho = trainset[30][0]
thi = trainset[30][1]
# print("123")
fig, axs = plt.subplots(1, 1)
guance = MT1D(T, rho, thi)
drawres(rho, axs)
xtick = [1, 10, 100, 1000]
axs.set_xticks(xtick)
# rho = [400, 200, 200, 100, 100]  # 高低高低高
# drawres(rho, axs)
# guance1 = MT1D(T, rho, thi)
plt.figure()
plt.plot(range(len(guance[0])), guance[0])
plt.show()
# thi = [1820, 1900, 4535, 2150]  # >4000算厚层，
#
#
#
#
# #5层模型
# trainset = []
# #1层模型
# rho = [[100], []]
# trainset.append(rho)
# rho = [[800], []]
# trainset.append(rho)
# #3层模型
# rho = [[1000, 600, 900], [3810, 11665]]  # H
# trainset.append(rho)
# rho = [[800, 600, 1000], [8000, 4000]]  # H
# trainset.append(rho)
# rho = [[500, 800, 400], [5000, 2000]]  # K
# trainset.append(rho)
# rho = [[300, 600, 200], [2000, 20000]]  # K
# trainset.append(rho)
# rho = [[200, 500, 800], [1000, 20000]]  # A
# trainset.append(rho)
# rho = [[300, 700, 900], [2000, 2000]]  # A
# trainset.append(rho)
# rho = [[900, 600, 300], [1502, 3000]]  # Q
# trainset.append(rho)
# rho = [[300, 600, 200], [10000, 3810]]  # Q
# trainset.append(rho)
# #5层模型
# rho = [[100, 200, 600, 800, 1000], [1820, 1900, 4535, 2150]]  # AAA
# trainset.append(rho)
# rho = [[10, 40, 80, 200, 10], [1820, 1900, 4535, 2150]]  # AAK
# trainset.append(rho)
# rho = [[10, 40, 200, 100, 10], [1820, 1900, 4535, 2150]]  # AKQ
# trainset.append(rho)
# rho = [[10, 40, 200, 10, 400], [1100, 1585, 5570, 2150]]  # AKH
# trainset.append(rho)
# rho = [[10, 200, 10, 400, 200], [500, 2185, 2280, 5440]]  # KHK
# trainset.append(rho)
# rho = [[10, 200, 10, 400, 600], [1100, 3865, 1495, 3945]]  # KHA
# trainset.append(rho)
# rho = [[10, 600, 200, 100, 40], [1100, 3865, 1495, 3945]]  # KQQ
# trainset.append(rho)
# rho = [[10, 600, 200, 40, 400], [1100, 3865, 1495, 3945]]  # KQH
# trainset.append(rho)
# rho = [[400, 10, 400, 10, 200], [3720, 1245, 3290, 2150]]  # HKH
# trainset.append(rho)
# rho = [[200, 20, 400, 600, 800], [500, 2185, 2280, 5440]]  # HAA
# trainset.append(rho)
# rho = [[200, 20, 100, 600, 100], [3720, 1245, 3290, 2150]]  # HAK
# trainset.append(rho)
# rho = [[200, 20, 400, 100, 10], [3720, 1245, 3290, 2150]]  # HKQ
# trainset.append(rho)
# rho = [[600, 200, 100, 40, 10], [3720, 1245, 3290, 2150]]  # QQQ
# trainset.append(rho)
# rho = [[600, 200, 100, 40, 100], [1820, 1900, 4535, 2150]]  # QQH
# trainset.append(rho)
# rho = [[600, 200, 100, 200, 400], [500, 2185, 2280, 5440]]  # QHA
# trainset.append(rho)
# rho = [[600, 200, 100, 200, 100], [500, 2185, 2280, 5440]]  # QHK
# trainset.append(rho)
#
# trainset = np.array(trainset, dtype='object')
# np.save('trainset16(改).npy', trainset)
# trainset = np.load('trainset16(改).npy', allow_pickle=True)
# print(len(trainset))
# rho = trainset[8][0]
# thi = trainset[8][1]
# # print("123")
# fig, axs = plt.subplots(1, 1)
# guance = MT1D(T, rho, thi)
# drawres(rho, axs)
# xtick = [1, 10, 100, 1000]
# axs.set_xticks(xtick)
# # rho = [400, 200, 200, 100, 100]  # 高低高低高
# # drawres(rho, axs)
# # guance1 = MT1D(T, rho, thi)
# plt.figure()
# plt.plot(range(len(guance[0])), guance[0])
# plt.show()
