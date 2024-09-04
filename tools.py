def set_ch():
    from pylab import mpl
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
def set_gpu(num:str):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = num
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
def plt_struct_data(rho, thi,prethi=1000):
    res = []
    for i in range(len(rho) - 1):
        for j in range(int(thi[i])):
            res.append(rho[i])
    if prethi>len(res):
        for i in range(prethi-len(res)):
            res.append(rho[-1])
    return res
def plt_struct_yinyin(rho, thi):
    dotlist=[]
    t=0
    for i in range(len(rho) - 1):
        dot=[rho[i],t,t+thi[i]]
        dotlist.append(dot)
        t=t+thi[i]
    dot=[rho[-1],t,t+10000]
    dotlist.append(dot)
    return dotlist
def plt_data(data, flag, islabel, fig):
    if flag == 0:
        if islabel:
            fig.semilogx(data, range(len(data)), color='r', label="true")
        else:
            fig.semilogx(data, range(len(data)), color='k', alpha=0.01)
    if flag == 1:
        if islabel:
            fig.scatter(range(len(data)), data, color='b', label="true")
            fig.set_xlabel("Frequency")
            fig.set_title("Apparent resistivity")
        else:
            fig.plot(range(len(data)), data, color='r', label="pre", linestyle='--')
    if flag == 2:
        fig.plot(range(len(data)), data, color='b', label="reward")
        fig.set_title("sum reward")
    if flag == 3:
        fig.plot(range(len(data)), data, color='b', label="loss")
        fig.set_title("loss")
def save_data(data, path):
    import numpy as np
    data_array = np.array(data)
    np.save(path, data_array)
def smooth(data, sm=1):
    import numpy as np
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm)*1.0
            d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data
