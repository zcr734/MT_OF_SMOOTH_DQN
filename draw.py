from matplotlib import pyplot as plt
import numpy as np
reward=np.load('res/DQN/meanR.npy')
plt.plot(range(len(reward)),reward)
reward=np.load('res/DQN_yu/meanR.npy')
plt.plot(range(len(reward)),reward)
plt.figure()
loss=np.load('res/DQN/lasterror.npy')
plt.plot(range(len(loss)),loss)
loss=np.load('res/DQN_yu/lasterror.npy')
plt.plot(range(len(loss)),loss)
plt.figure()
meanQ=np.load('res/DQN/meanQ.npy')
plt.plot(range(len(meanQ)),meanQ)
meanQ=np.load('res/DQN_yu/meanQ.npy')
plt.plot(range(len(meanQ)),meanQ)




plt.show()