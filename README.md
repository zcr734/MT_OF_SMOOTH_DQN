# MT_OF_SMOOTH_DQN

​	this code use the reinforcement learning with regularization reward to inversion the MT resistivity paremeters.

## Structure

- res:the result of the experiment.
- saved_model:the saved model.
- algorithm.py:the forward function and' the dqn algorithm of update the network weight.
- dqn.py:the design of the Q-network.
- environment.py:the state updation of the agent an the reward function.
- replay_memory.py:storing the experience  of agent and  sampling.
- pratice.py:begin to train.
- config.py:set some parements of  initial model ，update frequency，batchsize and so on.

## Usage

1. set the  paremeters in config.py.
2. run the pratice.py.