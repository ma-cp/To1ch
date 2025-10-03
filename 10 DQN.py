# DQN的历史沿革
# Q-learning本身是一种强化学习，它的主要优势是“保证收敛”，Q 是 Quality（质量）的意思
# DQN是Q-learning在深度神经网络方法下的应用
# 为什么这个文件不适用notebook了？ 因为notebook对gym的支持不够简单

# 训练流程如下：
# Q选择动作-->积累经验-->（Q选择动作，积累经验到一定阈值）-->学习
# 学习-->Q和目标计算器分别计算价值得到损失-->从损失更新Q-->（学习，定期更新目标计算器）
# 1 用 Q 网络选择动作并与环境交互
# 2 存储经验到回放缓冲区
# 3 从缓冲区采样批量数据
# 4 用 Q 网络计算当前 Q 值
# 5 用目标 Q 网络计算目标 Q 值
# 6 计算损失并更新 Q 网络参数
# 7 定期将 Q 网络参数复制到目标 Q 网络

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 50   # target update frequency
MEMORY_CAPACITY = 10


# https://gymnasium.org.cn/api/env/#gymnasium.Env.render
env = gym.make('CartPole-v1', render_mode="human") # 从gym引入一个cartpole模型
env = env.unwrapped # 解除env的封装
# action_space 对应有效动作的 Space 对象 返回 Discrete() 类型的对象
# Discrete() 参见 https://gymnasium.org.cn/api/spaces/fundamental/#gymnasium.spaces.Discrete
N_ACTIONS = env.action_space.n # Discrete().n 返回 此空间的元素数量
# observation_space 对应有效观察的 Space 对象
# https://gymnasium.org.cn/api/spaces/#gymnasium.spaces.Space
N_STATES = env.observation_space.shape[0] # shape 会返回一个只有一个元素的元组 使用[0]将其转化成int

# DQN的神经网络主体是线性的，甚至隐藏层并不需要很多
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        # eval_net 和 target_net 分别是做什么的？
        # 两个神经网络是为了解决目标值不稳定的问题。如果只用一个神经网络会出现Q值不断变化的情况。
        # eval 用来确定当前情况下各个动作的Q值
        # target 用来计算“目标”Q值
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            # print(torch.max(actions_value, 1))
            action = torch.max(actions_value, 1)[1].data.numpy()[0]     # return the argmax
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].unsqueeze(1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
ep_r_his = []
for i_episode in range(4000):
    s = env.reset()[0]
    ep_r = 0
    while True:
        env.render()

        # s:([action1, action2, ...], {info}) ，s[0]意味着只传入action数组
        a = dqn.choose_action(s)
        # break

        # take action
        s_, r, done, _, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                ep_r_his.append(ep_r)
                if i_episode%MEMORY_CAPACITY == 0:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2), end=", ")
                    if  len(ep_r_his) != 0 :
                        print(f"Last average reward is:{sum(ep_r_his) / len(ep_r_his)}")
                        ep_r_his = []
            if dqn.learn_step_counter % (TARGET_REPLACE_ITER*100) == 0:
                print(f"memory updated {dqn.learn_step_counter / TARGET_REPLACE_ITER}times,Ep {i_episode}")



        if done:
            break
        s = s_
