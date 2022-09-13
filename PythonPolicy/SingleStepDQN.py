from myUtils import getStateUtil,fileUtil
import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import os
from environment import PowerDeploymentEnv
import json

filePath = os.path.dirname(os.path.realpath(__file__)) 

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,epsilon, target_update,device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

        dirs = os.listdir(filePath)
        netFile = 'q_net.pkl'
        if netFile in dirs:
            self.q_net = torch.load(os.path.join(filePath,'q_net.pkl'))
            self.target_q_net = torch.load(os.path.join(filePath,'target_q_net.pkl'))

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        torch.save(self.q_net, os.path.join(filePath,'q_net.pkl'))
        torch.save(self.target_q_net, os.path.join(filePath,'target_q_net.pkl'))
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1
        torch.save(self.q_net, os.path.join(filePath,'q_net.pkl'))
        torch.save(self.target_q_net, os.path.join(filePath,'target_q_net.pkl'))
        
"""
    重写一下gym环境(主要由于gym环境的reset问题)
"""
class myWrapper(gym.Wrapper):
    def __init__(self, env,action_list):
        gym.Wrapper.__init__(self, env)
        self.action_list=action_list

    def reset(self, addition=None, **kwargs):
        obs_n = self.env.reset(**kwargs)
        #print(obs_n)
        #print(self.action_list)
        if len(self.action_list)!=0:
            for action in self.action_list:
                obs_n,_,_,_=self.env.step(action)
        return obs_n

    def step(self, actions):
        obs_n, reward_n, done_n, info_n = self.env.step(actions)
        return obs_n, reward_n, done_n, info_n


print("**********Singlestep Dqn*******************")
json_path = os.path.join(filePath,"para.json")
with open(json_path,'r') as f:
    para_dict = json.load(f)
lr = para_dict['lr']
num_episodes = para_dict['num_episodes']
hidden_dim = para_dict['hidden_dim']
gamma = para_dict['gamma']
epsilon = para_dict['epsilon']
target_update = para_dict['target_update']
buffer_size = para_dict['buffer_size']
minimal_size = para_dict['minimal_size']
batch_size = para_dict['batch_size']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

replay_buffer = ReplayBuffer(buffer_size)
fileUtil = fileUtil()
call_count,id_episode,episode_return,done,count_target,return_list,state,action_list,replay_buffer\
 = fileUtil.readFiles(buffer_size) #读取参数

env = PowerDeploymentEnv(replay_buffer)

random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

agent.count=count_target
#print(id_episode,episode_return,done,count_target,return_list,state,action_list,replay_buffer.buffer)
call_count+=1


if not done:
    print("******call count:",call_count)
    print("******episode_return:",episode_return)
    #print(replay_buffer.buffer)
    state = env.reset()
    print("********current_state:",state)
    cpuUtilization_list = []

    action = agent.take_action(state)
    print("******action:",action)
    action_list.append(action)
    next_state, reward, done, _ ,episode_return= env.step(action,episode_return)
    replay_buffer.add(state, action, reward, next_state, done)
    state = next_state
    episode_return += reward
    # 当buffer数据的数量超过一定值后,才进行Q网络训练
    if replay_buffer.size() > minimal_size:
        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
        transition_dict = {
            'states': b_s,
            'actions': b_a,
            'next_states': b_ns,
            'rewards': b_r,
            'dones': b_d
        }
        agent.update(transition_dict)
    #print(id_episode, episode_return, done, count_target, return_list, state,action_list,replay_buffer.buffer)
    #print(replay_buffer.buffer)
else:
    return_list.append(episode_return)
    id_episode+=1
    episode_return=0.0
    state=env.reset()
    done=False
    action_list=[]
    
fileUtil.writeFiles(call_count,id_episode,episode_return,return_list,done,agent.count,state,action_list,replay_buffer)


