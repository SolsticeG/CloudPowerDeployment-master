import gym
from gym import spaces
import numpy as np
from myUtils import getStateUtil

class PowerDeploymentEnv(gym.Env):
    def __init__(self,replaybuffer):
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0,high=np.finfo(np.float32).max,shape=(45,),dtype=np.float32)
        self.replaybuffer = replaybuffer

    def step(self, action,episode_return):#由于要加上上一个step的reward值所以传参episode_return
        StateUtil = getStateUtil()
        state=StateUtil.getState()
        if self.replaybuffer.size() > 0:
            transition=self.replaybuffer.buffer.pop()#更改replaybuffer里的上一条transition的next_state以及reward
            #计算奖励(cpu利用率方差差)
            last_state=transition[0]
            print("********last state:",last_state)
            cpuUtilization_old=[last_state[9],last_state[14],last_state[19],last_state[24],last_state[29],last_state[34],last_state[39],last_state[44]]
            cpuUtilization_new=[state[9],state[14],state[19],state[24],state[29],state[34],state[39],state[44]]
            reward=-(np.std(cpuUtilization_new)-np.std(cpuUtilization_old))
            print(cpuUtilization_old,cpuUtilization_new,reward)
            self.replaybuffer.add(transition[0],transition[1],reward,state,transition[4])
            episode_return+=reward
        #当前step的next_state与reward分别用0代替
        next_state = np.zeros((1,45),dtype=np.float32)
        done = False
        reward= 0.0
        info = {}
        return next_state, reward, done, info,episode_return

    def reset(self):
        StateUtil = getStateUtil()
        return StateUtil.getState()

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass


