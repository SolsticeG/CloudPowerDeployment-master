# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:38:21 2022

@author: gxq
"""

import sys
import os
import numpy as np
import collections
import random
import torch

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
    
class fileUtil:
    """
        需要保存的内容
        return_list:每个episode的reward(npy)
        episode_return:当前episode积累的reward(txt)
        id_episode:当前episode的索引(txt)
        done:当前episode完成标识
        count_target:目标网络参数更新计数器
        state:当前状态
        action_list:当前episode采取的动作队列(本程序里主要用于同步env)
        replaybuffer:(txt)
        q_net.pkl:q网络参数
        target_q_net.pkl:目标网络参数
    """
    def __init__(self):
        pass
    
    def readFiles(self,buffer_size):
        dirs = os.listdir(filePath)
        parameterFile = 'parameters.txt'

        call_count = 0  # python程序调用次数
        id_episode = 0  #episode序号
        episode_return = 0.0  #episode累计奖励
        done = False  #episode结束标志
        count_target = 0  #目标网络更新计数器

        if parameterFile in dirs:
            file_read = open(os.path.join(filePath,parameterFile), "r")
            call_count=int(file_read.readline())
            id_episode = int(file_read.readline())
            episode_return = float(file_read.readline())
            done=file_read.readline()
            if done=='False\n':
                done=False
            else:
                done=True
            count_target=int(file_read.readline())

        return_list = np.array([])#存储每个episode的总reward
        rewardListFile = 'returnList.npy'
        if rewardListFile in dirs:
            return_list = np.load(os.path.join(filePath,rewardListFile))
        return_list=list(return_list)

        action_list = np.array([])  # 存储每个episode的总reward
        actionListFile = 'actionList.npy'
        if actionListFile in dirs:
            action_list = np.load(os.path.join(filePath,actionListFile))
        action_list = list(action_list)


        state = None
        stateFile = 'state.npy'
        if stateFile in dirs:
            state = np.load(os.path.join(filePath,stateFile))

        replay_buffer=ReplayBuffer(buffer_size)

        replaybuffer_stateFile='replaybuffer_state.npy'
        replaybuffer_actionFile = 'replaybuffer_action.npy'
        replaybuffer_rewardFile = 'replaybuffer_reward.npy'
        replaybuffer_nextstateFile = 'replaybuffer_nextstate.npy'
        replaybuffer_doneFile = 'replaybuffer_done.npy'

        dirs = os.listdir(filePath)

        if replaybuffer_stateFile in dirs:
            rb_state=list(np.load(os.path.join(filePath,replaybuffer_stateFile)))
            rb_action=list(np.load(os.path.join(filePath,replaybuffer_actionFile)))
            rb_reward=list(np.load(os.path.join(filePath,replaybuffer_rewardFile)))
            rb_nextstate=list(np.load(os.path.join(filePath,replaybuffer_nextstateFile),allow_pickle=True))
            rb_done=list(np.load(os.path.join(filePath,replaybuffer_doneFile)))
            for i in range(len(rb_state)):
                replay_buffer.add(rb_state[i],rb_action[i],rb_reward[i],rb_nextstate[i],rb_done[i])
        return call_count,id_episode,episode_return,done,count_target,return_list,state,action_list,replay_buffer


    def writeFiles(self,call_count,id_episode,episode_return,return_list,done,count_target,state,action_list,replaybuffer):
        parameterFile = 'parameters.txt'
        file_write = open(os.path.join(filePath,parameterFile), "w")
        file_write.write(str(call_count)+'\n')
        file_write.write(str(id_episode)+'\n')
        file_write.write(str(episode_return)+'\n')
        file_write.write(str(done) + '\n')
        file_write.write(str(count_target) + '\n')

        rewardListFile = 'returnList.npy'
        np.save(os.path.join(filePath,rewardListFile), return_list)

        stateFile = 'state.npy'
        np.save(os.path.join(filePath,stateFile), np.array(state))

        actionFile = 'actionList.npy'
        np.save(os.path.join(filePath,actionFile), np.array(action_list))

        replaybuffer_stateFile = 'replaybuffer_state.npy'
        replaybuffer_actionFile = 'replaybuffer_action.npy'
        replaybuffer_rewardFile = 'replaybuffer_reward.npy'
        replaybuffer_nextstateFile = 'replaybuffer_nextstate.npy'
        replaybuffer_doneFile = 'replaybuffer_done.npy'
        rb_state=[]
        rb_action=[]
        rb_reward=[]
        rb_nextstate=[]
        rb_done=[]
        for i in range(replaybuffer.size()):
            transition=replaybuffer.buffer.pop()
            rb_state.append(transition[0])
            rb_action.append(transition[1])
            rb_reward.append(transition[2])
            rb_nextstate.append(transition[3])
            rb_done.append(transition[4])
        np.save(os.path.join(filePath,replaybuffer_stateFile),np.array(rb_state))
        np.save(os.path.join(filePath,replaybuffer_actionFile), np.array(rb_action))
        np.save(os.path.join(filePath,replaybuffer_rewardFile), np.array(rb_reward))
        np.save(os.path.join(filePath,replaybuffer_nextstateFile), np.array(rb_nextstate))
        np.save(os.path.join(filePath,replaybuffer_doneFile), np.array(rb_done))

class getStateUtil:
    def __init__(self):
        pass
    
    def getState(self):
        list_str = []
        vmResources = []
        hostInfo = []
        for i in range(1, len(sys.argv)):
            list_str.append(sys.argv[i].replace(",", ""))
        list_str[0] = list_str[0].replace("[", "")
        list_str[len(sys.argv) - 2] = list_str[len(sys.argv) - 2].replace("]", "")
        temp = ','.join(list_str)
        arr = temp.split('][')
        hostInfo = arr[0].split(',')
        hostInfo = list(map(float, hostInfo))
        vmResources = arr[1].split(',')
        vmResources = list(map(float, vmResources))
        #vmResources = vmResources[1:]
        state=vmResources+hostInfo
        #print(state)
        return np.array(state,dtype=np.float32)

if __name__ == '__main__':

    Util=getStateUtil()
    Util.getState()
    #state=Util.getCurrentState()
    #print(state)



