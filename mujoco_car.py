# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 22:57:02 2026

@author: yzyja
"""

import mujoco
import mujoco.viewer
import torch
import numpy as np
import time
from collections import namedtuple, deque
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.distributions import MultivariateNormal

import torch.nn as nn
# Load a model (replace with your model path)
# m = mujoco.MjModel.from_xml_path('C:/Users/yzyja/Mujoco/model/humanoid/humanoid.xml')
# d = mujoco.MjData(m)

# # Launch the viewer as a context manager
# with mujoco.viewer.launch_passive(m, d) as viewer:
#   # Simulate and sync the viewer
#   while viewer.is_running():
#     mujoco.mj_step(m, d)
#     viewer.sync()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory,batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class Actor(nn.Module):
    
    def __init__(self, state_dim, control_dim):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,control_dim)
            )
        
        self.sigma = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, control_dim)
            )
        self.optim = optim.Adam(self.network.parameters(), lr = 0.0003)
    def forward(self, state):
        mu = self.mu(state)
        sigma = self.sigma(state)
        z = torch.normal(0,1, size=mu.shape)
        control = mu + z * sigma
        dist = MultivariateNormal(0,1)
        
        return control, dist.log_prob(z)

class Value(nn.Module):
    
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
            )
        
        self.optim = optim.Adam(self.network.parameters(), lr = 0.0003)
        
    
    def forward(self, state):
        
        
        return self.network(state)
    

class Cars():
    def __init__(self, max_steps = 3*500, seed=0,render=False):
     self.model = mujoco.MjModel.from_xml_path('C:/Users/yzyja/Mujoco/model/car/car.xml')
     self.data = mujoco.MjData(self.model)
     self.duration = int(max_steps//500)
     self.single_action_space = 2
     self.single_observation_space = 13
     self.viewer = None
     self.reset()
     if render:
         self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
     
    def reset(self):
         mujoco.mj_resetData(self.model, self.data)
         self.episodic_return = 0
         state = np.hstack((self.data.body('car').xpos[:3],
                             self.data.body('car').cvel, 
                             self.data.body('car').xquat))
         if self.viewer is not None:
             self.viewer.close()
             self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
         return state
     
    def reward(self, state, action):
         car_dist = (np.linalg.norm(np.array([-1,4]-state[:2])))
         return np.exp(-((car_dist)))
    
    def render(self):
         if self.viewer.is_running():
             self.viewer.sync()
    
    def close(self):
        self.viewer.close()
     
    def step(self, action):
         self.data.ctrl = np.tanh(action)
         mujoco.mj_step(self.model, self.data)
         
         state = np.hstack((self.data.body('car').xpos[:3], 
                             self.data.body('car').cvel, 
                             self.data.body('car').xquat))
         reward = self.reward(state, np.tanh(action))
         self.episodic_return += reward
         done = False
         info = {}
         if self.data.time>=self.duration:
             done = True
             info.update({'episode':{'r':self.episodic_return,'l':self.data.time}})
             info.update({"terminal_observation":state.copy()})
             state = self.reset()
         return state, reward, done, info

def main():
    duration = 50
    env = Cars(max_steps=duration*500,render=True)
    
    
    n_action = env.single_action_space
    n_obs = env.single_observation_space
    
    
    
    
    #2000000 is the training iterations
    policy = Actor(n_obs,n_action)
    value = Value(n_obs)
    
    
    
    state = env.reset()
    start = time.time()
    env.render()
    
    cum_return = 0
    
    batch_state = []
    batch_action = []
    batch_reward = []
    while time.time() - start < duration:
        
        with torch.no_grad():
            action, log_prob = policy(torch.Tensor(state).to(device)).cpu().numpy()[:2]
            state, reward, done, info = env.step(action)
            
            cum_return = gamma * cum_return + reward
        if done:
            break
        time.sleep(0.003)
    
    env.close()
    
def roll_out_trajectories(time_horizon, num_rollouts):
    state_batch = torch.empty((num_rollouts, time_horizon, state_dim))
    action_batch = torch.empty((num_rollouts,time_horizon, control_dim))
    reward_batch = torch.empty((num_rollouts,time_horizon, 1))
    return_batch = torch.empty((num_rollouts,time_horizon, 1))
    
    for n in range(num_rollouts):
        state = env.reset()
        for t in range(time_horizon):
            action, log_prob = policy(torch.Tensor(state).to(device)).cpu().numpy()[:2]
            next_state, reward, done, info = env.step(action)
            
            state_batch[n,t,:] = state
            action_batch[n,t,:] = action
            reward_batch[n,t,:] = reward
    
    return_batch[:, -1] = reward_batch[:,-1].copy()
    for t in range(time_horizon-2,-1,-1):
        return_batch[:, t] = reward_batch + gamma * return_batch[:,t+1]
    
    return state_batch, action_batch, reward_batch, return_batch
            

    
def compute_advantage(return_batch, value_fn, state_batch):
    
    advantage_batch = return_batch - value_fn(state_batch)
    return advantage_batch
    
def train_ppo(env):
    #Rollout trajectories
    roll_out_trajectories(time_horizon, num_rollouts)
    
    #get state, action, reward
    #get return
    #Get advantage
    
    #Compute loss
    #   actor loss
    #       policy ratio * advantage, clip
    #   value function loss
    #       cumulative reward vs value(state)
    #   exploration loss
    #Update
    
    
    #
    
    
    
    
if __name__ =='__main__':
    
    #Training parameter setup
    duration = 5
    gamma = 0.99
    
    #Environment initialization
    env = Cars(max_steps=duration*500,render=True)
    
    
    
    #Policy initialization
    n_action = env.single_action_space
    n_obs = env.single_observation_space
    
    policy = Actor(n_obs,n_action)
    value = Value(n_obs)
    
    
    #Policy training
    train_ppo()
    
    #Policy testing
    state = env.reset()
    start = time.time()
    
    while time.time() - start < duration:
        env.render()
        with torch.no_grad():
            action = policy(torch.Tensor(state).to(device)).cpu().numpy()[:2]+np.array([1,0])
            state, reward, done, info = env.step(action)
            
            
            
            
        
    
    # env.close()