#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:23:30 2026

@author: zyuan
"""

import torch
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib
import random
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple, deque
from itertools import count
try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    try:
        import imageio
        IMAGEIO_AVAILABLE = True
    except ImportError:
        IMAGEIO_AVAILABLE = False
        print("Warning: imageio not available. Install with 'pip install imageio' for GIF creation.")
env = gym.make("CartPole-v1")

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


class DQN(nn.Module):
    
    def __init__(self, n_observation, n_actions, n_hidden=128):
        super(DQN,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_observation, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,n_actions)            
            )
    
    def forward(self,x):
        return self.network(x)
    
    
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 0.0003

n_actions = env.action_space.n
state, info = env.reset()
n_observation  = len(state)

policy_net = DQN(n_observation, n_actions).to(device)
target_net = DQN(n_observation, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 *steps_done / EPS_DECAY)
    
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]], device = device)

# Rollout env based on DQN and collect trajectories
## Select action based on DQN 

# Update the DQN using new trajectories

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
def optimize_model():
    
    if len(memory)<BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)), 
                    device = device, 
                    dtype = torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1,action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step()
    
if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i in tqdm.tqdm(range(num_episodes)):
    state, info = env.reset()
    state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device = device)
        done = terminated or truncated
        
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype = torch.float32, device = device).unsqueeze(0)
        
        memory.push(state, action, next_state,reward)
        
        state = next_state
        
        optimize_model()
        
        # Soft update target network (vectorized, no for-loop)
        with torch.no_grad():
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.mul_(1 - TAU).add_(policy_param.data, alpha=TAU)
        
        if done:
            episode_durations.append(t+1)
            break

print('Complete')
plot_durations(show_result=True)
plt.savefig('results.png')
plt.ioff()
plt.show()

# Generate animation of a rollout
def create_rollout_animation(num_episodes=1, max_steps=500, filename='cartpole_rollout.gif'):
    """Create an animation showing the trained agent's rollout"""
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            print("Cannot create animation: imageio not installed. Install with 'pip install imageio'")
            return
    
    # Create environment with rendering enabled
    render_env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames = []
    
    print(f"\nGenerating rollout animation ({num_episodes} episode(s))...")
    
    state, info = render_env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Record initial frame
    frame = render_env.render()
    frames.append(frame)
    
    for step in tqdm.tqdm(range(max_steps),desc="Generating rollout animation"):
        # Use trained policy (no exploration)
        with torch.no_grad():
            action = policy_net(state_tensor).max(1).indices.view(1, 1)
        
        observation, reward, terminated, truncated, _ = render_env.step(action.item())
        frame = render_env.render()
        frames.append(frame)
        
        if terminated or truncated:
            break
        
        state_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    
    render_env.close()
    
    # Save as GIF
    if frames:
        print(f"Saving animation to {filename}...")
        imageio.mimsave(filename, frames, fps=30, loop=0)
        print(f"Animation saved successfully! ({len(frames)} frames)")
    else:
        print("No frames captured!")

# Create the animation after training
create_rollout_animation(num_episodes=1, filename='cartpole_rollout.gif')