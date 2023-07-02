from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
TARGET_UPDATE_FREQ=1000
LEARNING_RATE = 5e-4
NUM_ENV = 4

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim = 1)[0]
        action = max_q_index.detach().item()
        return action


env = gym.make("CartPole-v1")
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0,0], maxlen=100)
episode_reward = 0.0
online_net = Network(env)
target_net = Network(env)
target_net.load_state_dict(online_net.state_dict())
optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

#inicijalizacija replay bufera
obs = env.reset() #dovijamo prvu observaciju
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, rew, done, _, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    
    replay_buffer.append(transition)
    obs = new_obs
    if done:
        obs = env.reset()
obs = env.reset()
for step in itertools.count(): # seksi while petlja
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])  # seksi smanjivanje epsilona
    rnd_sample = random.random()
    if rnd_sample <= epsilon or len(obs) == 2:
        a = random.randint(0, 1)
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)
    new_obs, rew, done, _, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs
    episode_reward += rew
    if done:
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0
    
    #pocetak gradijentnog
    transitions = random.sample(replay_buffer, BATCH_SIZE)
    filthered_obs = []
    filthered_new_obs = []
    for t in transitions:
        if type(t[0]) is tuple:
            filthered_obs.append(t[0][0])
        else:
            filthered_obs.append(t[0])
        if type(t[4]) is tuple:
            filthered_new_obs.append(t[4][0])
        else:
            filthered_new_obs.append(t[4])

    obses = np.array(filthered_obs, dtype=np.float32)
    actions = np.asarray([t[1] for t in transitions], dtype=np.int64)
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.array([t[4] for t in transitions], dtype=np.float32)

    obses_t = torch.from_numpy(obses)
    actions_t = torch.from_numpy(actions).unsqueeze(-1)
    rews_t = torch.from_numpy(rews).unsqueeze(-1)
    dones_t = torch.from_numpy(dones).unsqueeze(-1)
    new_obses_t = torch.from_numpy(new_obses)

    #ciljevi
    targer_q_values = target_net(new_obses_t)
    max_target_q_values = targer_q_values.max(dim=1, keepdim=True)[0]
    
    targets = rews_t + GAMMA * max_target_q_values

    #loss
    q_values = online_net(obses_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    #decent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #azururanje mreze
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    #logovi
    if step % 1000 == 0:
        print()
        print("Korak", step)
        print("Prosek nagrada", np.mean(rew_buffer))