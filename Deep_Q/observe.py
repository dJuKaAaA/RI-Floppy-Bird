import os, random
import pokusaj10
import numpy as np
import torch
from torch import nn
import itertools
import time

import pickle
from Deep_Q.game2 import skip_frames

from game import Game

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        in_features = int(np.prod((4,)))
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim = 1)[0]
        action = max_q_index.detach().item()
        return action
    
    def save(self, save_path):
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
        # params = self.state_dict().items()
        with open(save_path, "wb") as f:
            pickle.dump(params, f)
    
    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)
        
        with open(load_path, "rb") as f:
            params_numpy = pickle.load(f)
        params = {k: torch.from_numpy(v) for k, v, in params_numpy.items()}
        self.load_state_dict(params)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('device:', device)
env = Game()
episode_reward = 0.0
net= Network()

net.load('./Deep_Q/saved_networks/v1')

obs, _, _, _ = env.nextFrame(0)
print(type(obs[0]))
beginning_episode = True
for t in itertools.count():
    while True:

        action = net.act(obs)

        env.nextFrame(action)
        obs, _, done, _, _ = skip_frames(env, pokusaj10.FRAMES_SKIPED)

        env.render()
        if done: 
            env = Game()
            obs, _, _, _ = env.nextFrame(0)
            break