import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from random import sample

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        # self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        # self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))

        # self.fc1 = nn.Sequential(nn.Linear(self.fc1_dims, self.fc2_dims), nn.ReLU(inplace=True))
        # self.fc2 = nn.Linear(self.fc2_dims, n_actions)
        self.fc1 = nn.Sequential(nn.Linear(*self.input_dims, self.fc1_dims), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(self.fc1_dims, self.fc2_dims), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(self.fc2_dims, self.n_actions), nn.ReLU(inplace=True))

        self._create_weights()
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else "cpu")
        self.to(self.device)
        # self._create_weights()
    
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        # output = self.conv1(state)
        # output = self.conv2(output)
        # output = self.conv3(output)
        # output = output.view(output.size(0), -1)
        output = self.fc1(state)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
    
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_siye, n_actions,
                 max_mem_size=100000, min_eps = 0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [elem for elem in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_siye
        self.mem_counter = 0
        self.replay_memory = []

        self.Q_eval = DeepQNetwork(lr=self.lr, input_dims=input_dims, 
                                   fc1_dims=7*7*64, fc2_dims=512, n_actions=n_actions)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)


    def save_gotten_information(self, state, action, reward, state_, done):
        # if len(self.replay_memory) < self.batch_size:
        #     return #imamo previse nula u tabeli da bi ista naucili
        # print("popunili buffer")
        index = self.mem_counter % self.mem_size
        self.mem_counter += 1

        self.new_state_memory[index] = state_

        self.replay_memory.append([state, action, reward, state_, done])
        if len(self.replay_memory) > self.mem_size:
            del self.replay_memory[0]
        batch = sample(self.replay_memory, min(len(self.replay_memory), self.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
        print("AAAAAAAAAAAAAAA")
        print("state batch")
        print(state_batch)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.cat(tuple(state for state in state_batch))
        action_batch = T.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = T.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = T.cat(tuple(state for state in next_state_batch))

        if T.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        current_prediction_batch = self.Q_eval(state) #pucaa
        next_prediction_batch = self.Q_eval(state_)

        y_batch = T.cat(
            tuple(reward if terminal else reward + self.gamma * T.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)  
        q_target = reward_batch + self.gamma + T.max(q_next, dim = 1)[0]
      
        self.Q_eval.optimizer.zero_grad()
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.epsilon -= self.eps_dec
        if self.epsilon < self.min_eps:
            self.epsilon = self.min_eps

    def chose_action(self, observation):
        if np.random.random() <= self.epsilon:
            action = np.random.choice(self.action_space)
            return action
        # state = T.tensor([observation]).to(self.Q_eval.device)
        # actions = self.Q_eval.forward(state)
        # print(actions)
        # action = T.argmax(actions).item()
        prediction = self.Q_eval(observation)[0]
        action = T.argmax(prediction).item()
        return action
    
    