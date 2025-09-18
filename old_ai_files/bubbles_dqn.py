import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# --- Hyperparameters and Explanations ---
# Learning rate: How fast the network learns. 1e-4 is a good starting point for DQN with Adam optimizer.
LEARNING_RATE = 1e-4
# Discount factor: How much future rewards are valued. 0.99 encourages long-term planning.
GAMMA = 0.99
# Replay buffer size: How many past experiences to store. 100,000 is typical for stability.
REPLAY_BUFFER_SIZE = 100_000
# Batch size: Number of samples per training step. 64 is a good tradeoff between stability and speed.
BATCH_SIZE = 64
# Epsilon parameters for exploration
EPS_START = 1.0  # Start fully random
EPS_END = 0.05   # Minimum exploration
EPS_DECAY = 100_000  # Decay over 100k steps
# Target network update frequency
TARGET_UPDATE_FREQ = 1000  # Steps between target net updates

# --- Neural Network for Q-value approximation ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # Simple 3-layer MLP
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Experience Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.steps_done = 0
        self.epsilon = EPS_START

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy policy.
        With probability epsilon, choose a random action (exploration).
        Otherwise, choose the action with the highest Q-value (exploitation).
        """
        self.steps_done += 1
        self.epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * self.steps_done / EPS_DECAY)
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer for experience replay.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        Samples a batch from the replay buffer and performs a DQN update step.
        """
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0]
            target = reward + GAMMA * next_q_values * (1 - done)
        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """
        Copies the weights from the policy network to the target network.
        This is done periodically for stable learning.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """
        Saves the policy network to a file.
        """
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """
        Loads the policy network from a file.
        """
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) 