import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import math

# Hyperparameters for target-based DQN
INITIAL_LEARNING_RATE = 2e-4  # Lowered per request
GAMMA = 0.75  # Keep
REPLAY_BUFFER_SIZE = 30_000
BATCH_SIZE = 64  # Per selected combo (C2)
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 4_000  # Slower decay for 10k-step runs
TARGET_UPDATE_FREQ = 2000  # Less frequent target updates per request

# Training stability hyperparameters
GRADIENT_CLIP_NORM = 1.0  # Maximum gradient norm for clipping
#HUBER_DELTA = 1.0  # Delta parameter for Huber loss

# Target-based action space
GRID_ROWS = 20
GRID_COLS = 35
TARGET_ACTION_SIZE = GRID_ROWS * GRID_COLS  # 700 possible target positions
# NEW: Color-based action space
COLOR_ACTION_SIZE = 6  # 6 actions (one per color)
BUBBLE_COLORS = 6
# NOTE: State size depends on the environment/game; keep the network flexible by not hard-coding here.

class SafeBatchNorm1d(nn.Module):
    """BatchNorm1d wrapper that handles single samples gracefully during training"""
    def __init__(self, num_features):
        super(SafeBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.num_features = num_features
    
    def forward(self, x):
        if x.size(0) == 1:
            # Single sample: use eval mode to avoid batch norm issues
            self.bn.eval()
            result = self.bn(x)
            self.bn.train()
            return result
        else:
            # Multiple samples: normal batch norm
            return self.bn(x)

class TargetDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=None, use_batchnorm=True, dropout=0.1):
        super(TargetDQN, self).__init__()
        if hidden_layers is None:
            # One hidden layer of width 512 per request
            hidden_layers = [512]
        layers = []
        in_dim = state_dim
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(SafeBatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout if i < len(hidden_layers) - 1 else min(dropout, 0.05)))
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # Handle single samples for batch normalization
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Safety check: if batch size is 1, we need to handle BatchNorm specially
        if x.size(0) == 1:
            # For single samples, temporarily disable batch norm updates
            self.net.eval()  # Set to eval mode to use running statistics
            result = self.net(x)
            self.net.train()  # Restore training mode
            return result
        
        return self.net(x)

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

class TargetDQNAgent:
    def __init__(self, state_dim, action_dim, device='cpu',
                 learning_rate: float = None,
                 gamma: float = None,
                 batch_size: int = None,
                 eps_start: float = None,
                 eps_end: float = None,
                 eps_decay: int = None,
                 target_update_freq: int = None,
                 grad_clip_norm: float = None,
                 hidden_layers=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        # Hyperparameters (instance-specific overrides)
        self.gamma = gamma if gamma is not None else GAMMA
        self.batch_size = batch_size if batch_size is not None else BATCH_SIZE
        self.eps_start = eps_start if eps_start is not None else EPS_START
        self.eps_end = eps_end if eps_end is not None else EPS_END
        self.eps_decay = eps_decay if eps_decay is not None else EPS_DECAY
        self.target_update_freq = target_update_freq if target_update_freq is not None else TARGET_UPDATE_FREQ
        self.grad_clip_norm = grad_clip_norm if grad_clip_norm is not None else GRADIENT_CLIP_NORM
        
        # Regular DQN with enhanced architecture
        # - Batch normalization for training stability
        # - Dropout layers to prevent overfitting
        # - Huber loss instead of MSE for robustness
        # - Gradient clipping to prevent exploding gradients
        self.policy_net = TargetDQN(state_dim, action_dim, hidden_layers=hidden_layers).to(device)
        self.target_net = TargetDQN(state_dim, action_dim, hidden_layers=hidden_layers).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.initial_lr = learning_rate if learning_rate is not None else INITIAL_LEARNING_RATE
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.initial_lr)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.steps_done = 0
        self.epsilon = self.eps_start
        # Diagnostics for action selection
        self.last_selection_random = False
        self.last_selected_action = None
        self.last_epsilon = self.epsilon
    
    def set_evaluation_mode(self):
        """Set the policy network to evaluation mode (disable dropout and batch norm updates)"""
        self.policy_net.eval()
    
    def set_training_mode(self):
        """Set the policy network to training mode (enable dropout and batch norm updates)"""
        self.policy_net.train()
    
    def apply_action_mask(self, q_values, valid_targets):
        """
        Mask Q-values for invalid targets with -infinity.
        This ensures the AI automatically avoids unreachable positions.
        
        Args:
            q_values: Q-values tensor from policy network
            valid_targets: list of (row, col) tuples that are reachable
            
        Returns:
            Masked Q-values where invalid actions have -infinity
        """
        masked_q_values = q_values.clone()
        
        # Convert valid targets to action indices
        valid_actions = set()
        for row, col in valid_targets:
            action_idx = row * 35 + col  # GRID_COLS = 35
            valid_actions.add(action_idx)
        
        # Mask invalid actions with -infinity
        for action_idx in range(self.action_dim):
            if action_idx not in valid_actions:
                masked_q_values[0][action_idx] = -float('inf')
        
        return masked_q_values
    
    def select_action(self, state, valid_targets=None, training_mode=True):
        """
        Select a target to shoot at (0-699).
        valid_targets: list of valid (row, col) positions that can be reached
        training_mode: if False, disables all learning and exploration (pure inference)
        """
        if training_mode:
            # Only update training parameters when in training mode
            self.steps_done += 1
            self.epsilon = max(self.eps_end, self.eps_start - (self.eps_start - self.eps_end) * self.steps_done / self.eps_decay)
            self.last_epsilon = self.epsilon
            
            if random.random() < self.epsilon:
                # Random exploration: choose from valid targets only
                if valid_targets and len(valid_targets) > 0:
                    # Convert valid targets to action indices for random selection
                    valid_action_indices = []
                    for row, col in valid_targets:
                        action_idx = row * 35 + col  # GRID_COLS = 35
                        valid_action_indices.append(action_idx)
                    action = random.choice(valid_action_indices)
                    self.last_selection_random = True
                    self.last_selected_action = action
                    return action
                else:
                    # No valid targets - choose random action (will be masked)
                    action = random.randrange(self.action_dim)
                    self.last_selection_random = True
                    self.last_selected_action = action
                    return action
        else:
            # Set evaluation mode for inference
            self.set_evaluation_mode()
        
        # Always use exploitation (best action) when not in training mode
        # Exploitation: choose best valid target
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        
        # Apply action masking to avoid unreachable targets
        if valid_targets and len(valid_targets) > 0:
            q_values = self.apply_action_mask(q_values, valid_targets)
            # Select best action (invalid ones are masked with -infinity)
            action = q_values.argmax().item()
            self.last_selection_random = False
            self.last_selected_action = action
            return action
        else:
            # No valid targets - return random action (will be handled by environment)
            action = random.randrange(self.action_dim)
            self.last_selection_random = True
            self.last_selected_action = action
            return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Ensure network is in training mode for batch norm and dropout
        self.set_training_mode()
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # Current Q-values for the actions taken
        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # Regular DQN target: max_a' Q_target(next_state, a')
            next_q_values_all = self.target_net(next_state)
            next_q_values, _ = next_q_values_all.max(dim=1)
            target = reward + self.gamma * next_q_values * (1 - done)
        
        # Use MSE loss for value regression
        mse_loss = nn.functional.mse_loss(current_q_values, target)
        
        # Add L2 regularization to combat overfitting
        l2_lambda = 1e-4  # L2 regularization coefficient
        l2_reg = torch.tensor(0., device=self.device)
        for param in self.policy_net.parameters():
            l2_reg += torch.norm(param, 2)  # L2 norm of parameters
        
        # Total loss = MSE loss + L2 regularization
        total_loss = mse_loss + l2_lambda * l2_reg
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # NEW: Gradient clipping to prevent exploding gradients
        # This improves training stability, especially with deep networks
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        
        self.optimizer.step()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_learning_rate(self, episode):
        """
        Implement learning rate decay based on episode number.
        Reduces learning rate gradually to allow for more precise learning in later stages.
        """
        # Decay learning rate every 10,000 episodes
        decay_interval = 10000
        decay_factor = 0.7  # Reduce by 30% each time
        
        # Calculate how many decay steps we've taken
        decay_steps = episode // decay_interval
        
        # Apply decay
        new_lr = self.initial_lr * (decay_factor ** decay_steps)
        
        # Ensure learning rate doesn't go below a minimum threshold
        min_lr = 1e-6
        new_lr = max(new_lr, min_lr)
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return new_lr
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def decode_target_action(action_idx):
    """Convert action index to (row, col) target position (for backward compatibility)"""
    row = action_idx // GRID_COLS
    col = action_idx % GRID_COLS
    return row, col

def decode_color_action(action_idx):
    """Convert color action index to color (0-5)"""
    return action_idx % COLOR_ACTION_SIZE

def get_color_name(color_idx):
    """Get human-readable color name for debugging"""
    color_names = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange"]
    return color_names[color_idx] if 0 <= color_idx < len(color_names) else f"Color_{color_idx}"

def calculate_angle_to_target(shooter_x, shooter_y, target_x, target_y):
    """Calculate the angle needed to shoot from shooter to target"""
    dx = target_x - shooter_x
    dy = target_y - shooter_y
    angle = math.degrees(math.atan2(dy, dx))
    return angle 