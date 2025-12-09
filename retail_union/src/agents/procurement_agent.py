import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Any

class DQN(nn.Module):
    """
    Deep Q-Network for Value Function Approximation.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ProcurementAgent:
    """
    The 'Blue Collar' Agent using Deep Q-Learning to manage inventory.
    It learns to balance Cash, Space, and Demand by interacting with the environment.

    Attributes:
        state_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space.
        device (str): Computation device ('cpu' or 'cuda').
        policy_net (DQN): Active network for action selection.
        target_net (DQN): Stable network for target Q-value calculation.
    """
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.device: str = device
        
        self.policy_net: DQN = DQN(state_dim, action_dim).to(device)
        self.target_net: DQN = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer: optim.Optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory: deque = deque(maxlen=10000)
        
        self.batch_size: int = 64
        self.gamma: float = 0.99
        self.epsilon: float = 1.0
        self.epsilon_decay: float = 0.995
        self.epsilon_min: float = 0.01

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action using Epsilon-Greedy policy.

        Args:
            state (np.ndarray): Current state vector.

        Returns:
            int: Selected action index.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax().item())

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Stores a transition tuple in the Replay Buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> None:
        """
        Performs one step of Gradient Descent on the Policy Network using Experience Replay.
        
        Key RL Concepts:
        1. Experience Replay: Breaks correlation between consecutive samples, stabilizing training.
        2. Double DQN (Target Net): Uses a separate 'frozen' network to calculate target values,
           preventing the 'Moving Target' problem where updates spiral out of control.
        """
        if len(self.memory) < self.batch_size:
            return  # Wait until we have enough data to form a batch
        
        # Sample a random batch of transitions from the Replay Buffer
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        # Convert to Pytorch Tensors for GPU/CPU processing
        state_batch_t = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch_t = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch_t = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch_t = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch_t = torch.FloatTensor(done_batch).to(self.device)
        
        # 1. Compute predicted Q-values: Q(s, a) using Policy Net
        # "What did we think the value of this action was?"
        q_values = self.policy_net(state_batch_t).gather(1, action_batch_t)
        
        # 2. Compute target Q-values: r + gamma * max Q(s', a') using Target Net
        # "What was the *actual* observed value (reward + future promise)?"
        # The .detach() ensures we don't backpropagate into the Target Net
        next_q_values = self.target_net(next_state_batch_t).max(1)[0].detach()
        expected_q_values = reward_batch_t + (self.gamma * next_q_values * (1 - done_batch_t))
        
        # 3. Compute Loss (MSE): Diff between Prediction and Reality
        loss = nn.MSELoss()(q_values.squeeze(), expected_q_values)
        
        # 4. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping (Edge Case): Prevents exploding gradients if loss is huge
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Epsilon decay: Gradually reduce exploration (Action Randomness) 
        # as the agent gets smarter.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self) -> None:
        """Copies weights from Policy Net to Target Net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
