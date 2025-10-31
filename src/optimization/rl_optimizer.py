"""
Reinforcement Learning-based hyperparameter optimization for GANs.
Implements Rainbow DQN and PPO for discrete and continuous action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import deque, namedtuple
import logging
import random

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for Rainbow DQN.
    Samples important experiences more frequently.
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent
            beta: Importance sampling weight
            beta_increment: Beta increment per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, *args):
        """Add experience to buffer."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(*args))
        else:
            self.buffer[self.position] = Experience(*args)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch from buffer with prioritization.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (experiences, indices, importance weights)
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
    
    def __len__(self):
        return len(self.buffer)


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration in Rainbow DQN.
    Adds learnable noise to weights.
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Initialize noisy linear layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            std_init: Initial standard deviation
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Register noise
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Sample new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise using factorized Gaussian noise."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    """
    Rainbow DQN: Combines 6 DQN improvements:
    1. Double Q-learning
    2. Prioritized replay
    3. Dueling networks
    4. Multi-step learning
    5. Distributional RL (C51)
    6. Noisy networks
    
    Used for discrete hyperparameter optimization.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0
    ):
        """
        Initialize Rainbow DQN.
        
        Args:
            state_dim: State dimension
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
            num_atoms: Number of atoms for distributional RL
            v_min: Minimum value for distribution
            v_max: Maximum value for distribution
        """
        super(RainbowDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Support for distributional RL
        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, num_atoms)
        )
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Feature extraction
        self.feature_layer = nn.Sequential(
            NoisyLinear(state_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Dueling architecture
        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, num_atoms)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim * num_atoms)
        )
        
        logger.info(
            f"Created Rainbow DQN: state_dim={state_dim}, "
            f"action_dim={action_dim}, num_atoms={num_atoms}"
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute action value distribution.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            Action value distributions [batch_size, action_dim, num_atoms]
        """
        batch_size = state.size(0)
        
        # Extract features
        features = self.feature_layer(state)
        
        # Compute value and advantage
        value = self.value_stream(features)  # [batch_size, num_atoms]
        advantage = self.advantage_stream(features)  # [batch_size, action_dim * num_atoms]
        
        # Reshape advantage
        advantage = advantage.view(batch_size, self.action_dim, self.num_atoms)
        
        # Dueling: Q = V + (A - mean(A))
        value = value.view(batch_size, 1, self.num_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probability distribution
        q_dist = F.softmax(q_atoms, dim=-1)
        
        return q_dist
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values (expected values).
        
        Args:
            state: State tensor
        
        Returns:
            Q-values [batch_size, action_dim]
        """
        q_dist = self.forward(state)
        q_values = (q_dist * self.support).sum(dim=-1)
        return q_values
    
    def reset_noise(self):
        """Reset noise in all noisy layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class RainbowAgent:
    """
    Agent using Rainbow DQN for hyperparameter optimization.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        n_step: int = 3,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize Rainbow agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension
            learning_rate: Learning rate
            gamma: Discount factor
            n_step: Number of steps for multi-step learning
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            device: Device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Create networks
        self.online_net = RainbowDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = RainbowDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=learning_rate
        )
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
        
        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        self.update_count = 0
        
        logger.info(f"Created Rainbow Agent with {n_step}-step learning")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using online network.
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if training:
            self.online_net.train()
        else:
            self.online_net.eval()
        
        with torch.no_grad():
            q_values = self.online_net.get_q_values(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            # Compute n-step return
            state, action, _, _, _ = self.n_step_buffer[0]
            _, _, _, next_state, done = self.n_step_buffer[-1]
            
            n_step_reward = sum([
                self.gamma ** i * transition[2]
                for i, transition in enumerate(self.n_step_buffer)
            ])
            
            self.replay_buffer.push(state, action, n_step_reward, next_state, done)
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step.
        
        Returns:
            Training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample from replay buffer
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Compute current Q distribution
        q_dist = self.online_net(states)
        q_dist = q_dist[range(self.batch_size), actions]
        
        # Compute target Q distribution (Double DQN)
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.online_net.get_q_values(next_states).argmax(dim=1)
            
            # Evaluate using target network
            next_q_dist = self.target_net(next_states)
            next_q_dist = next_q_dist[range(self.batch_size), next_actions]
            
            # Project distribution
            target_support = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.gamma ** self.n_step) * self.online_net.support.unsqueeze(0)
            target_support = target_support.clamp(self.online_net.v_min, self.online_net.v_max)
            
            # Compute categorical projection
            b = (target_support - self.online_net.v_min) / self.online_net.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            target_q_dist = torch.zeros_like(next_q_dist)
            for i in range(self.batch_size):
                for j in range(self.online_net.num_atoms):
                    target_q_dist[i, l[i, j]] += next_q_dist[i, j] * (u[i, j] - b[i, j])
                    target_q_dist[i, u[i, j]] += next_q_dist[i, j] * (b[i, j] - l[i, j])
        
        # Compute loss (cross-entropy)
        loss = -(target_q_dist * q_dist.log()).sum(dim=1)
        loss = (loss * weights).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        td_errors = (target_q_dist - q_dist).abs().sum(dim=1).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Reset noise
        self.online_net.reset_noise()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        return {
            'loss': loss.item(),
            'q_value': q_dist.mean().item()
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    state_dim = 10
    action_dim = 5
    
    agent = RainbowAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Test action selection
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    print(f"Selected action: {action}")
    
    # Test training step
    for _ in range(100):
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = np.random.rand() < 0.1
        
        agent.store_transition(state, action, reward, next_state, done)
    
    metrics = agent.train_step()
    if metrics:
        print(f"Training metrics: {metrics}")
