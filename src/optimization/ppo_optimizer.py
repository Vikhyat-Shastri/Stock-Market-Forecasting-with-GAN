"""
Proximal Policy Optimization (PPO) for continuous hyperparameter optimization.
More sample-efficient than DQN for continuous action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Actor outputs policy (mean and std for continuous actions).
    Critic outputs value function.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        activation: str = 'tanh'
    ):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension (continuous)
            hidden_dim: Hidden layer dimension
            activation: Activation function
        """
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh() if activation == 'tanh' else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh() if activation == 'tanh' else nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh() if activation == 'tanh' else nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Log standard deviation (learned parameter)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh() if activation == 'tanh' else nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
        
        logger.info(
            f"Created ActorCritic: state_dim={state_dim}, "
            f"action_dim={action_dim}, hidden_dim={hidden_dim}"
        )
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            Tuple of (action mean, value)
        """
        features = self.shared_layers(state)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor [batch_size, state_dim]
            deterministic: Whether to sample deterministically
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_mean, value = self.forward(state)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            # Sample from Gaussian distribution
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for training).
        
        Args:
            state: State tensor
            action: Action tensor
        
        Returns:
            Tuple of (log_prob, value, entropy)
        """
        action_mean, value = self.forward(state)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy


class PPORolloutBuffer:
    """
    Rollout buffer for PPO algorithm.
    Stores trajectory data for on-policy training.
    """
    
    def __init__(self):
        """Initialize buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ):
        """Add transition to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self) -> Tuple[np.ndarray, ...]:
        """
        Get all data from buffer.
        
        Returns:
            Tuple of numpy arrays
        """
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.dones)
        )
    
    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO agent for continuous hyperparameter optimization.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm
            n_epochs: Number of epochs per update
            batch_size: Mini-batch size
            device: Device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Create network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
            eps=1e-5
        )
        
        # Rollout buffer
        self.buffer = PPORolloutBuffer()
        
        logger.info(
            f"Created PPO Agent: clip_epsilon={clip_epsilon}, "
            f"n_epochs={n_epochs}, batch_size={batch_size}"
        )
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action from policy.
        
        Args:
            state: Current state
            deterministic: Whether to act deterministically
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)
        
        action = action.cpu().numpy()[0]
        log_prob = log_prob.item() if log_prob is not None else 0.0
        value = value.item()
        
        return action, log_prob, value
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ):
        """Store transition in buffer."""
        self.buffer.add(state, action, log_prob, reward, value, done)
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward array
            values: Value array
            dones: Done flags array
            last_value: Value of last state
        
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        
        # Append last value
        values_with_last = np.append(values, last_value)
        
        # Compute advantages backward
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
                next_gae = 0.0
            else:
                next_value = values_with_last[t + 1]
                next_gae = last_gae
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_gae
        
        # Returns = advantages + values
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Update policy using collected trajectories.
        
        Args:
            last_value: Value of last state
        
        Returns:
            Training metrics
        """
        if len(self.buffer) == 0:
            return {}
        
        # Get data from buffer
        states, actions, old_log_probs, rewards, values, dones = self.buffer.get()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0
        
        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            # Mini-batch updates
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs.unsqueeze(-1))
                policy_loss_1 = -batch_advantages.unsqueeze(-1) * ratio
                policy_loss_2 = -batch_advantages.unsqueeze(-1) * torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                
                # Value loss (MSE)
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                
                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    total_approx_kl += approx_kl.item()
                
                n_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'approx_kl': total_approx_kl / n_updates
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    state_dim = 10
    action_dim = 3
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Test action selection
    state = np.random.randn(state_dim)
    action, log_prob, value = agent.select_action(state)
    print(f"Action: {action}")
    print(f"Log prob: {log_prob:.4f}")
    print(f"Value: {value:.4f}")
    
    # Simulate trajectory collection
    for _ in range(100):
        state = np.random.randn(state_dim)
        action, log_prob, value = agent.select_action(state)
        reward = np.random.randn()
        done = np.random.rand() < 0.1
        
        agent.store_transition(state, action, log_prob, reward, value, done)
    
    # Update policy
    metrics = agent.update()
    print(f"Training metrics: {metrics}")
