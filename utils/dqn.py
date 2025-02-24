import numpy as np
import os
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from utils.fem import evaluate_composite
from tqdm import tqdm

from config import (MATRIX_SIZE, DESIRED_MODULUS, DESIRED_VOL_FRAC, MAX_STEPS,
                    LEARNING_RATE, BATCH_SIZE, GAMMA, EPSILON_DECAY, EPSILON_MIN,
                    FIRST_HIDDEN_NEURONS, SECOND_HIDDEN_NEURONS,
                    NUM_CYCLES, EPISODES_PER_CYCLE, OPT_STEPS_PER_CYCLE, E_STIFF, E_COMP)

# -------------------------------
# Environment Setup
# -------------------------------
class CompositeDesignEnv(gym.Env):
    """
    Composite Design Environment for Effective Young's Modulus.
    
    The design is a MATRIX_SIZE × MATRIX_SIZE binary grid.
    The state is a vector of length (MATRIX_SIZE² + 4), where:
      - First MATRIX_SIZE² entries: the design (0: stiff, 1: compliant).
      - The last 4 entries: current effective modulus (scaled), current volume fraction,
        desired modulus (scaled), and desired volume fraction.
    
    Actions (Discrete MATRIX_SIZE² + 1):
      - 0 to MATRIX_SIZE² - 1: flip the corresponding cell.
      - Last action: null action (do nothing).
    
    Reward: 0 if the effective modulus is within ±50 MPa of desired and volume fraction
            is within ±0.04 of desired; -1 otherwise.
    
    Each episode is limited to MAX_STEPS.
    """
    def __init__(self):
        super(CompositeDesignEnv, self).__init__()
        self.grid_size = (MATRIX_SIZE, MATRIX_SIZE)
        self.num_cells = MATRIX_SIZE * MATRIX_SIZE
        self.max_steps = MAX_STEPS
        self.max_modulus = max(E_STIFF, E_COMP)
        
        # Desired properties
        self.desired_modulus = DESIRED_MODULUS
        self.desired_vol_frac = DESIRED_VOL_FRAC
        
        # Define action space: one action per cell + a null action.
        self.action_space = spaces.Discrete(self.num_cells + 1)
        # Observation: flattened grid (num_cells) plus 4 additional metrics.
        self.observation_space = spaces.Box(low=0.0, high=1.0, 
                                            shape=(self.num_cells + 4,), dtype=np.float32)
        self.reset()
        self.saved_designs = []
        self.comp_des_num = 0

    def reset(self):
        # Initialize a random MATRIX_SIZE×MATRIX_SIZE design (0: stiff, 1: compliant)
        self.grid = np.random.randint(0, 2, size=self.grid_size)
        self.current_step = 0

        # Evaluate effective modulus using the FEM module
        self.current_modulus = evaluate_composite(self.grid)
        # Compute current volume fraction (fraction of compliant cells)
        self.current_vol_frac = np.sum(self.grid) / self.num_cells
        
        return self._get_state()
    
    def _get_state(self):
        grid_flat = self.grid.flatten().astype(np.float32)
        state = np.concatenate([
            grid_flat,
            [self.current_modulus / self.max_modulus, self.current_vol_frac,
             self.desired_modulus / self.max_modulus, self.desired_vol_frac]
        ])
        return state.astype(np.float32)
    
    def step(self, action):
        self.current_step += 1
        # Flip a cell if action is less than num_cells
        if action < self.num_cells:
            row = action // MATRIX_SIZE
            col = action % MATRIX_SIZE
            self.grid[row, col] = 1 - self.grid[row, col]
        # Else: null action (do nothing)
        
        # Re-evaluate composite properties
        self.current_modulus = evaluate_composite(self.grid)
        self.current_vol_frac = np.sum(self.grid) / self.num_cells
        # Compute reward
        if (abs(self.current_modulus - self.desired_modulus) <= 50) and \
           (abs(self.current_vol_frac - self.desired_vol_frac) <= 0.04):
            reward = 0.0
            self.render()  # Render only on satisfying design
        else:
            reward = -1.0
        done = self.current_step >= self.max_steps
        return self._get_state(), reward, done, {}
    
    def render(self, mode='human'):
        # Avoid duplicate plots for the same design
        for saved_design in self.saved_designs:
            if np.array_equal(saved_design, self.grid):
                print("repetative")
                return
        
        # Save unique design
        self.saved_designs.append(self.grid.copy())
        self.comp_des_num += 1
        plt.imshow(self.grid, cmap='gray', vmin=0, vmax=1)
        # Annotate the plot with effective modulus and volume fraction.
        title_str = (
            f"Composite Design (0: Stiff, 1: Compliant)\n"
            f"Effective Modulus: {self.current_modulus:.2f} MPa\n"
            f"Volume Fraction: {self.current_vol_frac:.2f}"
        )
        plt.title(title_str)
        plt.colorbar()
        file_path = os.path.join("composite designs", f"plot #{self.comp_des_num}.png")
        plt.savefig(file_path, dpi=300)
        plt.savefig("my_plot.png")
        plt.close()

# -------------------------------
# Q-Network
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim=None, hidden_dim1=FIRST_HIDDEN_NEURONS, 
                 hidden_dim2=SECOND_HIDDEN_NEURONS, output_dim=None):
        super(QNetwork, self).__init__()
        if input_dim is None:
            # Input dimension: flattened grid + 4 metrics
            input_dim = MATRIX_SIZE * MATRIX_SIZE + 4
        if output_dim is None:
            # Output dimension: one per cell + null action
            output_dim = MATRIX_SIZE * MATRIX_SIZE + 1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# -------------------------------
# Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# -------------------------------
# DQN Agent with Double DQN Update
# -------------------------------
class DQNAgent:
    def __init__(self, device, lr=LEARNING_RATE, gamma=GAMMA, batch_size=BATCH_SIZE):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.main_net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
        # Epsilon for epsilon-greedy exploration.
        self.epsilon = 1.0
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.main_net.out.out_features)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.main_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q = self.main_net(states).gather(1, actions)
        next_actions = self.main_net(next_states).argmax(dim=1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, next_actions)
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def soft_update_target(self, tau=0.05):
        for target_param, main_param in zip(self.target_net.parameters(), self.main_net.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * main_param.data)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# -------------------------------
# Training Loop
# -------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CompositeDesignEnv()
    agent = DQNAgent(device)
    
    episode_rewards = []
    
    for cycle in tqdm(range(NUM_CYCLES)):
        for ep in range(EPISODES_PER_CYCLE):
            state = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
            agent.decay_epsilon()
            episode_rewards.append(ep_reward)
            if (cycle * EPISODES_PER_CYCLE + ep + 1) % 2000 == 0:
                print(f"Episode {cycle * EPISODES_PER_CYCLE + ep + 1}")
                env.render()
        for _ in range(OPT_STEPS_PER_CYCLE):
            agent.update()
        agent.soft_update_target()
        print(f"Cycle {cycle+1}/{NUM_CYCLES} completed. Last episode reward: {ep_reward} Epsilon: {agent.epsilon:.3f}")
    
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.show()

if __name__ == "__main__":
    print(MAX_STEPS)
    train()
