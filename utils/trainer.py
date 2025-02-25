# utils/trainer.py
import os
import torch
import torch.serialization
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.dqn import CompositeDesignEnv, DQNAgent, ReplayBuffer
from config import NUM_CYCLES, EPISODES_PER_CYCLE, OPT_STEPS_PER_CYCLE

CHECKPOINT_PATH = "checkpoint.pth"

class Trainer:
    def __init__(self, device=None, checkpoint_path=CHECKPOINT_PATH):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.env = CompositeDesignEnv()
        self.agent = DQNAgent(self.device)
        self.episode_rewards = []
        self.start_cycle = 0

        # If checkpoint exists, load it.
        if os.path.exists(self.checkpoint_path):
            self.start_cycle, self.episode_rewards = self.load_checkpoint()

    def save_checkpoint(self, cycle):
        checkpoint = {
            'cycle': cycle,
            'episode_rewards': self.episode_rewards,
            'agent_main_net_state_dict': self.agent.main_net.state_dict(),
            'agent_target_net_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            # Save replay_buffer if you wish to resume its state.
            'replay_buffer': self.agent.replay_buffer,
            'epsilon': self.agent.epsilon,
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at cycle {cycle}")

    def load_checkpoint(self):
        # Allow ReplayBuffer as a safe global for unpickling.
        torch.serialization.add_safe_globals([ReplayBuffer])
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        self.agent.main_net.load_state_dict(checkpoint['agent_main_net_state_dict'])
        self.agent.target_net.load_state_dict(checkpoint['agent_target_net_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.agent.replay_buffer = checkpoint['replay_buffer']
        self.agent.epsilon = checkpoint['epsilon']
        print(f"Checkpoint loaded from cycle {checkpoint['cycle']}")
        return checkpoint['cycle'], checkpoint['episode_rewards']

    def train(self):
        for cycle in tqdm(range(self.start_cycle, NUM_CYCLES)):
            for ep in range(EPISODES_PER_CYCLE):
                state = self.env.reset()
                done = False
                ep_reward = 0
                while not done:
                    action = self.agent.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state
                    ep_reward += reward
                self.agent.decay_epsilon()
                self.episode_rewards.append(ep_reward)
            for _ in range(OPT_STEPS_PER_CYCLE):
                self.agent.update()
            self.agent.soft_update_target()
            print(f"Cycle {cycle+1}/{NUM_CYCLES} completed. Last episode reward: {ep_reward} Epsilon: {self.agent.epsilon:.3f}")
            # Save checkpoint at the end of each cycle.
            self.save_checkpoint(cycle+1)
        
        # Plot training rewards after training completes.
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Rewards")
        plt.show()
