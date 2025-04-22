# utils/trainer.py
import os
import torch
import torch.serialization
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from utils.dqn import CompositeDesignEnv, DQNAgent, ReplayBuffer
from config import NUM_CYCLES, EPISODES_PER_CYCLE, OPT_STEPS_PER_CYCLE, E_STIFF, E_COMP

CHECKPOINT_PATH = "checkpoint_1000.pth"

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
        self.checkpoint_path = f"checkpoint_{cycle}.pth"
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at cycle {cycle}")

    def load_checkpoint(self):
        # Allow ReplayBuffer as a safe global for unpickling.
        torch.serialization.add_safe_globals([ReplayBuffer])
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        self.agent.main_net.load_state_dict(checkpoint['agent_main_net_state_dict'])
        self.agent.target_net.load_state_dict(checkpoint['agent_target_net_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.agent.replay_buffer = checkpoint['replay_buffer']
        self.agent.epsilon = checkpoint['epsilon']
        print(f"Checkpoint loaded from cycle {checkpoint['cycle']}")
        return checkpoint['cycle'], checkpoint['episode_rewards']
    

    def apply_hindsight(self, episode_transitions):
        """
        Perform 4 passes of HER:
          1. Full episode with the true final state.
          2. Episode truncated by 1 step (and use second-to-last state as final).
          3. Truncated by 2 steps.
          4. Truncated by 3 steps.
        """
        if not episode_transitions:
            return

        # Pass 1: standard final
        # final_state is from the last transition's next_state
        final_state = episode_transitions[-1][3]
        self.apply_her_to_sequence(episode_transitions, final_state)

        # For the second, third, fourth: we ensure we have enough transitions
        # and then slice off the last N transitions, mark the new last step done=1,
        # and use that as the final design.

        # Pass 2: remove the last step, use transitions[:-1]
        if len(episode_transitions) >= 2:
            truncated2 = episode_transitions[:-1]  # remove last
            # Mark the new last step's 'done' as True
            truncated2 = list(truncated2)  # ensure it's mutable
            last_step = list(truncated2[-1])  # (s, a, r, ns, done)
            last_step[4] = True  # set done=1
            truncated2[-1] = tuple(last_step)

            final_state_2 = truncated2[-1][3]  # next_state of the new last step
            self.apply_her_to_sequence(truncated2, final_state_2)
        """
        # Pass 3: remove the last 2 steps, use transitions[:-2]
        if len(episode_transitions) >= 3:
            truncated3 = episode_transitions[:-2]
            truncated3 = list(truncated3)
            last_step = list(truncated3[-1])
            last_step[4] = True
            truncated3[-1] = tuple(last_step)

            final_state_3 = truncated3[-1][3]
            self.apply_her_to_sequence(truncated3, final_state_3)

        # Pass 4: remove the last 3 steps, use transitions[:-3]
        if len(episode_transitions) >= 4:
            truncated4 = episode_transitions[:-3]
            truncated4 = list(truncated4)
            last_step = list(truncated4[-1])
            last_step[4] = True
            truncated4[-1] = tuple(last_step)

            final_state_4 = truncated4[-1][3]
            self.apply_her_to_sequence(truncated4, final_state_4)
        """

    def apply_her_to_sequence(self, episode_transitions, final_state):
        """
        Helper: apply standard HER relabeling to a sequence of transitions,
        using `final_state` as the 'goal' state, and push them into replay buffer.
        """
        # final_state is assumed to have the "actual" final modulus/vol_frac
        # in the first 2 of its last 4 fields: final_state[-4], final_state[-3]
        max_mod = max(E_STIFF, E_COMP)
        final_modulus_scaled = final_state[-4]
        final_vol_frac = final_state[-3]
        final_modulus = final_modulus_scaled * max_mod

        for (state, action, reward, next_state, done) in episode_transitions:
            # Copy states so we can overwrite their "desired" portion
            state_her = state.copy()
            next_state_her = next_state.copy()

            # Overwrite the last two "goal" fields in state & next_state
            state_her[-2] = final_modulus_scaled
            state_her[-1] = final_vol_frac
            next_state_her[-2] = final_modulus_scaled
            next_state_her[-1] = final_vol_frac

            # Recompute the reward with the new "goal"
            current_modulus_scaled = state_her[-4]  # current E / max_mod
            current_vol_frac = state_her[-3]
            current_modulus = current_modulus_scaled * max_mod

            new_reward = self.env.compute_reward(
                current_modulus,
                current_vol_frac,
                final_modulus,
                final_vol_frac
            )

            # Push relabeled transition
            self.agent.replay_buffer.push(state_her, action, new_reward, next_state_her, done)

    def train(self):
        for cycle in tqdm(range(self.start_cycle, NUM_CYCLES)):
            for ep in range(EPISODES_PER_CYCLE):
                episode_transitions = []
                state = self.env.reset()
                done = False
                ep_reward = 0
                while not done:
                    action = self.agent.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    episode_transitions.append((state, action, reward, next_state, done))
                    self.agent.replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state
                    ep_reward += reward
                
                self.apply_hindsight(episode_transitions)
                self.agent.decay_epsilon()
                self.episode_rewards.append(ep_reward)
            for _ in range(OPT_STEPS_PER_CYCLE):
                self.agent.update()
            self.agent.soft_update_target(tau=0.07122440382899094)
            print(f"Cycle {cycle+1}/{NUM_CYCLES} completed. Last episode reward: {ep_reward} Epsilon: {self.agent.epsilon:.3f}")
            # Save checkpoint at the end of each cycle.
            if (cycle+1) % 200 == 0:
                self.save_checkpoint(cycle+1)
        self.plot_rewards()
        
    def plot_rewards(self, window=1000):
        """
        Plot the rolling (moving) average of episode rewards over the specified window size.
        """
        if len(self.episode_rewards) == 0:
            print("No rewards to plot.")
            return

        rolling_means = []
        for i in range(len(self.episode_rewards)):
            # The start index is chosen so that you only look at the last 'window' episodes.
            start_idx = max(0, i - window + 1)
            windowed_rewards = self.episode_rewards[start_idx:i+1]
            rolling_means.append(np.mean(windowed_rewards))

        plt.plot(rolling_means)
        plt.xlabel("Episode")
        plt.ylabel(f"Rolling Mean Reward (window={window})")
        plt.title("Training Rewards (Smoothed)")
        plt.show()
