# utils/trainer.py

import os
import psutil
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import time

from utils.dqn import CompositeDesignEnv, DQNAgent # Assumes FCN versions
from config import (
    EPISODES_PER_CYCLE, OPT_STEPS_PER_CYCLE, NUM_CYCLES # Defaults
)
# Assuming CompositeDesignEnv has _check_goal_met method
from utils.fem import voigt_model, reuss_model # Needed for HER

def get_memory_usage_str(process):
    """Returns a formatted string of memory usage."""
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return (
        f"Proc RSS: {mem_info.rss / (1024**3):.2f} GB | "
        f"Sys Mem Used: {vm.percent:.1f}% ({vm.available / (1024**3):.2f} GB Avail) | "
        f"Sys Swap Used: {swap.percent:.1f}%"
    )

class Trainer:
    """Handles the training loop, checkpointing, and logging for a DQNAgent."""
    def __init__(
        self,
        env: CompositeDesignEnv,
        agent: DQNAgent,
        device: torch.device,
        trial_number: int,
        *,
        # Loop parameters
        episodes_per_cycle: int = EPISODES_PER_CYCLE,
        opt_steps_per_cycle: int = OPT_STEPS_PER_CYCLE,
        num_cycles: int = NUM_CYCLES, # Total cycles target for this specific run
        # Checkpointing
        checkpoint_dir: str = "checkpoints",
        # Resumption state
        start_cycle: int = 0,
        total_episodes_offset: int = 0,
        total_steps_offset: int = 0,
        initial_rewards: list[float] | None = None,
        initial_success_flags: list[bool] | None = None, # <<< For resuming success tracking
        # Hyperparameters used for this run (for logging/checkpointing)
        hyperparams: dict | None = None,
    ):
        self.env = env
        self.agent = agent
        self.device = device
        self.trial_number = trial_number
        self.episodes_per_cycle = episodes_per_cycle
        self.opt_steps_per_cycle = opt_steps_per_cycle
        self.num_cycles = num_cycles # Target cycles for this instance

        # Checkpointing setup
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_trial_{self.trial_number}.pth")

        # State for resuming training progress
        self.current_cycle = start_cycle
        self.total_episodes_run = total_episodes_offset # Episodes completed *before* this trainer instance started
        self.total_steps_run = total_steps_offset # Steps completed *before* this trainer instance started

        # Initialize reward and success tracking lists
        self.episode_rewards = initial_rewards if initial_rewards is not None else []
        # NOTE: Storing all flags can make checkpoints very large for long runs.
        # Consider alternatives (like only storing recent N flags or just counts) if size becomes an issue.
        self.episode_success_flags = initial_success_flags if initial_success_flags is not None else []
        # Ensure consistency if resuming
        if len(self.episode_rewards) != len(self.episode_success_flags):
            print(f"Warning: Resumed rewards ({len(self.episode_rewards)}) and success flags ({len(self.episode_success_flags)}) lengths differ. Resetting success flags.")
            self.episode_success_flags = [False] * len(self.episode_rewards) # Or handle more gracefully
        # Recalculate total successful count from flags if resuming
        self.total_successful_episodes = sum(self.episode_success_flags)


        self.hyperparams_for_this_run = hyperparams if hyperparams is not None else {}

        # Performance tracking
        self.cycle_times = []
        self.process = psutil.Process(os.getpid())
        # Attributes to store final calculated rates
        self.final_avg_score = -float('inf')
        self.final_window_success_rate = 0.0


    def _save_checkpoint(self):
        """Saves the current state of the trainer and agent."""
        agent_state_for_checkpoint = self.agent.get_state()

        trainer_checkpoint_state = {
            'trial_number': self.trial_number,
            'current_cycle': self.current_cycle,
            'total_episodes_run': self.total_episodes_run,
            'total_steps_run': self.total_steps_run,
            'episode_rewards': self.episode_rewards,
            'episode_success_flags': self.episode_success_flags, # <<< Save success flags list
            # total_successful_episodes can be recalculated from the list on load
            'cycle_times': self.cycle_times,
            'hyperparams_of_this_run': self.hyperparams_for_this_run,
            'trainer_loop_config': {
                'episodes_per_cycle': self.episodes_per_cycle,
                'opt_steps_per_cycle': self.opt_steps_per_cycle,
                'num_cycles_target': self.num_cycles,
            }
        }
        checkpoint_content = {
            'trainer_checkpoint_state': trainer_checkpoint_state,
            'agent_checkpoint_state': agent_state_for_checkpoint,
        }
        # Atomic Save
        temp_path = self.checkpoint_path + ".tmp"
        try:
            torch.save(checkpoint_content, temp_path)
            os.replace(temp_path, self.checkpoint_path)
        except Exception as e:
            print(f"ERROR saving checkpoint for trial {self.trial_number}: {e}")
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except OSError: pass

    @staticmethod
    def load_checkpoint_data(checkpoint_path: str):
        """Loads checkpoint data. Returns dict or None."""
        if not os.path.isfile(checkpoint_path): return None # Check if it's a file
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Basic validation
            if not all(k in checkpoint for k in ['trainer_checkpoint_state', 'agent_checkpoint_state']): return None
            if 'current_cycle' not in checkpoint['trainer_checkpoint_state']: return None
            if 'main_net_state_dict' not in checkpoint['agent_checkpoint_state']: return None
            return checkpoint
        except Exception as e:
            print(f"ERROR loading checkpoint {checkpoint_path}: {e}. Corrupted?")
            # Consider deleting corrupted checkpoint here?
            # try: os.remove(checkpoint_path) except OSError: pass
            return None

    def apply_hindsight(self, episode_transitions):
        """Applies Hindsight Experience Replay (HER) strategy: 'final'."""
        # --- This function remains the same as the previous version ---
        # It modifies transitions based on the final achieved state and pushes
        # them to the replay buffer. It uses env.compute_reward and
        # env._check_goal_met for the relabeled transitions.
        if not episode_transitions: return
        final_transition = episode_transitions[-1]
        final_next_state_flat = final_transition[3]
        final_grid_flat = final_next_state_flat[:self.agent.flat_grid_dim_for_buffer]
        final_grid_hw = final_grid_flat.reshape(self.agent.H, self.agent.W)
        achieved_goal_vf = (self.agent.H * self.agent.W - np.sum(final_grid_hw)) / (self.agent.H * self.agent.W)
        achieved_goal_mod_abs = (voigt_model(achieved_goal_vf) + reuss_model(achieved_goal_vf)) / 2
        for i, (s_flat, a, r_orig, s_prime_flat, done_orig) in enumerate(episode_transitions):
            s_her_flat = s_flat.copy(); s_prime_her_flat = s_prime_flat.copy()
            # Re-calc state metrics based on HER goal
            s_her_grid_flat = s_her_flat[:self.agent.flat_grid_dim_for_buffer]
            s_her_vf = (self.agent.H*self.agent.W - np.sum(s_her_grid_flat.reshape(self.agent.H,self.agent.W)))/(self.agent.H*self.agent.W)
            s_her_mod_abs = (voigt_model(s_her_vf) + reuss_model(s_her_vf)) / 2
            s_her_dist_e_scaled = np.clip((s_her_mod_abs - achieved_goal_mod_abs) / self.env.max_modulus if self.env.max_modulus > 0 else 0.0, -1.0, 1.0)
            s_her_flat[self.agent.flat_grid_dim_for_buffer + 2] = s_her_dist_e_scaled
            s_her_flat[self.agent.flat_grid_dim_for_buffer + 3] = achieved_goal_vf
            s_prime_her_grid_flat = s_prime_her_flat[:self.agent.flat_grid_dim_for_buffer]
            s_prime_her_vf = (self.agent.H*self.agent.W - np.sum(s_prime_her_grid_flat.reshape(self.agent.H,self.agent.W)))/(self.agent.H*self.agent.W)
            s_prime_her_mod_abs = (voigt_model(s_prime_her_vf) + reuss_model(s_prime_her_vf)) / 2
            s_prime_her_dist_e_scaled = np.clip((s_prime_her_mod_abs - achieved_goal_mod_abs) / self.env.max_modulus if self.env.max_modulus > 0 else 0.0, -1.0, 1.0)
            s_prime_her_flat[self.agent.flat_grid_dim_for_buffer + 2] = s_prime_her_dist_e_scaled
            s_prime_her_flat[self.agent.flat_grid_dim_for_buffer + 3] = achieved_goal_vf
            # Recompute reward and done status for HER transition
            new_reward = self.env.compute_reward(s_prime_her_mod_abs, s_prime_her_vf, achieved_goal_mod_abs, achieved_goal_vf)
            her_done = self.env._check_goal_met(s_prime_her_mod_abs, s_prime_her_vf, achieved_goal_mod_abs, achieved_goal_vf)
            self.agent.replay_buffer.push(s_her_flat, a, new_reward, s_prime_her_flat, her_done)


    def train(self) -> float:
        """Runs the main training loop."""
        print(f"--- Trial {self.trial_number}: Starting Training ---")
        print(f" Target Cycles: {self.num_cycles}, Start Cycle: {self.current_cycle + 1}")
        print(f" Episodes/Cycle: {self.episodes_per_cycle}, Opt Steps/Cycle: {self.opt_steps_per_cycle}")
        print(f" Resumed State: Start Eps={self.total_episodes_run}, Start Successes={self.total_successful_episodes}")
        print(f" Device: {self.device}, Buffer: '{self.agent.replay_buffer.directory}', Size: {len(self.agent.replay_buffer)}")
        print(f"Initial Memory: {get_memory_usage_str(self.process)}")
        print("-" * 40)


        LOSS_THRESHOLD = 1e3 # Example: Stop if loss exceeds 1 million
        AWFUL_SCORE_ON_DIVERGENCE = -1e9 # Value to return if training diverges


        start_time_train = time.time()
        diverged = False
        cycle_pbar = tqdm(range(self.current_cycle, self.num_cycles),
                          total=self.num_cycles, initial=self.current_cycle,
                          desc=f"T{self.trial_number} Cyc", unit="cyc") # Shorter desc

        for cycle_idx in cycle_pbar:
            if diverged:
                print(f"\n[Trainer] Divergence detected in previous cycle. Stopping training early for Trial {self.trial_number}.")
                break

            self.current_cycle = cycle_idx
            start_time_cycle = time.time()
            cycle_episode_rewards = []
            cycle_losses = []
            cycle_success_flags_in_cycle = []

            # --- Episode Loop ---
            ep_pbar = tqdm(range(self.episodes_per_cycle), desc=f" C{cycle_idx+1} Eps", unit="ep", leave=False)
            for _ in ep_pbar:
                episode_transitions = []
                state_flat = self.env.reset()
                done = False; ep_reward = 0.0; ep_steps = 0
                last_info = {}

                while not done:
                    action = self.agent.select_action(state_flat)
                    next_state_flat, reward, done, info = self.env.step(action)
                    last_info = info # Store info from the last step
                    # Store original flat transitions for buffer and HER
                    self.agent.replay_buffer.push(state_flat, action, reward, next_state_flat, done)
                    episode_transitions.append((state_flat, action, reward, next_state_flat, done))
                    state_flat = next_state_flat
                    ep_reward += reward; ep_steps += 1
                    self.total_steps_run += 1

                # --- Post-Episode Processing ---
                self.total_episodes_run += 1
                self.episode_rewards.append(ep_reward)
                cycle_episode_rewards.append(ep_reward)

                # Check goal met based on the *last* state of the episode
                episode_goal_met = last_info.get('goal_met', False)
                self.episode_success_flags.append(episode_goal_met) # <<< Store flag
                cycle_success_flags_in_cycle.append(episode_goal_met) # Track for cycle rate
                if episode_goal_met:
                    self.total_successful_episodes += 1 # Update overall count

                self.apply_hindsight(episode_transitions)
                self.agent.decay_epsilon()
            # --- End Post-Episode ---


            # --- Optimization Loop ---
            opt_pbar = tqdm(range(self.opt_steps_per_cycle), desc=f" C{cycle_idx+1} Opt", unit="upd", leave=False)
            cycle_diverged_this_cycle = False
            for opt_step_count in opt_pbar:
                loss = self.agent.update()
                if loss is not None:
                    cycle_losses.append(loss)
                    if not np.isfinite(loss) or loss > LOSS_THRESHOLD:
                        print(f"\n[Trainer] Trial {self.trial_number}: Loss exploded/NaN/Inf ({loss:.4e}) at Opt Step {opt_step_count+1} in Cycle {cycle_idx+1}. Stopping early.")
                        diverged = True
                        cycle_diverged_this_cycle = True
                        break # Exit the optimization step loop
                # Soft update target network periodically
                if (opt_step_count + 1) % 50 == 0:
                    self.agent.soft_update_target()
            if cycle_diverged_this_cycle:
                break

            # --- Post-Cycle Reporting & Checkpoint ---
            cycle_duration = time.time() - start_time_cycle
            self.cycle_times.append(cycle_duration)
            if self.agent.lr_scheduler:
                self.agent.lr_scheduler.step()

            # Calculate cycle statistics
            avg_rew = np.mean(cycle_episode_rewards) if cycle_episode_rewards else 0.0
            avg_loss = np.mean(cycle_losses) if cycle_losses else 0.0
            # Calculate success rate for *this cycle*
            cycle_success_rate = (sum(cycle_success_flags_in_cycle) / len(cycle_success_flags_in_cycle) * 100) \
                                 if cycle_success_flags_in_cycle else 0.0

            # Update overall progress bar description
            cycle_pbar.set_postfix({
                "AvgRew": f"{avg_rew:.2f}",
                "Succ(%)": f"{cycle_success_rate:.1f}", # <<< Cycle success rate
                "AvgLoss": f"{avg_loss:.4f}",
                "Eps": f"{self.agent.epsilon:.3f}",
                "LR": f"{self.agent.get_current_lr():.1e}",
                "Buf": f"{len(self.agent.replay_buffer)/self.agent.replay_buffer.capacity:.1%}",
                "Time": f"{cycle_duration:.1f}s"
            }, refresh=True) # Refresh needed to show updates

            # Log memory periodically
            if (cycle_idx + 1) % (max(1, self.num_cycles // 10)) == 0: # ~10 times per run
                print(f"\n End Cyc {cycle_idx+1}. Mem: {get_memory_usage_str(self.process)}")

            self._save_checkpoint() # Save after every cycle

        # --- Training Finished ---
        cycle_pbar.close()
        total_training_time = time.time() - start_time_train
        print("-" * 40)
        print(f"Trial {self.trial_number}: Training finished. Total time: {total_training_time:.2f}s")
        print(f" Total episodes run (incl. resumed): {self.total_episodes_run}")
        print(f" Total steps run (incl. resumed): {self.total_steps_run}")
        print(f" Total goal achieved episodes: {self.total_successful_episodes}")

        if diverged:
            print(" Training stopped early due to divergence.")
            self.final_avg_score = AWFUL_SCORE_ON_DIVERGENCE # Assign awful score
            self.final_window_success_rate = 0.0 # Success rate is likely meaningless
        else:
            # Calculate final scores based on window (same as before)
            score_window_size = min(1000, self.total_episodes_run)
            if score_window_size > 0:
                self.final_avg_score = float(np.mean(self.episode_rewards[-score_window_size:]))
                final_window_flags = self.episode_success_flags[-score_window_size:]
                self.final_window_success_rate = (sum(final_window_flags) / len(final_window_flags) * 100)
            else:
                self.final_avg_score = -float('inf')
                self.final_window_success_rate = 0.0
            print(f" Avg Reward (last {score_window_size} eps): {self.final_avg_score:.4f}")
            print(f" Success Rate (last {score_window_size} eps): {self.final_window_success_rate:.2f}%")

        print(f"Final Memory: {get_memory_usage_str(self.process)}")
        print("-" * 40)

        self.agent.close_buffer()
        return self.final_avg_score


    def plot_rewards(self, window=100, save_path=None):
        """Plots rolling mean reward."""
        # --- This method remains unchanged ---
        if not self.episode_rewards: print(f"T{self.trial_number}: No rewards for plot."); return
        rewards_arr = np.array(self.episode_rewards); eff_win = min(window, len(rewards_arr))
        if eff_win <= 0: return
        try: import pandas as pd
        except ImportError: pd=None
        if pd: roll_mean = pd.Series(rewards_arr).rolling(window=eff_win, min_periods=1).mean(); idx = np.arange(len(rewards_arr))
        else: roll_mean = np.convolve(rewards_arr, np.ones(eff_win)/eff_win, mode='valid'); idx = np.arange(eff_win - 1, len(rewards_arr))
        plt.figure(figsize=(10, 5)); plt.plot(idx, roll_mean, label=f'RollMean(w={eff_win})', c='b')
        plt.plot(np.arange(len(rewards_arr)), rewards_arr, alpha=0.2, label='Raw Reward', c='lightblue')
        plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title(f"T{self.trial_number} Train Rewards")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        if save_path:
            try: plt.savefig(save_path, dpi=120)
            except Exception as e: print(f"Err saving plot {save_path}: {e}")
        else: plt.show()
        plt.close()