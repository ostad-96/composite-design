# project_root/utils/trainer.py

import os
import psutil
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import time

from utils.dqn import CompositeDesignEnv, DQNAgent
from utils.replay_buffer import ReplayBuffer # <<< ADD THIS IMPORT
from config import (
    OPTUNA_DEFAULT_EPISODES_PER_CYCLE, OPTUNA_DEFAULT_OPT_STEPS_PER_CYCLE
)
from utils.fem import voigt_model, reuss_model

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
        episodes_per_cycle: int = None,
        opt_steps_per_cycle: int = None,
        num_cycles: int = None, # Total cycles target for this specific run
        # Checkpointing
        checkpoint_dir: str = "checkpoints",
        # Resumption state
        start_cycle: int = 0,
        total_episodes_offset: int = 0,
        total_steps_offset: int = 0,
        initial_rewards: list[float] | None = None,
        initial_success_flags: list[bool] | None = None,
        # Hyperparameters used for this run (for logging/checkpointing)
        hyperparams: dict | None = None,
    ):
        self.env = env
        self.agent = agent
        self.device = device
        self.trial_number = trial_number
        self.episodes_per_cycle = episodes_per_cycle if episodes_per_cycle is not None else OPTUNA_DEFAULT_EPISODES_PER_CYCLE
        self.opt_steps_per_cycle = opt_steps_per_cycle if opt_steps_per_cycle is not None else OPTUNA_DEFAULT_OPT_STEPS_PER_CYCLE
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
        self.episode_success_flags = initial_success_flags if initial_success_flags is not None else []
        if len(self.episode_rewards) != len(self.episode_success_flags):
            print(f"Warning: Resumed rewards ({len(self.episode_rewards)}) and success flags ({len(self.episode_success_flags)}) lengths differ. Resetting success flags.")
            self.episode_success_flags = [False] * len(self.episode_rewards)
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
            'episode_success_flags': self.episode_success_flags,
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
            return None

    def apply_hindsight(self, episode_transitions):
        """Applies Hindsight Experience Replay (HER) strategy: 'final'."""
        if not episode_transitions: return

        final_transition_info = episode_transitions[-1] # (s_flat, a, r_orig, s_prime_flat, done_orig)
        final_achieved_state_flat = final_transition_info[3] # s_prime_flat from the last step

        # Extract all scalar metrics from the final achieved state
        final_achieved_scalar_metrics = final_achieved_state_flat[self.agent.flat_grid_dim_for_buffer:]
        
        # Based on CompositeDesignEnv._get_flat_state() order:
        # 0: current_VF of achieved state (this is our HER goal VF)
        # 1: target_VF of achieved state (this was the original target_VF, not used for HER goal directly)
        # 2: current_E_scaled of achieved state (this is our HER goal E_scaled)
        # 3: target_E_scaled of achieved state (original target_E_scaled)
        # 4: scaled_step of achieved state
        
        achieved_goal_vf = final_achieved_scalar_metrics[0] 
        achieved_goal_mod_abs = final_achieved_scalar_metrics[2] * self.env.max_modulus if self.env.max_modulus > 0 else 0.0
        her_target_e_scaled = final_achieved_scalar_metrics[2] # This is current_E_scaled of achieved state, becomes target_E_scaled for HER

        for i, (s_flat, a, r_orig, s_prime_flat, done_orig) in enumerate(episode_transitions):
            s_her_flat = s_flat.copy()
            s_prime_her_flat = s_prime_flat.copy()

            # Modify scalar metrics in s_her_flat for HER
            # Indices for scalar_metrics part:
            # 0: current_VF (unchanged from original s_flat)
            # 1: target_VF (now HER goal: achieved_goal_vf)
            # 2: current_E_scaled (unchanged from original s_flat)
            # 3: target_E_scaled (now HER goal: her_target_e_scaled)
            # 4: scaled_step (unchanged from original s_flat)
            s_her_flat[self.agent.flat_grid_dim_for_buffer + 1] = achieved_goal_vf
            s_her_flat[self.agent.flat_grid_dim_for_buffer + 3] = her_target_e_scaled
            
            # Modify scalar metrics in s_prime_her_flat for HER
            s_prime_her_scalar_metrics_orig = s_prime_her_flat[self.agent.flat_grid_dim_for_buffer:]
            s_prime_her_mod_abs_value = s_prime_her_scalar_metrics_orig[2] * self.env.max_modulus if self.env.max_modulus > 0 else 0.0 # current E of s_prime
            s_prime_her_vf_value = s_prime_her_scalar_metrics_orig[0] # current VF of s_prime

            s_prime_her_flat[self.agent.flat_grid_dim_for_buffer + 1] = achieved_goal_vf
            s_prime_her_flat[self.agent.flat_grid_dim_for_buffer + 3] = her_target_e_scaled
            
            # Recompute reward and done status for HER transition
            # Reward based on how well s_prime_her (with its original current_E/VF) meets the HER goal
            new_reward = self.env.compute_reward(s_prime_her_mod_abs_value, s_prime_her_vf_value,
                                                 achieved_goal_mod_abs, achieved_goal_vf)
            her_done = self.env._check_goal_met(s_prime_her_mod_abs_value, s_prime_her_vf_value,
                                               achieved_goal_mod_abs, achieved_goal_vf)
            
            self.agent.replay_buffer.push(s_her_flat, a, new_reward, s_prime_her_flat, her_done)


    def train(self) -> float:
        """Runs the main training loop."""
        print(f"--- Trial {self.trial_number}: Starting Training ---")
        print(f" Target Cycles: {self.num_cycles}, Start Cycle: {self.current_cycle + 1}")
        print(f" Episodes/Cycle: {self.episodes_per_cycle}, Opt Steps/Cycle: {self.opt_steps_per_cycle}")
        print(f" Resumed State: Start Eps={self.total_episodes_run}, Start Successes={self.total_successful_episodes}")
        print(f" Device: {self.device}, Buffer: '{self.agent.replay_buffer.directory}', Size: {len(self.agent.replay_buffer)}")
        print(f" Agent is configured for {self.agent.fcn_input_C} FCN input channels.")
        print(f" Env provides {self.env.num_scalar_metrics} scalar metrics: {self.env.metric_names}")
        # This line caused the error, now fixed by importing ReplayBuffer
        print(f" Replay buffer uses indices {ReplayBuffer.METRIC_INDICES_FOR_FCN_CHANNELS} from env metrics for FCN.")
        print(f"Initial Memory: {get_memory_usage_str(self.process)}")
        print("-" * 40)


        LOSS_THRESHOLD = 1e3
        AWFUL_SCORE_ON_DIVERGENCE = -1e9


        start_time_train = time.time()
        diverged = False
        cycle_pbar = tqdm(range(self.current_cycle, self.num_cycles),
                          total=self.num_cycles, initial=self.current_cycle,
                          desc=f"T{self.trial_number} Cyc", unit="cyc")

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
                    last_info = info
                    self.agent.replay_buffer.push(state_flat, action, reward, next_state_flat, done)
                    episode_transitions.append((state_flat, action, reward, next_state_flat, done))
                    state_flat = next_state_flat
                    ep_reward += reward; ep_steps += 1
                    self.total_steps_run += 1

                # --- Post-Episode Processing ---
                self.total_episodes_run += 1
                self.episode_rewards.append(ep_reward)
                cycle_episode_rewards.append(ep_reward)

                episode_goal_met = last_info.get('goal_met', False)
                self.episode_success_flags.append(episode_goal_met)
                cycle_success_flags_in_cycle.append(episode_goal_met)
                if episode_goal_met:
                    self.total_successful_episodes += 1

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
                        break
                if (opt_step_count + 1) % 50 == 0: # Soft update target network periodically
                    self.agent.soft_update_target()
            if cycle_diverged_this_cycle:
                break

            # --- Post-Cycle Reporting & Checkpoint ---
            cycle_duration = time.time() - start_time_cycle
            self.cycle_times.append(cycle_duration)
            if self.agent.lr_scheduler:
                self.agent.lr_scheduler.step()

            avg_rew = np.mean(cycle_episode_rewards) if cycle_episode_rewards else 0.0
            avg_loss = np.mean(cycle_losses) if cycle_losses else 0.0
            cycle_success_rate = (sum(cycle_success_flags_in_cycle) / len(cycle_success_flags_in_cycle) * 100) \
                                 if cycle_success_flags_in_cycle else 0.0

            cycle_pbar.set_postfix({
                "AvgRew": f"{avg_rew:.2f}",
                "Succ(%)": f"{cycle_success_rate:.1f}",
                "AvgLoss": f"{avg_loss:.4f}",
                "Eps": f"{self.agent.epsilon:.3f}",
                "LR": f"{self.agent.get_current_lr():.1e}",
                "Buf": f"{len(self.agent.replay_buffer)/self.agent.replay_buffer.capacity:.1%}",
                "Time": f"{cycle_duration:.1f}s"
            }, refresh=True)

            if (cycle_idx + 1) % (max(1, self.num_cycles // 10)) == 0:
                print(f"\n End Cyc {cycle_idx+1}. Mem: {get_memory_usage_str(self.process)}")

            self._save_checkpoint()

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
            self.final_avg_score = AWFUL_SCORE_ON_DIVERGENCE
            self.final_window_success_rate = 0.0
        else:
            score_window_size = min(1000, self.total_episodes_run) # Or some other reasonable number
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