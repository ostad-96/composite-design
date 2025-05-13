# project_root/utils/dqn.py
import numpy as np
import os
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.nn.functional import smooth_l1_loss, mse_loss # <<< Added mse_loss
from torch.nn.utils import clip_grad_norm_

import time
from utils.fem import voigt_model, reuss_model
from utils.replay_buffer import ReplayBuffer
from config import (
    MATRIX_SIZE, MAX_STEPS, LEARNING_RATE, BATCH_SIZE, GAMMA,
    EPSILON_DECAY, EPSILON_MIN, TAU, E_STIFF, E_COMP,
    FCN_INPUT_CHANNELS,
    FCN_NUM_FILTERS_RESBLOCK, FCN_NUM_RES_BLOCKS, FCN_KERNEL_SIZE,
    REPLAY_BUFFER_CAPACITY, REPLAY_BUFFER_DIR,
    CLIP_GRAD_NORM_MAX
)


# --- CompositeDesignEnv class remains the same ---
class CompositeDesignEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        super(CompositeDesignEnv, self).__init__()
        self.grid_H = MATRIX_SIZE
        self.grid_W = MATRIX_SIZE
        self.num_cells = self.grid_H * self.grid_W
        self.max_steps = MAX_STEPS
        self.max_modulus = max(E_STIFF, E_COMP)
        self.num_scalar_metrics = 4 # VF, step_scaled, distE_scaled, target_VF
        self.desired_modulus = None
        self.desired_vol_frac = None
        self.action_space = spaces.Discrete(self.num_cells)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_cells + self.num_scalar_metrics,),
            dtype=np.float32
        )
        self.grid = None
        self.current_step = 0
        self.current_modulus = None
        self.current_vol_frac = None
        self.plot_save_dir = "composite_designs_fcn"
        os.makedirs(self.plot_save_dir, exist_ok=True)
        self.render_counter = 0
        self.saved_designs_in_episode = set()
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self):
        self.saved_designs_in_episode.clear()
        self.grid = self.np_random.integers(0, 2, size=(self.grid_H, self.grid_W))
        self.current_step = 0
        self.current_vol_frac = (self.num_cells - np.sum(self.grid)) / self.num_cells
        self.current_modulus = (voigt_model(self.current_vol_frac) + reuss_model(self.current_vol_frac)) / 2
        min_modulus_diff = 100
        while True:
            phi_goal = self.np_random.uniform(0.05, 0.95)
            E_voigt_goal = voigt_model(phi_goal)
            E_reuss_goal = reuss_model(phi_goal)
            if np.isnan(E_reuss_goal):
                E_reuss_goal = E_voigt_goal
            desired_modulus = (E_voigt_goal + E_reuss_goal) / 2
            if abs(self.current_modulus - desired_modulus) >= min_modulus_diff:
                break
        self.desired_modulus = desired_modulus
        self.desired_vol_frac = phi_goal
        return self._get_flat_state()

    def _get_flat_state(self):
        grid_flat = self.grid.flatten().astype(np.float32)
        current_vf_metric = self.current_vol_frac
        scaled_step_metric = self.current_step / self.max_steps if self.max_steps > 0 else 0.0
        dist_e = self.current_modulus - self.desired_modulus
        scaled_dist_e_metric = np.clip(dist_e / self.max_modulus if self.max_modulus > 0 else 0.0, -1.0, 1.0)
        target_vf_metric = self.desired_vol_frac
        scalar_metrics = np.array([
            current_vf_metric, scaled_step_metric,
            scaled_dist_e_metric, target_vf_metric
        ], dtype=np.float32)
        return np.concatenate([grid_flat, scalar_metrics])

    def compute_reward(self, current_modulus, current_vol_frac, desired_modulus, desired_vol_frac, weight_E=1.0, weight_Vf=5.0):
        r_modulus = -abs(current_modulus - desired_modulus ) / self.max_modulus
        r_vol_frac = -abs(current_vol_frac - desired_vol_frac)
        return (weight_E * r_modulus) + (weight_Vf * r_vol_frac)

    def _check_goal_met(self, current_modulus, current_vol_frac, desired_modulus, desired_vol_frac):
        modulus_tolerance = 50
        vf_tolerance = 0.04
        modulus_met = abs(current_modulus - desired_modulus) <= modulus_tolerance
        vf_met = abs(current_vol_frac - desired_vol_frac) <= vf_tolerance
        return bool(modulus_met and vf_met)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        self.current_step += 1
        row, col = action // self.grid_W, action % self.grid_W
        self.grid[row, col] = 1 - self.grid[row, col]
        self.current_vol_frac = (self.num_cells - np.sum(self.grid)) / self.num_cells
        self.current_modulus = (voigt_model(self.current_vol_frac) + reuss_model(self.current_vol_frac)) / 2
        reward = self.compute_reward(self.current_modulus, self.current_vol_frac, self.desired_modulus, self.desired_vol_frac)
        done = (self.current_step >= self.max_steps)
        goal_met_this_step = self._check_goal_met(
            self.current_modulus, self.current_vol_frac,
            self.desired_modulus, self.desired_vol_frac
        )
        info = {
            'current_modulus': self.current_modulus,
            'current_vol_frac': self.current_vol_frac,
            'target_modulus': self.desired_modulus,
            'target_vol_frac': self.desired_vol_frac,
            'goal_met': goal_met_this_step
        }
        return self._get_flat_state(), reward, done, info

    def render(self, mode='human'):
        design_tuple = tuple(self.grid.flatten())
        if design_tuple in self.saved_designs_in_episode and mode !='rgb_array':
            return
        self.saved_designs_in_episode.add(design_tuple)
        fig, ax = plt.subplots()
        ax.imshow(self.grid, cmap='gray', vmin=0, vmax=1)
        title_str = (
            f"S:{self.current_step} E:{self.current_modulus:.0f} VF:{self.current_vol_frac:.2f}\n"
            f"Tg E:{self.desired_modulus:.0f} VF:{self.desired_vol_frac:.2f}"
        )
        ax.set_title(title_str, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        self.render_counter += 1
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"design_T{self.render_counter}_{timestamp}.png"
        file_path = os.path.join(self.plot_save_dir, filename)
        if mode == 'human':
            plt.show(block=False); plt.pause(0.1)
        elif mode == 'rgb_array':
            fig.canvas.draw(); img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,)); plt.close(fig); return img
        elif mode == 'save':
            plt.savefig(file_path, dpi=100); plt.close(fig)
        else:
            super(CompositeDesignEnv, self).render(mode=mode)
        if mode != 'rgb_array':
            plt.close(fig)

    def close(self):
        plt.close('all')


class ResBlock(nn.Module):
    def __init__(self, num_filters: int, kernel_size: int = 3, dilation: int = 1):
        super(ResBlock, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ResBlock kernel_size must be odd to use standard padding formula.")
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        out = out + identity
        out = self.relu(out)
        return out


class PixelQNetwork(nn.Module):
    def __init__(self, input_channels: int = FCN_INPUT_CHANNELS, H: int = MATRIX_SIZE, W: int = MATRIX_SIZE,
                 num_filters_resblock: int = FCN_NUM_FILTERS_RESBLOCK, num_res_blocks: int = FCN_NUM_RES_BLOCKS,
                 kernel_size_resblock: int = FCN_KERNEL_SIZE, dilation_factors_per_block: list[int] | None = None):
        super(PixelQNetwork, self).__init__()
        if dilation_factors_per_block is None:
            dilation_factors_per_block = [1] * num_res_blocks
        elif len(dilation_factors_per_block) != num_res_blocks:
            raise ValueError(f"Length of dilation_factors_per_block ({len(dilation_factors_per_block)}) "
                             f"must match num_res_blocks ({num_res_blocks}).")

        self.initial_processing = nn.Sequential(
            nn.Conv2d(input_channels, num_filters_resblock, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters_resblock),
            nn.ReLU(inplace=True)
        )
        res_layers = []
        for i in range(num_res_blocks):
            res_layers.append(ResBlock(num_filters=num_filters_resblock, kernel_size=kernel_size_resblock,
                                       dilation=dilation_factors_per_block[i]))
        self.res_blocks = nn.Sequential(*res_layers)
        self.final_conv = nn.Conv2d(num_filters_resblock, 1, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.initial_processing(x)
        x = self.res_blocks(x)
        q_map = self.final_conv(x)
        return q_map


class DQNAgent:
    def __init__(
            self,
            device: torch.device,
            *,
            matrix_size: int = MATRIX_SIZE,
            fcn_input_channels: int = FCN_INPUT_CHANNELS,
            num_scalar_metrics_env: int = 4,
            lr: float = LEARNING_RATE,
            gamma: float = GAMMA,
            batch_size: int = BATCH_SIZE,
            # FCN architecture HPs
            fcn_num_filters: int = FCN_NUM_FILTERS_RESBLOCK,
            fcn_blocks: int = FCN_NUM_RES_BLOCKS,
            fcn_kernel_size: int = FCN_KERNEL_SIZE,
            fcn_dilation_factors: list[int],
            epsilon_start: float = 1.0,
            epsilon_decay: float = EPSILON_DECAY,
            epsilon_min: float = EPSILON_MIN,
            tau: float = TAU,
            use_lr_scheduler: bool = True,
            lr_end_factor: float = 0.1,
            lr_decay_cycles: int = 100,
            buffer_capacity: int = REPLAY_BUFFER_CAPACITY,
            buffer_dir: str = REPLAY_BUFFER_DIR,
            clip_grad_norm_max: float = CLIP_GRAD_NORM_MAX,
            loss_function_name: str = "smooth_l1"  # <<< Added new parameter with default
    ):
        self.device = device
        self.H = matrix_size; self.W = matrix_size
        self.fcn_input_C = fcn_input_channels
        self.action_dim = self.H * self.W
        self.flat_grid_dim_for_buffer = self.H * self.W
        self.num_scalar_metrics_for_buffer = num_scalar_metrics_env
        self.flat_state_dim_for_buffer = self.flat_grid_dim_for_buffer + self.num_scalar_metrics_for_buffer

        self.lr, self.gamma, self.batch_size = lr, gamma, batch_size
        self.fcn_num_filters = fcn_num_filters
        self.fcn_blocks = fcn_blocks
        self.fcn_kernel_size = fcn_kernel_size
        self.fcn_dilation_factors = fcn_dilation_factors
        self.epsilon_start, self.epsilon_decay, self.epsilon_min, self.tau = epsilon_start, epsilon_decay, epsilon_min, tau
        self.use_lr_scheduler, self.lr_end_factor, self.lr_decay_cycles = use_lr_scheduler, lr_end_factor, lr_decay_cycles
        self.buffer_capacity, self.buffer_dir = buffer_capacity, buffer_dir
        self.clip_grad_norm_max = clip_grad_norm_max
        self.loss_function_name = loss_function_name # <<< Store loss function name

        if self.loss_function_name == "smooth_l1":
            self.loss_fn = smooth_l1_loss
        elif self.loss_function_name == "mse":
            self.loss_fn = mse_loss
        else:
            raise ValueError(f"Unsupported loss_function_name: {self.loss_function_name}. Choose 'smooth_l1' or 'mse'.")

        if len(fcn_dilation_factors) != fcn_blocks:
            raise ValueError("Length of fcn_dilation_factors must match fcn_blocks (num_res_blocks).")

        self.main_net = PixelQNetwork(
            input_channels=self.fcn_input_C, H=self.H, W=self.W,
            num_filters_resblock=self.fcn_num_filters, num_res_blocks=self.fcn_blocks,
            kernel_size_resblock=self.fcn_kernel_size, dilation_factors_per_block=self.fcn_dilation_factors
        ).to(device)
        self.target_net = PixelQNetwork(
            input_channels=self.fcn_input_C, H=self.H, W=self.W,
            num_filters_resblock=self.fcn_num_filters, num_res_blocks=self.fcn_blocks,
            kernel_size_resblock=self.fcn_kernel_size, dilation_factors_per_block=self.fcn_dilation_factors
        ).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict()); self.target_net.eval()

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        self.lr_scheduler = None
        if use_lr_scheduler and lr_decay_cycles > 0:
            self.lr_scheduler = LinearLR(self.optimizer, 1.0, lr_end_factor, lr_decay_cycles)

        os.makedirs(buffer_dir, exist_ok=True)
        self.replay_buffer = ReplayBuffer(
            flat_state_dim=self.flat_state_dim_for_buffer, H=self.H, W=self.W,
            num_scalar_metrics=self.num_scalar_metrics_for_buffer,
            fcn_output_channels_from_buffer=self.fcn_input_C,
            capacity=buffer_capacity, directory=buffer_dir, device=device,
            pin_memory=(device.type == 'cuda')
        )
        self.epsilon = epsilon_start

    def _convert_flat_single_state_to_fcn_input(self, flat_state_numpy: np.ndarray) -> torch.Tensor:
        grid_flat = flat_state_numpy[:self.flat_grid_dim_for_buffer]
        metrics_scalar = flat_state_numpy[self.flat_grid_dim_for_buffer:]
        grid_tensor = torch.from_numpy(grid_flat.astype(np.float32)).view(1, 1, self.H, self.W)
        broadcasted_metric_channels = []
        for metric_idx in ReplayBuffer.METRIC_INDICES_FOR_FCN_CHANNELS:
            scalar_val = torch.tensor(metrics_scalar[metric_idx], dtype=torch.float32).view(1,1,1,1)
            broadcasted_metric_channels.append(scalar_val.expand(1, 1, self.H, self.W))
        fcn_input = torch.cat([grid_tensor] + broadcasted_metric_channels, dim=1)
        return fcn_input.to(self.device)

    def select_action(self, flat_state_from_env: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            fcn_input_state = self._convert_flat_single_state_to_fcn_input(flat_state_from_env)
            self.main_net.eval()
            with torch.no_grad():
                q_map = self.main_net(fcn_input_state)
            self.main_net.train()
            return q_map.view(1, -1).argmax().item()

    def update(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states_chw, actions, rewards, next_states_chw, dones = self.replay_buffer.sample(self.batch_size)

        self.main_net.eval(); self.target_net.eval()
        with torch.no_grad():
            next_q_maps_main = self.main_net(next_states_chw)
            next_actions_indices = next_q_maps_main.view(self.batch_size, -1).argmax(dim=1, keepdim=True)
            next_q_maps_target = self.target_net(next_states_chw)
            next_q_values = next_q_maps_target.view(self.batch_size, -1).gather(1, next_actions_indices)
            target_q = rewards + self.gamma * next_q_values * (1.0 - dones)

        self.main_net.train()
        current_q_maps_main = self.main_net(states_chw)
        current_q_for_actions = current_q_maps_main.view(self.batch_size, -1).gather(1, actions)

        loss = self.loss_fn(current_q_for_actions, target_q) # <<< Use selected loss_fn

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.main_net.parameters(), self.clip_grad_norm_max)
        self.optimizer.step()
        return loss.item()

    def soft_update_target(self):
        for tp, mp in zip(self.target_net.parameters(), self.main_net.parameters()):
            tp.data.copy_(self.tau * mp.data + (1.0 - self.tau) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_state(self) -> dict:
        buffer_chkpt_state = self.replay_buffer.get_buffer_state_for_checkpoint()
        agent_specific_hps = {
            'matrix_size': self.H,
            'fcn_input_channels': self.fcn_input_C,
            'num_scalar_metrics_env': self.num_scalar_metrics_for_buffer,
            'lr': self.lr,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'fcn_num_filters': self.fcn_num_filters,
            'fcn_blocks': self.fcn_blocks,
            'fcn_kernel_size': self.fcn_kernel_size,
            'fcn_dilation_factors': self.fcn_dilation_factors,
            'epsilon_start': self.epsilon_start,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'tau': self.tau,
            'use_lr_scheduler': self.use_lr_scheduler,
            'lr_end_factor': self.lr_end_factor,
            'lr_decay_cycles': self.lr_decay_cycles,
            'flat_state_dim_for_buffer': self.flat_state_dim_for_buffer,
            'action_dim': self.action_dim,
            'clip_grad_norm_max': self.clip_grad_norm_max,
            'loss_function_name': self.loss_function_name # <<< Add to state
        }
        agent_state = {
            'main_net_state_dict': self.main_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'buffer_checkpoint_state': buffer_chkpt_state,
            'agent_hyperparameters': agent_specific_hps
        }
        if self.lr_scheduler:
            agent_state['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        return agent_state

    def load_state(self, agent_state: dict):
        print("Loading agent state from FCN checkpoint...")
        self.main_net.load_state_dict(agent_state['main_net_state_dict'])
        self.target_net.load_state_dict(agent_state['target_net_state_dict'])
        # Optimizer state is loaded later after potential HP updates
        self.epsilon = agent_state['epsilon']
        loaded_agent_hps = agent_state.get('agent_hyperparameters', {})

        # --- Critical Architectural HP Checks ---
        mismatched_critical = []
        if self.H != loaded_agent_hps.get('matrix_size'): mismatched_critical.append("matrix_size")
        if self.fcn_input_C != loaded_agent_hps.get('fcn_input_channels'): mismatched_critical.append("fcn_input_channels")
        if self.fcn_blocks != loaded_agent_hps.get('fcn_blocks'): mismatched_critical.append("fcn_blocks")
        if self.fcn_kernel_size != loaded_agent_hps.get('fcn_kernel_size'): mismatched_critical.append("fcn_kernel_size")
        if list(self.fcn_dilation_factors) != list(loaded_agent_hps.get('fcn_dilation_factors', [])): mismatched_critical.append("fcn_dilation_factors")
        # Consider loss_function_name critical? If changed, optimizer was trained with different objective.
        # For now, let's treat it as non-critical that gets updated.
        # If desired to be critical, add:
        # if self.loss_function_name != loaded_agent_hps.get('loss_function_name'): mismatched_critical.append("loss_function_name")


        if mismatched_critical:
            print("CRITICAL HP MISMATCHES PREVENTING LOAD:")
            for key in mismatched_critical:
                current_val_str = f"'{getattr(self, key, 'N/A')}'"
                chkpt_val_str = f"'{loaded_agent_hps.get(key)}'"
                print(f" - {key}: Current={current_val_str}, Checkpoint={chkpt_val_str}")
            raise ValueError(f"Critical architectural hyperparameter mismatch: {', '.join(mismatched_critical)}")

        # --- Non-critical HP update/check ---
        non_critical_hps_to_update = [
            'lr', 'gamma', 'batch_size', 'epsilon_start', 'epsilon_decay',
            'epsilon_min', 'tau', 'use_lr_scheduler', 'lr_end_factor',
            'lr_decay_cycles', 'clip_grad_norm_max', 'loss_function_name' # <<< Added
        ]
        for hp_key in non_critical_hps_to_update:
            loaded_val = loaded_agent_hps.get(hp_key)
            current_val = getattr(self, hp_key, None)
            if loaded_val is not None and loaded_val != current_val:
                print(f" Info: Updating HP '{hp_key}' from checkpoint: {current_val} -> {loaded_val}")
                setattr(self, hp_key, loaded_val)
                if hp_key == 'loss_function_name': # <<< Re-assign loss_fn if name changed
                    if self.loss_function_name == "smooth_l1":
                        self.loss_fn = smooth_l1_loss
                    elif self.loss_function_name == "mse":
                        self.loss_fn = mse_loss
                    else:
                        # This should ideally not happen if validation was done at init
                        print(f"WARN: Invalid loss_function_name '{self.loss_function_name}' loaded. Defaulting to smooth_l1.")
                        self.loss_fn = smooth_l1_loss


        # Re-initialize optimizer and scheduler if relevant HPs changed
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=self.lr) # Re-init with potentially new lr
        if self.use_lr_scheduler and self.lr_decay_cycles > 0:
            self.lr_scheduler = LinearLR(self.optimizer, 1.0, self.lr_end_factor, self.lr_decay_cycles)

        # Load optimizer state *after* re-init
        if 'optimizer_state_dict' in agent_state:
             self.optimizer.load_state_dict(agent_state['optimizer_state_dict'])

        if 'lr_scheduler_state_dict' in agent_state and self.lr_scheduler:
            try:
                self.lr_scheduler.load_state_dict(agent_state['lr_scheduler_state_dict'])
            except Exception as e:
                print(f"WARN: Failed to load LR scheduler: {e}")

        if 'buffer_checkpoint_state' in agent_state:
            try:
                self.replay_buffer.load_buffer_state_from_checkpoint(agent_state['buffer_checkpoint_state'])
            except ValueError as e:
                print(f"Error loading buffer state via agent: {e}"); raise
        else:
            print("Warning: No 'buffer_checkpoint_state' in agent state.")

        self.target_net.eval()
        print("Agent state loaded (FCN).")

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def close_buffer(self):
        self.replay_buffer.close()