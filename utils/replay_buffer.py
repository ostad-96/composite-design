# utils/replay_buffer.py

import numpy as np
import torch
from pathlib import Path
import os
import json
import math # For sqrt
import time
import ctypes
import platform

def _lock_pages(arrays):
    system_platform = platform.system()
    if system_platform == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            mlock = libc.mlock
            mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; mlock.restype = ctypes.c_int
        except OSError as e:
            print(f"[ReplayBuffer] Warning: Failed to load libc.so.6 for mlock: {e}. Page locking disabled.")
            return
        total = 0
        for arr in arrays:
            addr = ctypes.c_void_p(arr.ctypes.data); length = ctypes.c_size_t(arr.nbytes)
            ret = mlock(addr, length)
            if ret != 0:
                err = ctypes.get_errno()
                error_message = f"mlock failed (ret: {ret}, errno: {err}) on {arr.nbytes/1e6:.1f}MB: {os.strerror(err)}."
                if err == getattr(ctypes.errno, 'EPERM', -1): error_message += " Insufficient permissions or ulimit -l too low."
                elif err == getattr(ctypes.errno, 'ENOMEM', -1): error_message += " ulimit -l too low or insufficient RAM."
                print(f"[ReplayBuffer] Warning: {error_message} Locking might not be effective.")
            else: total += arr.nbytes
        if total > 0: print(f"[ReplayBuffer] Locked {total/1e6:.1f} MB in RAM (Linux).")
    elif system_platform == "Darwin":
        print("[ReplayBuffer] Info: mlock via ctypes/libc.so.6 is not attempted on macOS. Pages not locked.")
    else:
        print(f"[ReplayBuffer] Info: Page locking designed for Linux. Platform '{system_platform}'. Skipping.")

class ReplayBuffer:
    STATE_FILENAME = "buffer_state.json"
    METRIC_INDICES_FOR_FCN_CHANNELS = [0, 1, 2] # current_VF, scaled_step, scaled_dist_E

    def __init__(
        self,
        flat_state_dim: int, H: int, W: int, num_scalar_metrics: int = 4,
        fcn_output_channels_from_buffer: int = 4, capacity: int = 1_000_000,
        directory: str | Path = "replay_buffer", device: str | torch.device = "cpu",
        pin_memory: bool = True,
    ) -> None:
        self.capacity = int(capacity); self.H = H; self.W = W
        self.flat_grid_dim = H * W; self.num_scalar_metrics = num_scalar_metrics
        if flat_state_dim != (self.flat_grid_dim + self.num_scalar_metrics):
            raise ValueError(f"flat_state_dim mismatch")
        self.fcn_output_channels_from_buffer = fcn_output_channels_from_buffer
        if fcn_output_channels_from_buffer != (1 + len(self.METRIC_INDICES_FOR_FCN_CHANNELS)):
            print(f"Warning: fcn_output_channels_from_buffer mismatch. Check config.")

        self.device = torch.device(device)
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.pos = 0; self.size = 0

        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.state_file_path = directory / self.STATE_FILENAME
        self._load_state_from_file()

        max_action_value = self.flat_grid_dim
        if max_action_value <= 255: self.action_dtype = np.uint8
        elif max_action_value <= 65535: self.action_dtype = np.uint16
        else: self.action_dtype = np.int32

        # --- Validity Test Flags ---
        self.debug_trigger_size = 100000 # Target size for first debug print
        self.debug_triggered_on_size = False
        self.debug_trigger_pos_cycle = 50000 # Target pos for overwrite debug print
        self.debug_triggered_on_pos_cycle = False
        # --- End Validity Test Flags ---

        def mmap_file(name, dtype, shape): # Renamed for clarity
            path = directory / f"{name}.dat"
            file_exists = path.exists(); file_size = os.path.getsize(path) if file_exists else 0
            mode = 'r+' if file_exists and file_size > 0 else 'w+'
            expected_bytes = np.dtype(dtype).itemsize * np.prod(shape)
            if file_exists and file_size != expected_bytes:
                print(f"Warning: Memmap {path} size mismatch (got {file_size}, exp {expected_bytes}). Recreating.")
                mode = 'w+'; self.pos = 0; self.size = 0; self._delete_state_file()
            try:
                if mode == 'w+': path.parent.mkdir(parents=True, exist_ok=True)
                return np.memmap(path, dtype=dtype, mode=mode, shape=shape, order="C")
            except Exception as e: print(f"Error mmap {path} mode {mode}: {e}"); raise

        self.grid_states = mmap_file("grid_states", np.uint8, (self.capacity, 1, self.H, self.W))
        self.next_grid_states = mmap_file("next_grid_states", np.uint8, (self.capacity, 1, self.H, self.W))
        self.metric_states = mmap_file("metric_states", np.float32, (self.capacity, self.num_scalar_metrics))
        self.next_metric_states = mmap_file("next_metric_states", np.float32, (self.capacity, self.num_scalar_metrics))
        self.actions = mmap_file("actions", self.action_dtype, (self.capacity,))
        self.rewards = mmap_file("rewards", np.float32, (self.capacity,))
        self.dones = mmap_file("dones", np.uint8, (self.capacity,))

        if platform.system() == "Linux":
            _lock_pages([self.grid_states, self.metric_states, self.next_grid_states,
                         self.next_metric_states, self.actions, self.rewards, self.dones])
        # else: print(f"[ReplayBuffer] Info: Page locking only on Linux. System: {platform.system()}.")
        print(f"[ReplayBuffer] Initialized. Capacity={self.capacity}, Size={self.size}, Pos={self.pos}, Dir='{self.directory}'")


    def _print_buffer_debug_info(self, index: int, trigger_name: str):
        """Helper to print debug information for a specific index."""
        print(f"\n--- Replay Buffer Validity Check ({trigger_name} for index {index}) ---")
        print(f" Buffer current state -> pos: {self.pos}, size: {self.size}, capacity: {self.capacity}")
        # Safely access data, as index might be for a just-overwritten slot if pos cycled
        action_val = self.actions[index] if index < self.capacity else "N/A (index out of bounds)"
        reward_val = self.rewards[index] if index < self.capacity else "N/A"
        done_val = self.dones[index] if index < self.capacity else "N/A"

        print(f" Data at index {index}:")
        print(f"  Action stored: {action_val}")
        print(f"  Reward stored: {reward_val}")
        print(f"  Done stored: {done_val}")

        if index < self.capacity:
            grid_slice = self.grid_states[index][0, :min(3, self.H), :min(3, self.W)]
            print(f"  Grid stored (shape {self.grid_states[index].shape}, dtype {self.grid_states[index].dtype}), top-left 3x3:\n{grid_slice}")
            print(f"  Metrics stored (shape {self.metric_states[index].shape}): {self.metric_states[index]}")

            next_grid_slice = self.next_grid_states[index][0, :min(3, self.H), :min(3, self.W)]
            print(f"  Next Grid stored (shape {self.next_grid_states[index].shape}), top-left 3x3:\n{next_grid_slice}")
            print(f"  Next Metrics stored (shape {self.next_metric_states[index].shape}): {self.next_metric_states[index]}")
        if self.size > 20: # Print another random sample for comparison
            rand_idx = np.random.randint(0, self.size)
            print(f"  Random previous item at index {rand_idx}: Action={self.actions[rand_idx]}, Reward={self.rewards[rand_idx]}")
        print("--- End Replay Buffer Check ---")


    def push(self, flat_state: np.ndarray, action: int, reward: float, next_flat_state: np.ndarray, done: bool):
        if flat_state.shape[0] != (self.flat_grid_dim + self.num_scalar_metrics) or \
           next_flat_state.shape[0] != (self.flat_grid_dim + self.num_scalar_metrics):
            raise ValueError(f"State dim mismatch in push")
        current_write_idx = self.pos # Index where data will be written

        grid_part_flat = flat_state[:self.flat_grid_dim]
        metrics_part_scalar = flat_state[self.flat_grid_dim:]
        self.grid_states[current_write_idx] = grid_part_flat.reshape(1, self.H, self.W).astype(np.uint8)
        self.metric_states[current_write_idx] = metrics_part_scalar.astype(np.float32)

        next_grid_part_flat = next_flat_state[:self.flat_grid_dim]
        next_metrics_part_scalar = next_flat_state[self.flat_grid_dim:]
        self.next_grid_states[current_write_idx] = next_grid_part_flat.reshape(1, self.H, self.W).astype(np.uint8)
        self.next_metric_states[current_write_idx] = next_metrics_part_scalar.astype(np.float32)

        self.actions[current_write_idx] = self.action_dtype(action)
        self.rewards[current_write_idx] = float(reward)
        self.dones[current_write_idx] = 1 if done else 0

        self.pos = (current_write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # --- Validity Test Logic ---
        if not self.debug_triggered_on_size and self.size == self.debug_trigger_size:
            self._print_buffer_debug_info(current_write_idx, f"Size Trigger ({self.debug_trigger_size})")
            self.debug_triggered_on_size = True
        # Check if pos hits the cycle trigger *and* we are at full capacity (meaning overwriting)
        # current_write_idx is the slot that was just written to.
        if not self.debug_triggered_on_pos_cycle and \
           current_write_idx == self.debug_trigger_pos_cycle and \
           self.size == self.capacity:
            self._print_buffer_debug_info(current_write_idx, f"Position Cycle Trigger (pos={self.debug_trigger_pos_cycle} overwritten)")
            self.debug_triggered_on_pos_cycle = True
        # --- End Validity Test Logic ---


    def sample(self, batch_size: int): # Same as your last version
        if self.size == 0: raise RuntimeError("Cannot sample from empty buffer.")
        effective_batch_size = min(batch_size, self.size)
        idx_random = np.random.randint(0, self.size, size=effective_batch_size)
        order = np.argsort(idx_random); idx_sorted = idx_random[order]; inv_order = np.argsort(order)

        grid_s_memmap = self.grid_states[idx_sorted]
        metric_s_memmap = self.metric_states[idx_sorted]
        next_grid_s_memmap = self.next_grid_states[idx_sorted]
        next_metric_s_memmap = self.next_metric_states[idx_sorted]
        actions_memmap = self.actions[idx_sorted]
        rewards_memmap = self.rewards[idx_sorted]
        dones_memmap = self.dones[idx_sorted]

        grid_s_tensor = self._to_tensor(grid_s_memmap, torch.float32)
        metric_s_tensor = self._to_tensor(metric_s_memmap, torch.float32)
        next_grid_s_tensor = self._to_tensor(next_grid_s_memmap, torch.float32)
        next_metric_s_tensor = self._to_tensor(next_metric_s_memmap, torch.float32)
        a_tensor = self._to_tensor(actions_memmap, torch.long).unsqueeze(1)
        r_tensor = self._to_tensor(rewards_memmap, torch.float32).unsqueeze(1)
        d_tensor = self._to_tensor(dones_memmap, torch.float32).unsqueeze(1)

        def construct_fcn_state(grid_tensor_bchw, metrics_tensor_b_numscalar, current_batch_s):
            broadcasted_metric_channels = []
            for metric_idx in self.METRIC_INDICES_FOR_FCN_CHANNELS:
                if metric_idx >= metrics_tensor_b_numscalar.shape[1]:
                    raise IndexError(f"FCN metric index {metric_idx} OOB for stored scalar metrics (count: {metrics_tensor_b_numscalar.shape[1]}).")
                scalar_values = metrics_tensor_b_numscalar[:, metric_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                broadcasted_metric_channels.append(scalar_values.expand(current_batch_s, 1, self.H, self.W))
            return torch.cat([grid_tensor_bchw] + broadcasted_metric_channels, dim=1)

        s_fcn = construct_fcn_state(grid_s_tensor, metric_s_tensor, effective_batch_size)
        ns_fcn = construct_fcn_state(next_grid_s_tensor, next_metric_s_tensor, effective_batch_size)

        s_fcn, ns_fcn = s_fcn[inv_order], ns_fcn[inv_order]
        a_tensor,r_tensor,d_tensor = a_tensor[inv_order],r_tensor[inv_order],d_tensor[inv_order]
        s_fcn = s_fcn.to(self.device, non_blocking=(self.pin_memory and s_fcn.is_pinned()))
        ns_fcn = ns_fcn.to(self.device, non_blocking=(self.pin_memory and ns_fcn.is_pinned()))
        a_tensor = a_tensor.to(self.device); r_tensor = r_tensor.to(self.device); d_tensor = d_tensor.to(self.device)
        return s_fcn, a_tensor, r_tensor, ns_fcn, d_tensor

    def _to_tensor(self, arr, dtype): # Same as your last version
        contiguous_arr = np.asarray(arr, order='C'); t = torch.as_tensor(contiguous_arr, dtype=dtype, device="cpu")
        if self.pin_memory:
            try: return t.pin_memory()
            except RuntimeError: return t # Ignore pinning error
            except Exception: return t
        return t

    def __len__(self): return self.size # Same
    def flush(self): # Same
        memmap_attrs = ["grid_states", "metric_states", "next_grid_states", "next_metric_states", "actions", "rewards", "dones"]
        for attr_name in memmap_attrs:
            if hasattr(self, attr_name):
                m_obj = getattr(self, attr_name)
                if isinstance(m_obj, np.memmap) and hasattr(m_obj, 'flush') and callable(m_obj.flush):
                    try: m_obj.flush()
                    except ValueError: pass # Ignore if already closed

    def close(self): # Same
        self.flush(); self.save_state_to_file()
        memmap_attrs = ["grid_states", "metric_states", "next_grid_states", "next_metric_states", "actions", "rewards", "dones"]
        for attr_name in memmap_attrs:
            if hasattr(self, attr_name):
                try: delattr(self, attr_name)
                except AttributeError: pass

    def _load_state_from_file(self): # Same
        try:
            if self.state_file_path.exists():
                with open(self.state_file_path, 'r') as f: state_data = json.load(f)
                loaded_pos = int(state_data.get('pos', 0)); loaded_size = int(state_data.get('size', 0))
                if 0 <= loaded_pos < self.capacity and 0 <= loaded_size <= self.capacity:
                    self.pos, self.size = loaded_pos, loaded_size
                else: self._delete_state_file()
        except (json.JSONDecodeError, Exception): self._delete_state_file()

    def save_state_to_file(self): # Same
        state_data = {'pos': self.pos, 'size': self.size}
        temp_filename = self.state_file_path.with_suffix(".tmp")
        try:
            self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_filename, 'w') as f: json.dump(state_data, f)
            os.replace(temp_filename, self.state_file_path)
        except Exception as e:
            if temp_filename.exists():
                try: os.remove(temp_filename)
                except OSError: pass

    def _delete_state_file(self): # Same
        try:
            if self.state_file_path.exists(): os.remove(self.state_file_path)
        except OSError: pass

    def get_buffer_state_for_checkpoint(self) -> dict: # Same
        return {'capacity': self.capacity, 'pos': self.pos, 'size': self.size, 'H': self.H, 'W': self.W,
                'num_scalar_metrics': self.num_scalar_metrics,
                'fcn_output_channels_from_buffer': self.fcn_output_channels_from_buffer,
                'action_dtype_str': np.dtype(self.action_dtype).name, 'directory': str(self.directory)}

    def load_buffer_state_from_checkpoint(self, buffer_state: dict): # Same
        print(f"Loading buffer state from CHKPT: pos={buffer_state['pos']}, size={buffer_state['size']}")
        if self.capacity != buffer_state.get('capacity'): raise ValueError("Capacity mismatch")
        if self.H != buffer_state.get('H') or self.W != buffer_state.get('W'): raise ValueError("Grid H/W mismatch")
        if self.num_scalar_metrics != buffer_state.get('num_scalar_metrics'): raise ValueError("num_scalar_metrics mismatch")
        if self.fcn_output_channels_from_buffer != buffer_state.get('fcn_output_channels_from_buffer'):
            raise ValueError("fcn_output_channels mismatch")
        self.pos = buffer_state['pos']; self.size = buffer_state['size']
        print(f"Buffer state updated. Pos={self.pos}, Size={self.size}")