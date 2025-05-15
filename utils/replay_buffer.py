# project_root/utils/replay_buffer.py

import numpy as np
import torch
from pathlib import Path
import os
import json
# import math # Not used
# import time # Not used directly in this snippet, but useful for debugging
import ctypes
import platform

# _lock_pages function remains unchanged
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
        if total > 0: print(f"[ReplayBuffer] Locked {total/1e6:.1f} MB of memmap files in RAM (Linux).")
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
        directory: str | Path = "replay_buffer",
        device: str | torch.device = "cpu", # This is the target TRAINING device for samples
        pin_memory: bool = True, # This will control pinning of RAM tensors
    ) -> None:
        self.capacity = int(capacity); self.H = H; self.W = W
        self.flat_grid_dim = H * W; self.num_scalar_metrics = num_scalar_metrics
        if flat_state_dim != (self.flat_grid_dim + self.num_scalar_metrics):
            raise ValueError(f"flat_state_dim mismatch")
        self.fcn_output_channels_from_buffer = fcn_output_channels_from_buffer
        if fcn_output_channels_from_buffer != (1 + len(self.METRIC_INDICES_FOR_FCN_CHANNELS)):
            print(f"Warning: fcn_output_channels_from_buffer mismatch. Check config.")

        self.ram_tensor_device = torch.device("cpu") # RAM tensors are always on CPU
        self.training_device = torch.device(device)
        self.pin_ram_tensors = pin_memory and self.training_device.type == 'cuda'

        self.pos = 0; self.size = 0

        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.state_file_path = directory / self.STATE_FILENAME
        self._load_state_from_file()

        max_action_value = self.flat_grid_dim
        if max_action_value <= 255:
            self.action_dtype_np = np.uint8
            self.action_dtype_torch = torch.uint8
        elif max_action_value <= 32767: # Max for torch.int16
            self.action_dtype_np = np.uint16 # NumPy can handle full uint16 range
            self.action_dtype_torch = torch.int16
            if max_action_value > 32767:
                 print(f"[ReplayBuffer] Warning: Max action value {max_action_value} > 32767. np.uint16 used for mmap, but torch.int16 for RAM. Ensure actions are representable if conversion occurs.")
        elif max_action_value <= 65535: # np.uint16 can go up to this
            self.action_dtype_np = np.uint16
            self.action_dtype_torch = torch.int16 # PyTorch still uses int16
            print(f"[ReplayBuffer] Warning: Max action value {max_action_value} fits np.uint16. PyTorch uses torch.int16. Ensure actions are non-negative and < 32768 when converting to tensor.")
        else:
            self.action_dtype_np = np.int32
            self.action_dtype_torch = torch.int32


        self.debug_trigger_size = 100000
        self.debug_triggered_on_size = False
        self.debug_trigger_pos_cycle = 50000
        self.debug_triggered_on_pos_cycle = False

        def mmap_file_init(name, dtype_np, shape):
            path = directory / f"{name}.dat"
            file_exists = path.exists(); file_size = os.path.getsize(path) if file_exists else 0
            mode = 'r+' if file_exists and file_size > 0 else 'w+'
            expected_bytes = np.dtype(dtype_np).itemsize * np.prod(shape)
            if file_exists and file_size != expected_bytes:
                print(f"Warning: Memmap {path} size mismatch (got {file_size}, exp {expected_bytes}). Recreating.")
                mode = 'w+'; self.pos = 0; self.size = 0; self._delete_state_file()
            try:
                if mode == 'w+': path.parent.mkdir(parents=True, exist_ok=True)
                return np.memmap(path, dtype=dtype_np, mode=mode, shape=shape, order="C")
            except Exception as e: print(f"Error mmap {path} mode {mode}: {e}"); raise

        self.grid_states_mmap = mmap_file_init("grid_states", np.uint8, (self.capacity, 1, self.H, self.W))
        self.next_grid_states_mmap = mmap_file_init("next_grid_states", np.uint8, (self.capacity, 1, self.H, self.W))
        self.metric_states_mmap = mmap_file_init("metric_states", np.float32, (self.capacity, self.num_scalar_metrics))
        self.next_metric_states_mmap = mmap_file_init("next_metric_states", np.float32, (self.capacity, self.num_scalar_metrics))
        self.actions_mmap = mmap_file_init("actions", self.action_dtype_np, (self.capacity,))
        self.rewards_mmap = mmap_file_init("rewards", np.float32, (self.capacity,))
        self.dones_mmap = mmap_file_init("dones", np.uint8, (self.capacity,))

        # --- RAM Tensor Initialization (Corrected) ---
        # Grid states: np.uint8 -> torch.uint8
        self.grid_states_ram = torch.empty(self.grid_states_mmap.shape, dtype=torch.uint8, device=self.ram_tensor_device)
        self.next_grid_states_ram = torch.empty(self.next_grid_states_mmap.shape, dtype=torch.uint8, device=self.ram_tensor_device)
        
        # Metric states: np.float32 -> torch.float32
        self.metric_states_ram = torch.empty(self.metric_states_mmap.shape, dtype=torch.float32, device=self.ram_tensor_device)
        self.next_metric_states_ram = torch.empty(self.next_metric_states_mmap.shape, dtype=torch.float32, device=self.ram_tensor_device)
        
        # Actions: self.action_dtype_np corresponds to self.action_dtype_torch for RAM
        self.actions_ram = torch.empty(self.actions_mmap.shape, dtype=self.action_dtype_torch, device=self.ram_tensor_device)
        
        # Rewards: np.float32 -> torch.float32
        self.rewards_ram = torch.empty(self.rewards_mmap.shape, dtype=torch.float32, device=self.ram_tensor_device)
        
        # Dones: np.uint8 -> torch.uint8
        self.dones_ram = torch.empty(self.dones_mmap.shape, dtype=torch.uint8, device=self.ram_tensor_device)

        if self.pin_ram_tensors:
            print("[ReplayBuffer] Pinning RAM tensors...")
            try:
                self.grid_states_ram = self.grid_states_ram.pin_memory()
                self.next_grid_states_ram = self.next_grid_states_ram.pin_memory()
                self.metric_states_ram = self.metric_states_ram.pin_memory()
                self.next_metric_states_ram = self.next_metric_states_ram.pin_memory()
                self.actions_ram = self.actions_ram.pin_memory()
                self.rewards_ram = self.rewards_ram.pin_memory()
                self.dones_ram = self.dones_ram.pin_memory()
                print("[ReplayBuffer] RAM tensors pinned.")
            except RuntimeError as e:
                print(f"[ReplayBuffer] Warning: Failed to pin RAM tensors: {e}. Performance may be impacted.")
                self.pin_ram_tensors = False

        if self.size > 0:
            print(f"[ReplayBuffer] Loading initial {self.size} samples from memmap to RAM tensors...")
            self._load_slice_from_mmap_to_ram(0, self.size)
            print(f"[ReplayBuffer] Finished loading initial data to RAM.")

        if platform.system() == "Linux":
            _lock_pages([self.grid_states_mmap, self.metric_states_mmap,
                         self.next_grid_states_mmap, self.next_metric_states_mmap,
                         self.actions_mmap, self.rewards_mmap, self.dones_mmap])

        total_ram_bytes = sum(t.nelement() * t.element_size() for t in
                              [self.grid_states_ram, self.next_grid_states_ram, self.metric_states_ram,
                               self.next_metric_states_ram, self.actions_ram, self.rewards_ram, self.dones_ram])
        print(f"[ReplayBuffer] Initialized. Capacity={self.capacity}, Size={self.size}, Pos={self.pos}, Dir='{self.directory}'")
        print(f"[ReplayBuffer] RAM Tensors: {total_ram_bytes / 1e9:.2f} GB on '{self.ram_tensor_device}', Pinned: {self.pin_ram_tensors}")

    def _load_slice_from_mmap_to_ram(self, start_idx: int, end_idx: int):
        if start_idx >= end_idx: return
        count = end_idx - start_idx
        # print(f"[ReplayBuffer] Syncing {count} mmap samples (idx {start_idx}-{end_idx-1}) to RAM tensors...") # Can be verbose
        with torch.no_grad():
            self.grid_states_ram[start_idx:end_idx].copy_(torch.from_numpy(self.grid_states_mmap[start_idx:end_idx].copy())) # .copy() ensures it's not a view that might get invalidated
            self.next_grid_states_ram[start_idx:end_idx].copy_(torch.from_numpy(self.next_grid_states_mmap[start_idx:end_idx].copy()))
            self.metric_states_ram[start_idx:end_idx].copy_(torch.from_numpy(self.metric_states_mmap[start_idx:end_idx].copy()))
            self.next_metric_states_ram[start_idx:end_idx].copy_(torch.from_numpy(self.next_metric_states_mmap[start_idx:end_idx].copy()))
            # For actions, ensure proper dtype conversion from np to torch
            actions_np_slice = self.actions_mmap[start_idx:end_idx].astype(self.action_dtype_np).copy() # Ensure correct np_dtype before conversion
            self.actions_ram[start_idx:end_idx].copy_(torch.from_numpy(actions_np_slice).to(self.action_dtype_torch))
            self.rewards_ram[start_idx:end_idx].copy_(torch.from_numpy(self.rewards_mmap[start_idx:end_idx].copy()))
            self.dones_ram[start_idx:end_idx].copy_(torch.from_numpy(self.dones_mmap[start_idx:end_idx].copy()))
        # print(f"[ReplayBuffer] Sync complete for slice.")


    def _print_buffer_debug_info(self, index: int, trigger_name: str):
        print(f"\n--- Replay Buffer Validity Check ({trigger_name} for index {index}) ---")
        print(f" Buffer current state -> pos: {self.pos}, size: {self.size}, capacity: {self.capacity}")
        action_val = self.actions_ram[index].item() if index < self.capacity else "N/A (index out of bounds)"
        reward_val = self.rewards_ram[index].item() if index < self.capacity else "N/A"
        done_val = self.dones_ram[index].item() if index < self.capacity else "N/A"
        print(f" Data at index {index} (from RAM tensors):")
        print(f"  Action stored: {action_val}")
        print(f"  Reward stored: {reward_val}")
        print(f"  Done stored: {done_val}")
        if index < self.capacity:
            grid_slice_ram = self.grid_states_ram[index][0, :min(3, self.H), :min(3, self.W)]
            print(f"  Grid stored (RAM shape {self.grid_states_ram[index].shape}, dtype {self.grid_states_ram[index].dtype}), top-left 3x3:\n{grid_slice_ram.cpu().numpy()}")
            print(f"  Metrics stored (RAM shape {self.metric_states_ram[index].shape}): {self.metric_states_ram[index].cpu().numpy()}")
            next_grid_slice_ram = self.next_grid_states_ram[index][0, :min(3, self.H), :min(3, self.W)]
            print(f"  Next Grid stored (RAM shape {self.next_grid_states_ram[index].shape}), top-left 3x3:\n{next_grid_slice_ram.cpu().numpy()}")
            print(f"  Next Metrics stored (RAM shape {self.next_metric_states_ram[index].shape}): {self.next_metric_states_ram[index].cpu().numpy()}")
        if self.size > 20:
            rand_idx = torch.randint(0, self.size, (1,)).item() # Use torch for consistency
            print(f"  Random previous item at RAM index {rand_idx}: Action={self.actions_ram[rand_idx].item()}, Reward={self.rewards_ram[rand_idx].item()}")
        print("--- End Replay Buffer Check ---")

    def push(self, flat_state: np.ndarray, action: int, reward: float, next_flat_state: np.ndarray, done: bool):
        if flat_state.shape[0] != (self.flat_grid_dim + self.num_scalar_metrics) or \
           next_flat_state.shape[0] != (self.flat_grid_dim + self.num_scalar_metrics):
            raise ValueError(f"State dim mismatch in push")
        current_write_idx = self.pos

        grid_part_flat_np = flat_state[:self.flat_grid_dim].reshape(1, self.H, self.W).astype(np.uint8)
        metrics_part_scalar_np = flat_state[self.flat_grid_dim:].astype(np.float32)
        next_grid_part_flat_np = next_flat_state[:self.flat_grid_dim].reshape(1, self.H, self.W).astype(np.uint8)
        next_metrics_part_scalar_np = next_flat_state[self.flat_grid_dim:].astype(np.float32)
        action_np_scalar = self.action_dtype_np(action) # Scalar numpy value with correct dtype
        reward_np_scalar = np.float32(reward)
        done_np_scalar = np.uint8(1 if done else 0)

        with torch.no_grad():
            self.grid_states_ram[current_write_idx].copy_(torch.from_numpy(grid_part_flat_np))
            self.metric_states_ram[current_write_idx].copy_(torch.from_numpy(metrics_part_scalar_np))
            self.next_grid_states_ram[current_write_idx].copy_(torch.from_numpy(next_grid_part_flat_np))
            self.next_metric_states_ram[current_write_idx].copy_(torch.from_numpy(next_metrics_part_scalar_np))
            # Convert scalar numpy to tensor for assignment
            self.actions_ram[current_write_idx] = torch.tensor(action_np_scalar.item(), dtype=self.action_dtype_torch) # .item() for scalar
            self.rewards_ram[current_write_idx] = torch.tensor(reward_np_scalar.item(), dtype=torch.float32)
            self.dones_ram[current_write_idx] = torch.tensor(done_np_scalar.item(), dtype=torch.uint8)

        self.grid_states_mmap[current_write_idx] = grid_part_flat_np
        self.metric_states_mmap[current_write_idx] = metrics_part_scalar_np
        self.next_grid_states_mmap[current_write_idx] = next_grid_part_flat_np
        self.next_metric_states_mmap[current_write_idx] = next_metrics_part_scalar_np
        self.actions_mmap[current_write_idx] = action_np_scalar
        self.rewards_mmap[current_write_idx] = reward_np_scalar
        self.dones_mmap[current_write_idx] = done_np_scalar

        self.pos = (current_write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if not self.debug_triggered_on_size and self.size == self.debug_trigger_size:
            self._print_buffer_debug_info(current_write_idx, f"Size Trigger ({self.debug_trigger_size})")
            self.debug_triggered_on_size = True
        if not self.debug_triggered_on_pos_cycle and \
           current_write_idx == self.debug_trigger_pos_cycle and \
           self.size == self.capacity:
            self._print_buffer_debug_info(current_write_idx, f"Position Cycle Trigger (pos={self.debug_trigger_pos_cycle} overwritten)")
            self.debug_triggered_on_pos_cycle = True

    def sample(self, batch_size: int):
        if self.size == 0: raise RuntimeError("Cannot sample from empty buffer.")
        effective_batch_size = min(batch_size, self.size)
        idx_random = torch.randint(0, self.size, (effective_batch_size,), device=self.ram_tensor_device)

        grid_s_tensor = self.grid_states_ram[idx_random].to(dtype=torch.float32)
        metric_s_tensor = self.metric_states_ram[idx_random]
        next_grid_s_tensor = self.next_grid_states_ram[idx_random].to(dtype=torch.float32)
        next_metric_s_tensor = self.next_metric_states_ram[idx_random]
        a_tensor = self.actions_ram[idx_random].to(dtype=torch.long).unsqueeze(1)
        r_tensor = self.rewards_ram[idx_random].unsqueeze(1)
        d_tensor = self.dones_ram[idx_random].to(dtype=torch.float32).unsqueeze(1)

        def construct_fcn_state(grid_tensor_bchw, metrics_tensor_b_numscalar):
            current_batch_s_val = grid_tensor_bchw.shape[0]
            broadcasted_metric_channels = []
            for metric_idx in self.METRIC_INDICES_FOR_FCN_CHANNELS:
                if metric_idx >= metrics_tensor_b_numscalar.shape[1]:
                    raise IndexError(f"FCN metric index {metric_idx} OOB for stored scalar metrics (count: {metrics_tensor_b_numscalar.shape[1]}).")
                scalar_values = metrics_tensor_b_numscalar[:, metric_idx].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                broadcasted_metric_channels.append(scalar_values.expand(current_batch_s_val, 1, self.H, self.W))
            return torch.cat([grid_tensor_bchw] + broadcasted_metric_channels, dim=1)

        s_fcn = construct_fcn_state(grid_s_tensor, metric_s_tensor)
        ns_fcn = construct_fcn_state(next_grid_s_tensor, next_metric_s_tensor)

        non_blocking_transfer = self.pin_ram_tensors and self.training_device.type == 'cuda'
        s_fcn = s_fcn.to(self.training_device, non_blocking=non_blocking_transfer)
        ns_fcn = ns_fcn.to(self.training_device, non_blocking=non_blocking_transfer)
        a_tensor = a_tensor.to(self.training_device, non_blocking=non_blocking_transfer)
        r_tensor = r_tensor.to(self.training_device, non_blocking=non_blocking_transfer)
        d_tensor = d_tensor.to(self.training_device, non_blocking=non_blocking_transfer)
        return s_fcn, a_tensor, r_tensor, ns_fcn, d_tensor

    def __len__(self): return self.size

    def flush(self):
        # print("[ReplayBuffer] Flushing memmap files to disk...") # Can be verbose
        memmap_attrs_instances = [
            self.grid_states_mmap, self.metric_states_mmap,
            self.next_grid_states_mmap, self.next_metric_states_mmap,
            self.actions_mmap, self.rewards_mmap, self.dones_mmap
        ]
        for m_obj in memmap_attrs_instances:
            if m_obj is not None and hasattr(m_obj, 'flush') and callable(m_obj.flush):
                try: m_obj.flush()
                except ValueError as e: print(f"[ReplayBuffer] Warning: Error flushing a memmap object: {e}")
        # print("[ReplayBuffer] Memmap flush complete.")


    def close(self):
        self.flush()
        self.save_state_to_file()
        memmap_attr_names = [
            "grid_states_mmap", "metric_states_mmap",
            "next_grid_states_mmap", "next_metric_states_mmap",
            "actions_mmap", "rewards_mmap", "dones_mmap"
        ]
        for attr_name in memmap_attr_names:
            if hasattr(self, attr_name):
                try:
                    memmap_obj = getattr(self, attr_name)
                    if isinstance(memmap_obj, np.memmap):
                        setattr(self, attr_name, None) # Remove reference
                    # Python's garbage collector should handle the rest if no other refs
                except AttributeError: pass
        # print("[ReplayBuffer] Closed (memmap handles marked for release, state saved).")


    def _load_state_from_file(self):
        try:
            if self.state_file_path.exists():
                with open(self.state_file_path, 'r') as f: state_data = json.load(f)
                loaded_pos = int(state_data.get('pos', 0)); loaded_size = int(state_data.get('size', 0))
                # Capacity is not known yet if this is called before self.capacity is set.
                # However, __init__ sets self.capacity first, then calls this.
                if 0 <= loaded_pos < self.capacity and 0 <= loaded_size <= self.capacity:
                    self.pos, self.size = loaded_pos, loaded_size
                    # print(f"[ReplayBuffer] Loaded state from JSON: pos={self.pos}, size={self.size}")
                else:
                    print(f"[ReplayBuffer] Invalid pos/size in JSON ({loaded_pos}/{loaded_size} for cap {self.capacity}). Resetting.")
                    self._delete_state_file(); self.pos = 0; self.size = 0
            else:
                # print(f"[ReplayBuffer] No state file found at '{self.state_file_path}'. Starting fresh.")
                self.pos = 0; self.size = 0
        except (json.JSONDecodeError, Exception) as e:
            print(f"[ReplayBuffer] Error loading state from JSON: {e}. Resetting state.")
            self._delete_state_file(); self.pos = 0; self.size = 0


    def save_state_to_file(self):
        state_data = {'pos': self.pos, 'size': self.size}
        temp_filename = self.state_file_path.with_suffix(".tmp")
        try:
            self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_filename, 'w') as f: json.dump(state_data, f)
            os.replace(temp_filename, self.state_file_path)
        except Exception as e:
            print(f"[ReplayBuffer] Error saving state to file: {e}")
            if temp_filename.exists():
                try: os.remove(temp_filename)
                except OSError: pass

    def _delete_state_file(self):
        try:
            if self.state_file_path.exists(): os.remove(self.state_file_path)
        except OSError as e: print(f"[ReplayBuffer] Error deleting state file: {e}")


    def get_buffer_state_for_checkpoint(self) -> dict:
        return {'capacity': self.capacity, 'pos': self.pos, 'size': self.size, 'H': self.H, 'W': self.W,
                'num_scalar_metrics': self.num_scalar_metrics,
                'fcn_output_channels_from_buffer': self.fcn_output_channels_from_buffer,
                'action_dtype_str': np.dtype(self.action_dtype_np).name,
                'directory': str(self.directory)}

    def load_buffer_state_from_checkpoint(self, buffer_state_chkpt: dict):
        # print(f"[ReplayBuffer] Loading buffer state from AGENT checkpoint: pos={buffer_state_chkpt['pos']}, size={buffer_state_chkpt['size']}")
        if self.capacity != buffer_state_chkpt.get('capacity'): raise ValueError("Capacity mismatch on chkpt load")
        if self.H != buffer_state_chkpt.get('H') or self.W != buffer_state_chkpt.get('W'): raise ValueError("Grid H/W mismatch on chkpt load")
        if self.num_scalar_metrics != buffer_state_chkpt.get('num_scalar_metrics'): raise ValueError("num_scalar_metrics mismatch on chkpt load")
        if self.fcn_output_channels_from_buffer != buffer_state_chkpt.get('fcn_output_channels_from_buffer'):
            raise ValueError("fcn_output_channels mismatch on chkpt load")

        size_before_chkpt_load = self.size
        self.pos = buffer_state_chkpt['pos']
        self.size = buffer_state_chkpt['size']

        # print(f"[ReplayBuffer] State updated by AGENT checkpoint. Prev size (from file/pushes): {size_before_chkpt_load}, New pos: {self.pos}, New size: {self.size}")

        if self.size > size_before_chkpt_load:
            # print(f"[ReplayBuffer] Agent chkpt size ({self.size}) > RAM content ({size_before_chkpt_load}). Syncing...")
            self._load_slice_from_mmap_to_ram(size_before_chkpt_load, self.size)
        # elif self.size < size_before_chkpt_load:
            # print(f"[ReplayBuffer] Warning: Agent chkpt size ({self.size}) < RAM content ({size_before_chkpt_load}). RAM beyond new size stale.")
        
        # print(f"[ReplayBuffer] Buffer state from AGENT checkpoint applied. Effective Pos={self.pos}, Size={self.size}")