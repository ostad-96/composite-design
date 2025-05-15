# project_root/main_single_run.py

import os
# Set OMP_NUM_THREADS to 1 if using FEniCS or other libraries sensitive to multi-threading
# For PyTorch on CPU, a higher number might be fine if not conflicting.
# For MPS/CUDA, this is less relevant for PyTorch itself.
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import json
import time
import shutil # For cleaning up buffer if desired

from utils.trainer import Trainer
from utils.dqn import DQNAgent, CompositeDesignEnv # DQNAgent and Env are in utils/dqn.py

# Import parameters for a single run from the reorganized config.py
from config import (
    # Common
    MATRIX_SIZE, FCN_INPUT_CHANNELS,
    # Single Run specific HPs
    SINGLE_RUN_LEARNING_RATE, SINGLE_RUN_TAU, SINGLE_RUN_BATCH_SIZE, SINGLE_RUN_GAMMA,
    SINGLE_RUN_EPSILON_START, SINGLE_RUN_EPSILON_DECAY, SINGLE_RUN_EPSILON_MIN,
    SINGLE_RUN_CLIP_GRAD_NORM_MAX, SINGLE_RUN_LOSS_FUNCTION_NAME,
    SINGLE_RUN_USE_LR_SCHEDULER, SINGLE_RUN_LR_END_FACTOR, SINGLE_RUN_LR_DECAY_CYCLES_CALCULATED,
    SINGLE_RUN_FCN_NUM_FILTERS_RESBLOCK, SINGLE_RUN_FCN_NUM_RES_BLOCKS, SINGLE_RUN_FCN_KERNEL_SIZE,
    SINGLE_RUN_FCN_DILATION_FACTORS_LIST, # Using the direct list
    SINGLE_RUN_NUM_CYCLES, SINGLE_RUN_EPISODES_PER_CYCLE, SINGLE_RUN_OPT_STEPS_PER_CYCLE,
    SINGLE_RUN_REPLAY_BUFFER_CAPACITY,
    # Single Run Paths and ID Tag
    SINGLE_RUN_ID_TAG,
    SINGLE_RUN_CHECKPOINT_DIR,
    SINGLE_RUN_PLOT_DIR,
    SINGLE_RUN_REPLAY_BUFFER_DIR
)

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

def run_single_trial():
    """
    Runs a single training trial with predefined hyperparameters loaded from config.py.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    run_name = f"single_run_{SINGLE_RUN_ID_TAG}" # Use the ID tag from config for naming

    print(f"\n--- Starting Single Run: {run_name} | Device: {device} ---")

    # Use directories directly from config for this specific run
    # These are already fully qualified paths like "checkpoints_single_run_trial57_best"
    current_checkpoint_dir = SINGLE_RUN_CHECKPOINT_DIR
    current_plot_dir = SINGLE_RUN_PLOT_DIR
    current_buffer_dir = SINGLE_RUN_REPLAY_BUFFER_DIR

    os.makedirs(current_checkpoint_dir, exist_ok=True)
    os.makedirs(current_plot_dir, exist_ok=True)
    os.makedirs(current_buffer_dir, exist_ok=True)

    # --- Hyperparameters for this run (loaded from config.py) ---
    # These are the specific values for the single run
    hps_for_agent = {
        "lr": SINGLE_RUN_LEARNING_RATE,
        "gamma": SINGLE_RUN_GAMMA,
        "epsilon_start": SINGLE_RUN_EPSILON_START, # Explicitly use the single run start
        "epsilon_decay": SINGLE_RUN_EPSILON_DECAY,
        "epsilon_min": SINGLE_RUN_EPSILON_MIN,
        "tau": SINGLE_RUN_TAU,
        "batch_size": SINGLE_RUN_BATCH_SIZE,
        "use_lr_scheduler": SINGLE_RUN_USE_LR_SCHEDULER,
        "lr_end_factor": SINGLE_RUN_LR_END_FACTOR,
        "fcn_input_channels": FCN_INPUT_CHANNELS, # Common config
        "fcn_num_filters": SINGLE_RUN_FCN_NUM_FILTERS_RESBLOCK,
        "fcn_blocks": SINGLE_RUN_FCN_NUM_RES_BLOCKS,
        "fcn_kernel_size": SINGLE_RUN_FCN_KERNEL_SIZE,
        "clip_grad_norm_max": SINGLE_RUN_CLIP_GRAD_NORM_MAX,
        "loss_function_name": SINGLE_RUN_LOSS_FUNCTION_NAME,
        "lr_decay_cycles": SINGLE_RUN_LR_DECAY_CYCLES_CALCULATED,
        "buffer_capacity": SINGLE_RUN_REPLAY_BUFFER_CAPACITY,
        "fcn_dilation_factors": SINGLE_RUN_FCN_DILATION_FACTORS_LIST
    }
    print("--- Hyperparameters for this Run (from config.py) ---")
    print(json.dumps(hps_for_agent, indent=2, default=str))
    print("-----------------------------------------------------")

    env = CompositeDesignEnv() # Initialize environment
    agent = DQNAgent(
        device=device,
        matrix_size=MATRIX_SIZE, # Common config
        fcn_input_channels=hps_for_agent['fcn_input_channels'],
        num_scalar_metrics_env=env.num_scalar_metrics, # From env instance
        lr=hps_for_agent['lr'],
        gamma=hps_for_agent['gamma'],
        batch_size=hps_for_agent['batch_size'],
        epsilon_start=hps_for_agent['epsilon_start'], # Fresh run epsilon start
        epsilon_decay=hps_for_agent['epsilon_decay'],
        epsilon_min=hps_for_agent['epsilon_min'],
        tau=hps_for_agent['tau'],
        clip_grad_norm_max=hps_for_agent['clip_grad_norm_max'],
        loss_function_name=hps_for_agent['loss_function_name'],
        fcn_num_filters=hps_for_agent['fcn_num_filters'],
        fcn_blocks=hps_for_agent['fcn_blocks'],
        fcn_kernel_size=hps_for_agent['fcn_kernel_size'],
        fcn_dilation_factors=hps_for_agent['fcn_dilation_factors'],
        use_lr_scheduler=hps_for_agent['use_lr_scheduler'],
        lr_end_factor=hps_for_agent['lr_end_factor'],
        lr_decay_cycles=hps_for_agent['lr_decay_cycles'],
        buffer_capacity=hps_for_agent['buffer_capacity'],
        buffer_dir=current_buffer_dir, # Use run-specific buffer directory
    )

    trainer = Trainer(
        env=env,
        agent=agent,
        device=device,
        trial_number=SINGLE_RUN_ID_TAG, # Use the ID tag for trainer's internal logging/naming
        # Loop parameters from single run config
        episodes_per_cycle=SINGLE_RUN_EPISODES_PER_CYCLE,
        opt_steps_per_cycle=SINGLE_RUN_OPT_STEPS_PER_CYCLE,
        num_cycles=SINGLE_RUN_NUM_CYCLES,
        # Paths from single run config
        checkpoint_dir=current_checkpoint_dir, # Trainer saves its checkpoint file *inside* this dir
        # Resumption state (all 0 for a fresh single run)
        start_cycle=0,
        total_episodes_offset=0,
        total_steps_offset=0,
        initial_rewards=None,
        initial_success_flags=None,
        # Hyperparams for logging in checkpoint (can be hps_for_agent or a broader dict)
        hyperparams=hps_for_agent
    )

    final_score = -float('inf')
    training_status = "UNKNOWN_INIT_FAIL"
    final_window_success_rate = 0.0
    start_run_time = time.time()

    try:
        final_score = trainer.train() # This runs the loop and saves checkpoints internally
        training_status = "COMPLETE"
        final_window_success_rate = trainer.final_window_success_rate # Get from trainer
        print(f"\nRun {run_name} COMPLETED.")
        print(f"  Final Score (avg last window): {final_score:.4f}")
        print(f"  Final Success Rate (avg last window): {final_window_success_rate:.2f}%")

        plot_path = os.path.join(current_plot_dir, f"{run_name}_rewards_final.png")
        trainer.plot_rewards(window=100, save_path=plot_path)
        print(f"  Reward plot saved to: {plot_path}")

        # The trainer saves checkpoints like "checkpoint_trial_trial57_best.pth"
        # inside current_checkpoint_dir. Decide if you want to keep it.
        final_checkpoint_path = os.path.join(current_checkpoint_dir, f"checkpoint_trial_{SINGLE_RUN_ID_TAG}.pth")
        if os.path.exists(final_checkpoint_path):
            print(f"  Final checkpoint for successful run kept at: {final_checkpoint_path}")
        else:
            print(f"  Note: Final checkpoint file not found at expected path: {final_checkpoint_path}")


    except KeyboardInterrupt:
        training_status = "INTERRUPTED_USER"
        print(f"\nRun {run_name} INTERRUPTED by user.")
        # Checkpoint would have been saved by trainer after the last completed cycle
    except Exception as e_train:
        training_status = "FAILED_RUNTIME"
        print(f"\nRun {run_name} FAILED with runtime error: {e_train}")
        import traceback
        traceback.print_exc()
        if hasattr(trainer, 'episode_rewards') and trainer.episode_rewards: # Check if trainer and rewards exist
            plot_path = os.path.join(current_plot_dir, f"{run_name}_rewards_FAILED.png")
            trainer.plot_rewards(window=100, save_path=plot_path)
            print(f"  Reward plot (on fail) saved to: {plot_path}")
    finally:
        end_run_time = time.time()
        print(f"\n--- Run {run_name} Concluded ---")
        print(f"  Status: {training_status}")
        print(f"  Total Runtime: {(end_run_time - start_run_time)/60:.2f} minutes")

        if 'agent' in locals() and agent and hasattr(agent, 'close_buffer'): # Check existence
            agent.close_buffer()
            print("  Replay buffer closed.")
        if 'env' in locals() and env and hasattr(env, 'close'): # Check existence
            env.close()
            print("  Environment closed.")

        # Optional: Cleanup buffer directory for this specific run after completion/failure
        # Set to True if you want to automatically delete the buffer data for this run
        cleanup_buffer_after_run = False
        if cleanup_buffer_after_run and os.path.exists(current_buffer_dir):
            try:
                shutil.rmtree(current_buffer_dir)
                print(f"  Cleaned up buffer directory: {current_buffer_dir}")
            except Exception as e_del:
                print(f"  ERROR deleting buffer directory {current_buffer_dir}: {e_del}")

        print("--- End of Script ---")

if __name__ == "__main__":
    run_single_trial()