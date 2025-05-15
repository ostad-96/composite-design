# project_root/optimize.py

from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8") # Or "1" for FEniCS safety
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "8") # Or "1"
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "8") # Or "1"

import argparse
import optuna
import torch
import numpy as np
import json
import re
import glob
import time
import traceback
import shutil

from utils.trainer import Trainer
from utils.dqn import DQNAgent, CompositeDesignEnv

# Import from the reorganized config.py
from config import (
    # Common
    MATRIX_SIZE, FCN_INPUT_CHANNELS,
    # Optuna defaults (for HPs that might not be suggested every time or for fixed HPs)
    OPTUNA_DEFAULT_REPLAY_BUFFER_CAPACITY, OPTUNA_DEFAULT_LOSS_FUNCTION_NAME,
    # Dilation strategy definitions
    FCN_DILATION_STRATEGIES, FCN_DILATION_PROGRESSIVE_MAX_BLOCKS,
    FCN_DILATION_PROGRESSIVE_PATTERN, FCN_DILATION_CYCLIC_PATTERN,
    # Optuna study settings (paths, total episodes target)
    OPTUNA_CHECKPOINT_DIR_BASE as DEFAULT_OPTUNA_CHECKPOINT_DIR_BASE,
    OPTUNA_REPLAY_BUFFER_DIR_BASE as DEFAULT_OPTUNA_REPLAY_BUFFER_DIR_BASE,
    OPTUNA_PLOT_DIR_BASE as DEFAULT_OPTUNA_PLOT_DIR_BASE,
    OPTUNA_RESULTS_FILE_BASE as DEFAULT_OPTUNA_RESULTS_FILE_BASE,
    OPTUNA_TOTAL_TARGET_EPISODES as DEFAULT_OPTUNA_TOTAL_TARGET_EPISODES,
    # Default training loop parameters for Optuna trials (if not overridden by suggestions)
    OPTUNA_DEFAULT_EPISODES_PER_CYCLE, OPTUNA_DEFAULT_OPT_STEPS_PER_CYCLE
)


# --- Global Paths (will be set by args, defaulting to config values) ---
# These need to be global so 'objective' can access them if not using args (e.g. programmatic call)
# However, best practice is to pass them via args to 'objective' or make them part of Trial's user_attrs.
# For now, we'll keep the global override pattern from your original script.
CHECKPOINT_DIR_BASE_ACTUAL = DEFAULT_OPTUNA_CHECKPOINT_DIR_BASE
REPLAY_BUFFER_DIR_BASE_ACTUAL = DEFAULT_OPTUNA_REPLAY_BUFFER_DIR_BASE
PLOT_DIR_BASE_ACTUAL = DEFAULT_OPTUNA_PLOT_DIR_BASE
RESULTS_FILE_BASE_ACTUAL = DEFAULT_OPTUNA_RESULTS_FILE_BASE
TOTAL_TARGET_EPISODES_ACTUAL = DEFAULT_OPTUNA_TOTAL_TARGET_EPISODES


torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1"))) # Often 1 is safer with FEniCS


def save_results_incrementally(result_item: dict, filename: str):
    try:
        with open(filename, 'a') as f:
            json.dump(result_item, f, default=str); f.write('\n')
    except Exception as e:
        print(f"\n--- WARNING: Failed to save incremental result: {e} ---")

def load_previous_results(filename: str) -> list[dict]:
    results = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                for l_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError as je:
                            print(f" Warn: JSON decode error in {filename} L{l_num}: {je}.")
        except Exception as e:
            print(f" Warn: Error loading results from {filename}: {e}.")
    return results

def find_resumable_checkpoint(checkpoint_dir: str, exclude_trial_num: int | None = None) -> str | None:
    potential_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_trial_*.pth"))
    if not potential_files: return None
    resumable_candidates = []
    for f_path in potential_files:
        if not os.path.isfile(f_path) or f_path.endswith(".tmp"): continue
        match = re.search(r"checkpoint_trial_(\d+)\.pth$", os.path.basename(f_path))
        if match:
            try:
                trial_num = int(match.group(1))
                if exclude_trial_num is not None and trial_num == exclude_trial_num: continue
                if os.path.getsize(f_path) > 200: # Basic size check
                    resumable_candidates.append((trial_num, f_path))
            except (ValueError, OSError): pass
    if not resumable_candidates: return None
    resumable_candidates.sort(key=lambda x: x[0]) # Prioritize lower trial numbers
    return resumable_candidates[0][1]


def generate_dilation_factors(strategy_name: str, num_blocks: int) -> list[int]:
    if strategy_name == "all_ones":
        return [1] * num_blocks
    elif strategy_name == "progressive_trim":
        # Ensure pattern is long enough, or repeat/truncate
        if num_blocks <= len(FCN_DILATION_PROGRESSIVE_PATTERN):
            return FCN_DILATION_PROGRESSIVE_PATTERN[:num_blocks]
        else: # Repeat the pattern if num_blocks is larger
            return (FCN_DILATION_PROGRESSIVE_PATTERN * (num_blocks // len(FCN_DILATION_PROGRESSIVE_PATTERN) + 1))[:num_blocks]
    elif strategy_name == "cyclic_124":
        return [FCN_DILATION_CYCLIC_PATTERN[i % len(FCN_DILATION_CYCLIC_PATTERN)] for i in range(num_blocks)]
    else:
        print(f"Warning: Unknown dilation strategy '{strategy_name}'. Defaulting to 'all_ones'."); return [1] * num_blocks


def objective(trial: optuna.Trial) -> float:
    # Use the _ACTUAL global paths which are set by main's argparse
    global CHECKPOINT_DIR_BASE_ACTUAL, REPLAY_BUFFER_DIR_BASE_ACTUAL, PLOT_DIR_BASE_ACTUAL, TOTAL_TARGET_EPISODES_ACTUAL

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    trial_number = trial.number
    print(f"\n--- Starting Optuna Trial {trial_number} | Device: {device} ---")

    start_cycle = 0; total_episodes_offset = 0; total_steps_offset = 0
    initial_rewards = []; initial_success_flags = []
    hyperparams_to_use = {}; loaded_agent_checkpoint_state = None
    loaded_trainer_loop_config = {}; is_resuming = False; resumed_from_path = None

    # Use _ACTUAL paths for this trial's checkpoint management
    current_trial_checkpoint_path = os.path.join(CHECKPOINT_DIR_BASE_ACTUAL, f"checkpoint_trial_{trial_number}.pth")

    if os.path.exists(current_trial_checkpoint_path):
        resumable_checkpoint_path = current_trial_checkpoint_path
    else:
        resumable_checkpoint_path = find_resumable_checkpoint(CHECKPOINT_DIR_BASE_ACTUAL, exclude_trial_num=trial_number)

    if resumable_checkpoint_path:
        print(f"Attempting to load state from: {resumable_checkpoint_path}")
        loaded_checkpoint_content = Trainer.load_checkpoint_data(resumable_checkpoint_path)
        if loaded_checkpoint_content:
            trainer_chkpt_state = loaded_checkpoint_content.get('trainer_checkpoint_state', {})
            loaded_agent_checkpoint_state = loaded_checkpoint_content.get('agent_checkpoint_state', None)

            # Check for necessary fields for resumption
            if (loaded_agent_checkpoint_state and
                'agent_hyperparameters' in loaded_agent_checkpoint_state and # Critical for DQNAgent.load_state
                'hyperparams_of_this_run' in trainer_chkpt_state and
                'episode_success_flags' in trainer_chkpt_state):
                is_resuming = True; resumed_from_path = resumable_checkpoint_path
                hyperparams_to_use = trainer_chkpt_state['hyperparams_of_this_run']
                start_cycle = trainer_chkpt_state.get('current_cycle', -1) + 1
                total_episodes_offset = trainer_chkpt_state.get('total_episodes_run', 0)
                total_steps_offset = trainer_chkpt_state.get('total_steps_run', 0)
                initial_rewards = trainer_chkpt_state.get('episode_rewards', [])
                initial_success_flags = trainer_chkpt_state.get('episode_success_flags', [])
                loaded_trainer_loop_config = trainer_chkpt_state.get('trainer_loop_config', {})
                resumed_successful_episodes_count = sum(initial_success_flags)
                print(f" Resuming T{trial_number} from end of cycle: {start_cycle - 1}. Prev Eps: {total_episodes_offset}. Prev Successes: {resumed_successful_episodes_count}")

                if resumable_checkpoint_path != current_trial_checkpoint_path:
                    try: # Attempt to move the checkpoint to the current trial's name
                        shutil.move(resumable_checkpoint_path, current_trial_checkpoint_path) # shutil.move
                        print(f" Moved checkpoint {os.path.basename(resumable_checkpoint_path)} to {os.path.basename(current_trial_checkpoint_path)}")
                        resumed_from_path = current_trial_checkpoint_path
                    except OSError as e:
                        print(f"ERROR moving checkpoint: {e}. Aborting T{trial_number} resumption from this file.")
                        is_resuming = False # Reset flags
                        start_cycle = 0; total_episodes_offset = 0; total_steps_offset = 0; initial_rewards = []; initial_success_flags = []
                        hyperparams_to_use = {}; loaded_agent_checkpoint_state = None; loaded_trainer_loop_config = {}
            else:
                print(f"ERROR: Checkpoint {resumable_checkpoint_path} incomplete or missing critical fields for resumption. Cannot resume.")
                if os.path.exists(resumable_checkpoint_path):
                     print(f" Deleting potentially corrupted/incomplete checkpoint: {resumable_checkpoint_path}")
                     try: os.remove(resumable_checkpoint_path)
                     except OSError as e: print(f" Error deleting corrupted chkpt: {e}")
    
    if is_resuming and hyperparams_to_use:
        print(f"Trial {trial_number} resuming with HPs loaded from checkpoint:")
        # Print only a subset or summary if too verbose
        # print(json.dumps(hyperparams_to_use, indent=2, default=repr))
        for key, value in hyperparams_to_use.items():
            try: trial.set_user_attr(f"loaded_{key}", value) # Log that HPs were loaded
            except Exception: pass # Optuna might complain if attr already exists
        # Trainer loop params from checkpoint
        eps_per_cycle = loaded_trainer_loop_config.get('episodes_per_cycle', OPTUNA_DEFAULT_EPISODES_PER_CYCLE)
        opt_per_cycle = loaded_trainer_loop_config.get('opt_steps_per_cycle', OPTUNA_DEFAULT_OPT_STEPS_PER_CYCLE)
        num_cycles_target = loaded_trainer_loop_config.get('num_cycles_target')
        if num_cycles_target is None: # Fallback if not in checkpoint
            num_cycles_target = int(np.ceil(TOTAL_TARGET_EPISODES_ACTUAL / eps_per_cycle)) if eps_per_cycle > 0 else 500 # Default num cycles
    else:
        is_resuming = False # Ensure this is false if not properly resumed
        print(f"Trial {trial_number} starting with newly suggested HPs.")
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.97, 0.999)
        epsilon_decay = trial.suggest_float("epsilon_decay", 0.9995, 0.99999)
        epsilon_min = trial.suggest_float("epsilon_min", 0.01, 0.05)
        tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        eps_per_cycle = trial.suggest_int("eps_per_cycle", 50, 200, step=50)
        opt_per_cycle = trial.suggest_int("opt_per_cycle", 200, 1000, step=100)
        use_lr_scheduler = trial.suggest_categorical("use_lr_scheduler", [True, False])
        lr_end_factor = trial.suggest_float("lr_end_factor", 0.01, 0.2) # if use_lr_scheduler
        lr_decay_cycle_fraction = trial.suggest_float("lr_decay_cycle_fraction", 0.6, 1.0) # if use_lr_scheduler
        
        fcn_num_filters = trial.suggest_int("fcn_num_filters", 4, 32, step=4)
        fcn_blocks = trial.suggest_int("fcn_blocks", 3, 8)
        fcn_kernel_size = trial.suggest_categorical("fcn_kernel_size", [3, 5])
        dilation_strategy_name = trial.suggest_categorical("dilation_strategy", FCN_DILATION_STRATEGIES)
        clip_grad_norm_max = trial.suggest_float("clip_grad_norm_max", 0.5, 5.0, log=True)
        # Allow suggesting loss function if desired, e.g., ["smooth_l1", "mse"]
        loss_function = trial.suggest_categorical("loss_function", [OPTUNA_DEFAULT_LOSS_FUNCTION_NAME])

        # Construct derived HPs
        num_cycles_target = int(np.ceil(TOTAL_TARGET_EPISODES_ACTUAL / eps_per_cycle)) if eps_per_cycle > 0 else 500 # Default cycles
        lr_decay_cycles_calculated = int(np.ceil(num_cycles_target * lr_decay_cycle_fraction)) if use_lr_scheduler else 0
        fcn_dilation_factors_list = generate_dilation_factors(dilation_strategy_name, fcn_blocks)

        hyperparams_to_use = {
            "lr": lr, "gamma": gamma, "epsilon_decay": epsilon_decay, "epsilon_min": epsilon_min,
            "tau": tau, "batch_size": batch_size, "eps_per_cycle": eps_per_cycle, "opt_per_cycle": opt_per_cycle,
            "use_lr_scheduler": use_lr_scheduler, "lr_end_factor": lr_end_factor,
            "lr_decay_cycle_fraction": lr_decay_cycle_fraction, # Log the fraction
            "fcn_input_channels": FCN_INPUT_CHANNELS, # Fixed from common config
            "fcn_num_filters": fcn_num_filters, "fcn_blocks": fcn_blocks, "fcn_kernel_size": fcn_kernel_size,
            "dilation_strategy": dilation_strategy_name, # Log the strategy name
            "clip_grad_norm_max": clip_grad_norm_max, "loss_function": loss_function,
            # Derived and fixed HPs for this trial run
            "num_cycles_target": num_cycles_target,
            "lr_decay_cycles_calculated": lr_decay_cycles_calculated,
            "buffer_capacity": OPTUNA_DEFAULT_REPLAY_BUFFER_CAPACITY, # Or suggest this too
            "fcn_dilation_factors": fcn_dilation_factors_list # The actual list used
        }

    print("--- Hyperparameters For Trial ---")
    print(json.dumps(hyperparams_to_use, indent=2, default=repr)) # default=repr for non-serializable like lists
    print("---------------------------------")

    env = CompositeDesignEnv() # One env per trial
    try:
        agent = DQNAgent(
            device=device,
            matrix_size=MATRIX_SIZE, # From common config
            fcn_input_channels=hyperparams_to_use['fcn_input_channels'],
            num_scalar_metrics_env=env.num_scalar_metrics,
            lr=hyperparams_to_use['lr'],
            gamma=hyperparams_to_use['gamma'],
            batch_size=hyperparams_to_use['batch_size'],
            # Use current epsilon from checkpoint if resuming, else fresh start
            epsilon_start=loaded_agent_checkpoint_state['epsilon'] if is_resuming and loaded_agent_checkpoint_state and 'epsilon' in loaded_agent_checkpoint_state else 1.0,
            epsilon_decay=hyperparams_to_use['epsilon_decay'],
            epsilon_min=hyperparams_to_use['epsilon_min'],
            tau=hyperparams_to_use['tau'],
            clip_grad_norm_max=hyperparams_to_use['clip_grad_norm_max'],
            loss_function_name=hyperparams_to_use['loss_function'],
            fcn_num_filters=hyperparams_to_use['fcn_num_filters'],
            fcn_blocks=hyperparams_to_use['fcn_blocks'],
            fcn_kernel_size=hyperparams_to_use['fcn_kernel_size'],
            fcn_dilation_factors=hyperparams_to_use['fcn_dilation_factors'],
            use_lr_scheduler=hyperparams_to_use['use_lr_scheduler'],
            lr_end_factor=hyperparams_to_use['lr_end_factor'],
            lr_decay_cycles=hyperparams_to_use['lr_decay_cycles_calculated'],
            buffer_capacity=hyperparams_to_use['buffer_capacity'],
            buffer_dir=REPLAY_BUFFER_DIR_BASE_ACTUAL, # Base dir for all Optuna buffers
        )
    except Exception as e:
        print(f"ERROR: DQNAgent init failed for T{trial_number}: {e}"); traceback.print_exc()
        raise optuna.TrialPruned(f"Agent init error: {e}")

    if is_resuming and loaded_agent_checkpoint_state:
        print(f"Loading agent state dict for Trial {trial_number} from {resumed_from_path}...")
        try:
            agent.load_state(loaded_agent_checkpoint_state)
        except Exception as e:
            print(f"ERROR loading agent state for T{trial_number}: {e}"); traceback.print_exc()
            raise optuna.TrialPruned(f"Agent state load error: {e}")

    trainer = Trainer(
        env=env, agent=agent, device=device, trial_number=trial_number,
        episodes_per_cycle=eps_per_cycle, # Use the resolved value for this trial
        opt_steps_per_cycle=opt_per_cycle, # Use the resolved value for this trial
        num_cycles=num_cycles_target,      # Use the resolved value for this trial
        checkpoint_dir=CHECKPOINT_DIR_BASE_ACTUAL, # Base dir for all Optuna checkpoints
        start_cycle=start_cycle,
        total_episodes_offset=total_episodes_offset,
        total_steps_offset=total_steps_offset,
        initial_rewards=initial_rewards,
        initial_success_flags=initial_success_flags,
        hyperparams=hyperparams_to_use # Log all HPs used for this trial run
    )

    final_score = -float('inf'); training_status = "UNKNOWN_INIT_FAIL"; final_window_success_rate = 0.0
    try:
        final_score = trainer.train() # Trainer handles internal checkpointing
        training_status = "COMPLETE"
        final_window_success_rate = trainer.final_window_success_rate
        print(f"Trial {trial_number} COMPLETED. Score: {final_score:.4f}, Win Success: {final_window_success_rate:.2f}%")

        # On successful completion, delete the specific checkpoint for this trial
        # as Optuna has logged the result.
        if os.path.exists(current_trial_checkpoint_path):
            try:
                os.remove(current_trial_checkpoint_path)
                print(f" Deleted successful trial checkpoint: {current_trial_checkpoint_path}")
            except OSError as e:
                print(f" Warning: Failed to delete successful T{trial_number} chkpt: {e}")

        plot_path = os.path.join(PLOT_DIR_BASE_ACTUAL, f"trial_{trial_number}_rewards_COMPLETE.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        trainer.plot_rewards(window=100, save_path=plot_path)
        return final_score # Return score for Optuna

    except KeyboardInterrupt:
        training_status = "INTERRUPTED_USER"
        print(f"\nTrial {trial_number} INTERRUPTED by user.")
        # Checkpoint is saved by trainer at the end of each cycle.
        # If current_trial_checkpoint_path exists, it's the latest.
        if os.path.exists(current_trial_checkpoint_path):
            print(f" Checkpoint for interrupted T{trial_number} remains: {current_trial_checkpoint_path}")
        raise optuna.TrialPruned("Trial interrupted by user, checkpoint may be preserved.")

    except optuna.TrialPruned as e_prune: # Catch Optuna's pruning signal
        training_status = "PRUNED_OPTUNA"
        print(f"Trial {trial_number} PRUNED by Optuna: {e_prune}")
        plot_path = os.path.join(PLOT_DIR_BASE_ACTUAL, f"trial_{trial_number}_rewards_PRUNED.png")
        if 'trainer' in locals() and hasattr(trainer, 'episode_rewards') and trainer.episode_rewards:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            trainer.plot_rewards(window=100, save_path=plot_path)
        raise e_prune # Re-raise to inform Optuna

    except Exception as e_train: # Catch other runtime errors during training
        training_status = "FAILED_RUNTIME"
        print(f"Trial {trial_number} FAILED with runtime error: {e_train}")
        traceback.print_exc()
        plot_path = os.path.join(PLOT_DIR_BASE_ACTUAL, f"trial_{trial_number}_rewards_FAILED.png")
        if 'trainer' in locals() and hasattr(trainer, 'episode_rewards') and trainer.episode_rewards:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            trainer.plot_rewards(window=100, save_path=plot_path)
        # For Optuna, a runtime error should also be treated as something that prunes the trial or marks it as failed.
        # Raising optuna.TrialPruned here is one way to signal Optuna.
        raise optuna.TrialPruned(f"Runtime error in T{trial_number}: {e_train}")

    finally:
        print(f"Executing 'finally' block for Trial {trial_number}. Status: {training_status}")
        if 'agent' in locals() and agent: agent.close_buffer()
        if 'env' in locals() and env: env.close()

        if 'trainer' in locals() and hasattr(trainer, 'final_window_success_rate'):
            final_window_success_rate = trainer.final_window_success_rate # Get latest value

        current_result_summary = {
            "trial_number": trial_number,
            "status": training_status,
            "final_score_reported_to_optuna": final_score if training_status == "COMPLETE" else None,
            "final_window_success_rate_percent": round(final_window_success_rate, 2),
            "hyperparameters_used": hyperparams_to_use,
            "last_cycle_completed": trainer.current_cycle if 'trainer' in locals() else start_cycle -1,
            "total_episodes_run_in_trial": (trainer.total_episodes_run - total_episodes_offset) if 'trainer' in locals() else 0,
            "was_resumed_at_start": is_resuming,
            "resumed_from_file_path": os.path.basename(resumed_from_path) if resumed_from_path else None,
            "timestamp_end": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_results_incrementally(current_result_summary, RESULTS_FILE_BASE_ACTUAL)
        print(f"--- Finished 'finally' for Trial {trial_number} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna FCN DQN Optimization")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials to run in this session.")
    parser.add_argument("--study-name", type=str, default="DQN_FCN_Optuna_Study", help="Name for the Optuna study.")
    parser.add_argument("--storage", type=str, default="sqlite:///dqn_fcn_optuna_study.db", help="Optuna storage URL.")
    # Allow overriding default paths from config
    parser.add_argument("--results-file", type=str, default=None, help=f"Path to JSONL results file (default from config: {DEFAULT_OPTUNA_RESULTS_FILE_BASE}).")
    parser.add_argument("--plot-dir", type=str, default=None, help=f"Dir for trial plots (default from config: {DEFAULT_OPTUNA_PLOT_DIR_BASE}).")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help=f"Dir for checkpoints (default from config: {DEFAULT_OPTUNA_CHECKPOINT_DIR_BASE}).")
    parser.add_argument("--buffer-dir", type=str, default=None, help=f"Dir for replay buffer data (default from config: {DEFAULT_OPTUNA_REPLAY_BUFFER_DIR_BASE}).")
    parser.add_argument("--target-episodes", type=int, default=None, help=f"Target episodes per trial (default from config: {DEFAULT_OPTUNA_TOTAL_TARGET_EPISODES}).")
    parser.add_argument("--cleanup-buffer-ask", action="store_true", help="Ask to cleanup Optuna's base buffer dir after study session.")
    args = parser.parse_args()

    # Set the _ACTUAL global paths based on args or defaults from config
    CHECKPOINT_DIR_BASE_ACTUAL = args.checkpoint_dir if args.checkpoint_dir is not None else DEFAULT_OPTUNA_CHECKPOINT_DIR_BASE
    REPLAY_BUFFER_DIR_BASE_ACTUAL = args.buffer_dir if args.buffer_dir is not None else DEFAULT_OPTUNA_REPLAY_BUFFER_DIR_BASE
    PLOT_DIR_BASE_ACTUAL = args.plot_dir if args.plot_dir is not None else DEFAULT_OPTUNA_PLOT_DIR_BASE
    RESULTS_FILE_BASE_ACTUAL = args.results_file if args.results_file is not None else DEFAULT_OPTUNA_RESULTS_FILE_BASE
    TOTAL_TARGET_EPISODES_ACTUAL = args.target_episodes if args.target_episodes is not None else DEFAULT_OPTUNA_TOTAL_TARGET_EPISODES

    os.makedirs(CHECKPOINT_DIR_BASE_ACTUAL, exist_ok=True)
    os.makedirs(REPLAY_BUFFER_DIR_BASE_ACTUAL, exist_ok=True) # Ensure base Optuna buffer dir exists
    os.makedirs(PLOT_DIR_BASE_ACTUAL, exist_ok=True)

    initial_results_count = len(load_previous_results(RESULTS_FILE_BASE_ACTUAL))
    print(f"Found {initial_results_count} previous execution records in '{RESULTS_FILE_BASE_ACTUAL}'.")
    print("\n--- Initializing Optuna Study ---")
    print(f" Study Name: {args.study_name}")
    print(f" Storage: {args.storage}")
    print(f" Target Trials for this run session: {args.trials}")
    print(f" Dirs: Chkpts='{CHECKPOINT_DIR_BASE_ACTUAL}', BufferBase='{REPLAY_BUFFER_DIR_BASE_ACTUAL}', Plots='{PLOT_DIR_BASE_ACTUAL}'")
    print(f" Results File: '{RESULTS_FILE_BASE_ACTUAL}'")
    print(f" Target Episodes per Trial: {TOTAL_TARGET_EPISODES_ACTUAL}")
    print("-" * 30)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize" # We want to maximize the score (e.g., average reward)
    )

    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            gc_after_trial=True, # Helps manage memory
            n_jobs=1, # FEniCS is often not thread-safe for multiple parallel FEM solves
            catch=(KeyboardInterrupt, optuna.exceptions.TrialPruned) # Catch pruning explicitly if needed
        )
    except KeyboardInterrupt:
        print("\nOptuna study.optimize() loop was INTERRUPTED by user.")
    except Exception as e: # Catch any other unexpected error during optimize loop
        print(f"\nOptuna study.optimize() loop FAILED with an unexpected error: {e}")
        traceback.print_exc()
    finally:
        print("\n--- Optuna Study Session Concluded ---")
        # ... (rest of your summary printing logic remains the same, using _ACTUAL paths) ...
        print(f"\n--- Summary from Optuna Study Object ({study.study_name}) ---")
        all_trials_from_study = []
        try: all_trials_from_study = study.get_trials(deepcopy=False)
        except Exception as e_study: print(f" Error fetching trials from study object: {e_study}")

        complete_study_trials = [t for t in all_trials_from_study if t.state == optuna.trial.TrialState.COMPLETE]
        # ... (your existing summary code)

        print(f"\n--- Summary from Incremental Results File ('{RESULTS_FILE_BASE_ACTUAL}') ---")
        all_logged_records = load_previous_results(RESULTS_FILE_BASE_ACTUAL)
        # ... (your existing summary code)

        if args.cleanup_buffer_ask:
            print(f"\nOptuna's base replay buffer data is in: {REPLAY_BUFFER_DIR_BASE_ACTUAL}")
            if os.path.exists(REPLAY_BUFFER_DIR_BASE_ACTUAL) and any(os.scandir(REPLAY_BUFFER_DIR_BASE_ACTUAL)):
                confirm = input(f" Delete all contents of Optuna's base replay buffer directory '{REPLAY_BUFFER_DIR_BASE_ACTUAL}'? [y/N]: ").lower()
                if confirm == 'y':
                    try:
                        shutil.rmtree(REPLAY_BUFFER_DIR_BASE_ACTUAL)
                        os.makedirs(REPLAY_BUFFER_DIR_BASE_ACTUAL) # Recreate
                        print(f" Optuna base buffer directory '{REPLAY_BUFFER_DIR_BASE_ACTUAL}' cleared and recreated.")
                    except Exception as e_del: print(f" ERROR deleting Optuna base buffer directory: {e_del}")
                else: print(" Optuna base buffer directory not deleted.")
            else: print(" Optuna base buffer directory is empty or does not exist. No cleanup needed.")
        print("\n--- Script Finished ---")