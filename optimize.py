# project_root/optimize.py

from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "4")
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "4")
import argparse, optuna, torch, numpy as np, json, re, glob, time, traceback, shutil
from utils.trainer import Trainer
from utils.dqn import DQNAgent, CompositeDesignEnv
from config import (
    MATRIX_SIZE, FCN_INPUT_CHANNELS,
    FCN_NUM_FILTERS_RESBLOCK, FCN_NUM_RES_BLOCKS as DEFAULT_FCN_BLOCKS, FCN_KERNEL_SIZE as DEFAULT_FCN_KERNEL_SIZE,
    FCN_DILATION_STRATEGIES, FCN_DILATION_PROGRESSIVE_MAX_BLOCKS,
    FCN_DILATION_PROGRESSIVE_PATTERN, FCN_DILATION_CYCLIC_PATTERN,
    REPLAY_BUFFER_CAPACITY, CLIP_GRAD_NORM_MAX, # <<< Added import
    EPISODES_PER_CYCLE, OPT_STEPS_PER_CYCLE, NUM_CYCLES, TOTAL_TARGET_EPISODES
)

# --- Global Paths (Set by args later) ---
CHECKPOINT_DIR_BASE = "checkpoints_fcn_dil" # Changed suffix for new runs
REPLAY_BUFFER_DIR_BASE = "replay_buffer_data_fcn_dil" # Changed suffix
PLOT_DIR_BASE = "trial_plots_fcn_dil" # Changed suffix
RESULTS_FILE_BASE = "trial_results_fcn_dil.jsonl" # Changed suffix

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

def save_results_incrementally(result_item: dict, filename: str):
    try:
        with open(filename, 'a') as f: json.dump(result_item, f, default=str); f.write('\n')
    except Exception as e: print(f"\n--- WARNING: Failed to save incremental result: {e} ---")

def load_previous_results(filename: str) -> list[dict]:
    results = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                for l_num, line in enumerate(f, 1):
                    if line.strip():
                        try: results.append(json.loads(line))
                        except json.JSONDecodeError as je: print(f"  Warn: JSON decode error in {filename} L{l_num}: {je}.")
        except Exception as e: print(f"  Warn: Error loading results from {filename}: {e}.")
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
                if os.path.getsize(f_path) > 200: resumable_candidates.append((trial_num, f_path)) # Basic size check
            except (ValueError, OSError): pass
    if not resumable_candidates: return None
    resumable_candidates.sort(key=lambda x: x[0]) # Prioritize lower trial numbers for reuse
    return resumable_candidates[0][1]

def generate_dilation_factors(strategy_name: str, num_blocks: int) -> list[int]:
    if strategy_name == "all_ones": return [1] * num_blocks
    elif strategy_name == "progressive_trim": return FCN_DILATION_PROGRESSIVE_PATTERN[:num_blocks]
    elif strategy_name == "cyclic_124": return [FCN_DILATION_CYCLIC_PATTERN[i % len(FCN_DILATION_CYCLIC_PATTERN)] for i in range(num_blocks)]
    else: print(f"Warning: Unknown dilation strategy '{strategy_name}'. Defaulting to 'all_ones'."); return [1] * num_blocks


def objective(trial: optuna.Trial) -> float:
    global CHECKPOINT_DIR_BASE, REPLAY_BUFFER_DIR_BASE, PLOT_DIR_BASE, TOTAL_TARGET_EPISODES

    if torch.cuda.is_available(): device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = torch.device("mps")
    else: device = torch.device("cpu")

    trial_number = trial.number
    print(f"\n--- Starting Optuna Trial {trial_number} (FCN+Dil+Success+Huber+Clip) | Device: {device} ---")

    start_cycle = 0; total_episodes_offset = 0; total_steps_offset = 0
    initial_rewards = []; initial_success_flags = []
    hyperparams_to_use = {}; loaded_agent_checkpoint_state = None
    loaded_trainer_loop_config = {}; is_resuming = False; resumed_from_path = None
    current_trial_checkpoint_path = os.path.join(CHECKPOINT_DIR_BASE, f"checkpoint_trial_{trial_number}.pth")

    if os.path.exists(current_trial_checkpoint_path):
        resumable_checkpoint_path = current_trial_checkpoint_path
    else:
        resumable_checkpoint_path = find_resumable_checkpoint(CHECKPOINT_DIR_BASE, exclude_trial_num=trial_number)

    if resumable_checkpoint_path:
        print(f"Attempting to load state from: {resumable_checkpoint_path}")
        loaded_checkpoint_content = Trainer.load_checkpoint_data(resumable_checkpoint_path)
        if loaded_checkpoint_content:
            trainer_chkpt_state = loaded_checkpoint_content.get('trainer_checkpoint_state', {})
            loaded_agent_checkpoint_state = loaded_checkpoint_content.get('agent_checkpoint_state', None)
            if loaded_agent_checkpoint_state and \
               'hyperparams_of_this_run' in trainer_chkpt_state and \
               'episode_success_flags' in trainer_chkpt_state: # Check for success flags
                is_resuming = True; resumed_from_path = resumable_checkpoint_path
                hyperparams_to_use = trainer_chkpt_state['hyperparams_of_this_run']
                start_cycle = trainer_chkpt_state.get('current_cycle', -1) + 1 # Start from next cycle
                total_episodes_offset = trainer_chkpt_state.get('total_episodes_run', 0)
                total_steps_offset = trainer_chkpt_state.get('total_steps_run', 0)
                initial_rewards = trainer_chkpt_state.get('episode_rewards', [])
                initial_success_flags = trainer_chkpt_state.get('episode_success_flags', []) # Load flags
                loaded_trainer_loop_config = trainer_chkpt_state.get('trainer_loop_config', {})
                resumed_successful_episodes_count = sum(initial_success_flags) # Recalculate from flags
                print(f"  Resuming from end of cycle: {start_cycle - 1}. Prev Eps: {total_episodes_offset}. Prev Successes: {resumed_successful_episodes_count}")
                if resumable_checkpoint_path != current_trial_checkpoint_path:
                    try:
                        os.replace(resumable_checkpoint_path, current_trial_checkpoint_path)
                        resumed_from_path = current_trial_checkpoint_path # Update path after successful move
                    except OSError as e:
                        print(f"ERROR renaming checkpoint: {e}. Aborting resumption attempt for T{trial_number}.")
                        is_resuming = False # Reset flags as resumption failed
                        start_cycle = 0; total_episodes_offset = 0; total_steps_offset = 0; initial_rewards = []; initial_success_flags = []
                        hyperparams_to_use = {}; loaded_agent_checkpoint_state = None; loaded_trainer_loop_config = {}
            else:
                print("ERROR: Checkpoint incomplete or missing critical fields. Cannot resume.")
                if os.path.exists(resumable_checkpoint_path): # If a bad chkpt was found (not current one)
                    print(f"  Deleting potentially corrupted/incomplete checkpoint: {resumable_checkpoint_path}")
                    try: os.remove(resumable_checkpoint_path)
                    except OSError as e: print(f"  Error deleting corrupted chkpt: {e}")

    if is_resuming and hyperparams_to_use:
        print(f"Trial {trial_number} resuming with HPs loaded from checkpoint:")
        print(json.dumps(hyperparams_to_use, indent=2, default=repr))
        for key, value in hyperparams_to_use.items():
            try: trial.set_user_attr(f"loaded_{key}", value) # Store loaded HPs in Optuna trial for reference
            except Exception: pass # Optuna might complain if type is unexpected
        eps_per_cycle = loaded_trainer_loop_config.get('episodes_per_cycle', EPISODES_PER_CYCLE)
        opt_per_cycle = loaded_trainer_loop_config.get('opt_steps_per_cycle', OPT_STEPS_PER_CYCLE)
        num_cycles_target = loaded_trainer_loop_config.get('num_cycles_target', NUM_CYCLES)
    else:
        is_resuming = False # Ensure this is false if not resuming
        print(f"Trial {trial_number} starting with newly suggested HPs.")
        hyperparams_to_use = {
            "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
            "gamma": trial.suggest_float("gamma", 0.97, 0.999),
            "epsilon_decay": trial.suggest_float("epsilon_decay", 0.9995, 0.99999),
            "epsilon_min": trial.suggest_float("epsilon_min", 0.01, 0.05),
            "tau": trial.suggest_float("tau", 0.005, 0.05, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "eps_per_cycle": trial.suggest_int("eps_per_cycle", 50, 200, step=50),
            "opt_per_cycle": trial.suggest_int("opt_per_cycle", 200, 1000, step=100),
            "use_lr_scheduler": trial.suggest_categorical("use_lr_scheduler", [True, False]),
            "lr_end_factor": trial.suggest_float("lr_end_factor", 0.01, 0.2),
            "lr_decay_cycle_fraction": trial.suggest_float("lr_decay_cycle_fraction", 0.6, 1.0),
            "fcn_input_channels": FCN_INPUT_CHANNELS, # Fixed based on env
            "fcn_num_filters": trial.suggest_int("fcn_num_filters", 4, 32, step=4),
            "fcn_blocks": trial.suggest_int("fcn_blocks", 3, 8),
            "fcn_kernel_size": trial.suggest_categorical("fcn_kernel_size", [3, 5]),
            "dilation_strategy": trial.suggest_categorical("dilation_strategy", FCN_DILATION_STRATEGIES),
            "clip_grad_norm_max": trial.suggest_float("clip_grad_norm_max", 0.5, 5.0, log=True), # <<< Added
        }
        eps_per_cycle = hyperparams_to_use['eps_per_cycle']
        opt_per_cycle = hyperparams_to_use['opt_per_cycle']
        num_cycles_target = int(np.ceil(TOTAL_TARGET_EPISODES / eps_per_cycle)) if eps_per_cycle > 0 else NUM_CYCLES
        hyperparams_to_use['num_cycles_target'] = num_cycles_target
        hyperparams_to_use['lr_decay_cycles_calculated'] = int(np.ceil(num_cycles_target * hyperparams_to_use['lr_decay_cycle_fraction']))
        hyperparams_to_use['buffer_capacity'] = REPLAY_BUFFER_CAPACITY # From config
        hyperparams_to_use['fcn_dilation_factors'] = generate_dilation_factors(
            hyperparams_to_use['dilation_strategy'], hyperparams_to_use['fcn_blocks']
        )
        print("--- Hyperparameters Suggested/Constructed ---")
        print(json.dumps(hyperparams_to_use, indent=2, default=repr))
        print("-------------------------------------------")

    env = CompositeDesignEnv()
    try:
        agent = DQNAgent(
            device=device, matrix_size=MATRIX_SIZE,
            fcn_input_channels=hyperparams_to_use['fcn_input_channels'],
            num_scalar_metrics_env=env.num_scalar_metrics,
            lr=hyperparams_to_use['lr'], gamma=hyperparams_to_use['gamma'], batch_size=hyperparams_to_use['batch_size'],
            fcn_num_filters=hyperparams_to_use['fcn_num_filters'], fcn_blocks=hyperparams_to_use['fcn_blocks'],
            fcn_kernel_size=hyperparams_to_use['fcn_kernel_size'], fcn_dilation_factors=hyperparams_to_use['fcn_dilation_factors'],
            epsilon_start=loaded_agent_checkpoint_state['epsilon'] if is_resuming and loaded_agent_checkpoint_state else 1.0,
            epsilon_decay=hyperparams_to_use['epsilon_decay'], epsilon_min=hyperparams_to_use['epsilon_min'], tau=hyperparams_to_use['tau'],
            use_lr_scheduler=hyperparams_to_use['use_lr_scheduler'], lr_end_factor=hyperparams_to_use['lr_end_factor'],
            lr_decay_cycles=hyperparams_to_use.get('lr_decay_cycles_calculated', num_cycles_target),
            buffer_capacity=hyperparams_to_use['buffer_capacity'], buffer_dir=REPLAY_BUFFER_DIR_BASE,
            clip_grad_norm_max=hyperparams_to_use['clip_grad_norm_max']
        )
    except Exception as e: print(f"ERROR: DQNAgent init failed: {e}"); traceback.print_exc(); raise optuna.TrialPruned(f"Agent init error: {e}")

    if is_resuming and loaded_agent_checkpoint_state:
        print(f"Loading agent state dict for Trial {trial_number}...")
        try: agent.load_state(loaded_agent_checkpoint_state)
        except Exception as e: print(f"ERROR loading agent state: {e}"); traceback.print_exc(); raise optuna.TrialPruned(f"Agent state load error: {e}")

    trainer = Trainer(
        env=env, agent=agent, device=device, trial_number=trial_number,
        episodes_per_cycle=eps_per_cycle,
        opt_steps_per_cycle=opt_per_cycle,
        num_cycles=num_cycles_target,
        checkpoint_dir=CHECKPOINT_DIR_BASE,
        start_cycle=start_cycle,
        total_episodes_offset=total_episodes_offset,
        total_steps_offset=total_steps_offset,
        initial_rewards=initial_rewards,
        initial_success_flags=initial_success_flags, # Pass loaded flags
        hyperparams=hyperparams_to_use
    )

    final_score = -float('inf'); training_status = "UNKNOWN_INIT_FAIL"; final_window_success_rate = 0.0
    try:
        final_score = trainer.train() # This now returns the final score (e.g., avg reward of last N episodes)
        training_status = "COMPLETE"
        final_window_success_rate = trainer.final_window_success_rate # Get from trainer
        print(f"Trial {trial_number} COMPLETED. Score: {final_score:.4f}, Win Success: {final_window_success_rate:.2f}%")

        # Delete checkpoint only if trial completed successfully and was not pruned/failed
        if os.path.exists(current_trial_checkpoint_path):
            try: os.remove(current_trial_checkpoint_path); print(f"  Deleted successful trial checkpoint: {current_trial_checkpoint_path}")
            except OSError as e: print(f"  Warning: Failed to delete successful trial chkpt: {e}")

        plot_path = os.path.join(PLOT_DIR_BASE, f"trial_{trial_number}_rewards_COMPLETE.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        trainer.plot_rewards(window=100, save_path=plot_path)
        return final_score # Return score to Optuna

    except KeyboardInterrupt:
        training_status = "INTERRUPTED_USER"
        print(f"\nTrial {trial_number} INTERRUPTED by user.")
        # Do NOT delete checkpoint on user interrupt, allow for later resumption
        # But if a checkpoint for *this specific trial number* exists, it's from this run.
        if os.path.exists(current_trial_checkpoint_path):
             print(f"  Checkpoint for interrupted T{trial_number} remains: {current_trial_checkpoint_path}")
        else:
             print(f"  No specific checkpoint file found for currently interrupted T{trial_number} to preserve (may have been renamed/resumed).")
        raise optuna.TrialPruned("Trial interrupted by user, checkpoint (if saved by trainer) may be preserved.")

    except optuna.TrialPruned as e_prune:
        training_status = "PRUNED_OPTUNA" # Optuna itself pruned it (e.g. via a pruner callback)
        print(f"Trial {trial_number} PRUNED by Optuna: {e_prune}")
        plot_path = os.path.join(PLOT_DIR_BASE, f"trial_{trial_number}_rewards_PRUNED.png")
        if 'trainer' in locals() and trainer.episode_rewards: # Check if trainer and rewards exist
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            trainer.plot_rewards(window=100, save_path=plot_path)
        # Checkpoint for pruned trial might still be useful if it ran for a bit.
        # Optuna's default is to not retry pruned trials.
        raise e_prune # Re-raise to Optuna

    except Exception as e_train: # Catch other errors during training
        training_status = "FAILED_RUNTIME"
        print(f"Trial {trial_number} FAILED with runtime error: {e_train}")
        traceback.print_exc()
        plot_path = os.path.join(PLOT_DIR_BASE, f"trial_{trial_number}_rewards_FAILED.png")
        if 'trainer' in locals() and trainer.episode_rewards:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            trainer.plot_rewards(window=100, save_path=plot_path)
        # Checkpoint is likely saved by trainer before error, can be useful for debugging
        raise optuna.TrialPruned(f"Runtime error in trial: {e_train}") # Prune on other exceptions too

    finally:
        print(f"Executing 'finally' block for Trial {trial_number}. Current status: {training_status}")
        if 'agent' in locals() and agent: agent.close_buffer()
        if 'env' in locals() and env: env.close()

        # Ensure final_window_success_rate is captured from trainer if it exists
        if 'trainer' in locals() and hasattr(trainer, 'final_window_success_rate'):
            final_window_success_rate = trainer.final_window_success_rate

        current_result_summary = {
            "trial_number": trial_number,
            "status": training_status,
            "final_score_reported_to_optuna": final_score if training_status == "COMPLETE" else None,
            "final_window_success_rate_percent": round(final_window_success_rate, 2),
            "hyperparameters_used": hyperparams_to_use,
            "last_cycle_completed": trainer.current_cycle if 'trainer' in locals() else start_cycle -1, # -1 if not even one cycle done
            "total_episodes_run_in_trial": (trainer.total_episodes_run - total_episodes_offset) if 'trainer' in locals() else 0,
            "was_resumed_at_start": is_resuming,
            "resumed_from_file_path": os.path.basename(resumed_from_path) if resumed_from_path else None,
            "timestamp_end": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_results_incrementally(current_result_summary, RESULTS_FILE_BASE)
        print(f"--- Finished 'finally' for Trial {trial_number} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna FCN DQN with Dilation, Success Tracking, Huber Loss, Grad Clip")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials to run.")
    parser.add_argument("--study-name", type=str, default="DQN_FCN_Dil_HuberClip", help="Name for the Optuna study.")
    parser.add_argument("--storage", type=str, default="sqlite:///dqn_fcn_dil_huberclip.db", help="Optuna storage URL.")
    parser.add_argument("--results-file", type=str, default=RESULTS_FILE_BASE, help="Path to the JSONL results file.")
    parser.add_argument("--plot-dir", type=str, default=PLOT_DIR_BASE, help="Directory to save trial plots.")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR_BASE, help="Directory for checkpoints.")
    parser.add_argument("--buffer-dir", type=str, default=REPLAY_BUFFER_DIR_BASE, help="Directory for replay buffer data.")
    parser.add_argument("--target-episodes", type=int, default=TOTAL_TARGET_EPISODES, help="Target episodes per trial.")
    parser.add_argument("--cleanup-buffer-ask", action="store_true", help="Ask to cleanup buffer dir after study.")
    args = parser.parse_args()

    CHECKPOINT_DIR_BASE = args.checkpoint_dir
    REPLAY_BUFFER_DIR_BASE = args.buffer_dir
    PLOT_DIR_BASE = args.plot_dir
    RESULTS_FILE_BASE = args.results_file
    TOTAL_TARGET_EPISODES = args.target_episodes

    os.makedirs(CHECKPOINT_DIR_BASE, exist_ok=True)
    os.makedirs(REPLAY_BUFFER_DIR_BASE, exist_ok=True) # Ensure buffer base dir exists
    os.makedirs(PLOT_DIR_BASE, exist_ok=True)

    initial_results_count = len(load_previous_results(RESULTS_FILE_BASE))
    print(f"Found {initial_results_count} previous execution records in '{RESULTS_FILE_BASE}'.")

    print("\n--- Initializing Optuna Study (FCN Dilation+Success+Huber+Clip) ---")
    print(f"  Study Name: {args.study_name}")
    print(f"  Storage: {args.storage}")
    print(f"  Target Trials for this run: {args.trials}")
    print(f"  Dirs: Checkpoints='{CHECKPOINT_DIR_BASE}', Buffer='{REPLAY_BUFFER_DIR_BASE}', Plots='{PLOT_DIR_BASE}'")
    print(f"  Results File: '{RESULTS_FILE_BASE}'")
    print(f"  Target Episodes per Trial: {TOTAL_TARGET_EPISODES}")
    print("-" * 30)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize" # Maximize the final score (e.g. average reward)
    )

    # Enqueue parameters from previous best trials or failed trials if desired
    # Example: Enqueue failed trials to retry them first
    # previous_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.FAIL,))
    # for t in previous_trials:
    # study.enqueue_trial(t.params)

    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            gc_after_trial=True, # Helps manage memory
            n_jobs=1, # FEniCS might not be thread-safe with some solvers/setups
            catch=(KeyboardInterrupt,) # Catch KeyboardInterrupt during optimize loop
        )
    except KeyboardInterrupt:
        print("\nOptuna study.optimize() loop was INTERRUPTED by user.")
    except Exception as e:
        print(f"\nOptuna study.optimize() loop FAILED with an unexpected error: {e}")
        traceback.print_exc()
    finally:
        print("\n--- Optuna Study Session Concluded ---")
        print(f"\n--- Summary from Optuna Study Object ({study.study_name}) ---")
        all_trials_from_study = []
        try:
            all_trials_from_study = study.get_trials(deepcopy=False) # Get all trials associated with the study name
        except Exception as e_study:
            print(f"  Error fetching trials from study object: {e_study}")

        complete_study_trials = [t for t in all_trials_from_study if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_study_trials = [t for t in all_trials_from_study if t.state == optuna.trial.TrialState.PRUNED]
        fail_study_trials = [t for t in all_trials_from_study if t.state == optuna.trial.TrialState.FAIL]
        running_study_trials = [t for t in all_trials_from_study if t.state == optuna.trial.TrialState.RUNNING] # Should be 0 if optimize finished
        waiting_study_trials = [t for t in all_trials_from_study if t.state == optuna.trial.TrialState.WAITING] # Should be 0

        print(f"Total Trials in Optuna DB for this study: {len(all_trials_from_study)}")
        print(f"  Complete: {len(complete_study_trials)}")
        print(f"  Pruned:   {len(pruned_study_trials)}")
        print(f"  Failed:   {len(fail_study_trials)}")
        if running_study_trials: print(f"  Running:  {len(running_study_trials)} (Should be 0 if optimize finished cleanly)")
        if waiting_study_trials: print(f"  Waiting:  {len(waiting_study_trials)} (Should be 0)")


        best_optuna_trial = None
        try: best_optuna_trial = study.best_trial
        except ValueError: print("  No best trial found in Optuna study (e.g., no trials completed).") # If no trials completed
        
        if best_optuna_trial:
            print(f"Best Optuna Trial (from study.best_trial, likely COMPLETE):")
            print(f"  Trial Number: {best_optuna_trial.number}")
            print(f"  Value (Score): {best_optuna_trial.value:.5f}")
            print(f"  Params: {json.dumps(best_optuna_trial.params, indent=4)}")
            # User attributes can also be insightful
            # print(f"  User Attrs: {json.dumps(best_optuna_trial.user_attrs, indent=4)}")
        else:
            # If study.best_trial fails or is None, try to find best from completed list
            if complete_study_trials:
                best_from_complete = max(complete_study_trials, key=lambda t: t.value if t.value is not None else -float('inf'))
                print(f"Best Optuna Trial (manually from COMPLETED list):")
                print(f"  Trial Number: {best_from_complete.number}")
                print(f"  Value (Score): {best_from_complete.value:.5f}")
                print(f"  Params: {json.dumps(best_from_complete.params, indent=4)}")


        print(f"\n--- Summary from Incremental Results File ('{RESULTS_FILE_BASE}') ---")
        all_logged_records = load_previous_results(RESULTS_FILE_BASE)
        print(f"Total execution attempts logged in JSONL: {len(all_logged_records)}")
        status_counts = {}
        final_scores_complete = []
        success_rates_complete = []

        for record in all_logged_records:
            status = record.get("status", "UNKNOWN_STATUS")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "COMPLETE":
                score = record.get("final_score_reported_to_optuna")
                if score is not None: final_scores_complete.append(float(score))
                success_rate = record.get("final_window_success_rate_percent")
                if success_rate is not None: success_rates_complete.append(float(success_rate))

        print("Counts by recorded status in JSONL:")
        for s, c in sorted(status_counts.items()): print(f"  {s}: {c}")

        if final_scores_complete:
            print(f"Avg Final Score (from COMPLETE in JSONL): {np.mean(final_scores_complete):.4f} (Std: {np.std(final_scores_complete):.4f}, N={len(final_scores_complete)})")
        else: print("No final scores from COMPLETE trials found in JSONL.")
        if success_rates_complete:
            print(f"Avg Window Success Rate (from COMPLETE in JSONL): {np.mean(success_rates_complete):.2f}% (Std: {np.std(success_rates_complete):.2f}%, N={len(success_rates_complete)})")
        else: print("No success rates from COMPLETE trials found in JSONL.")

        if args.cleanup_buffer_ask:
             print(f"\nReplay buffer data is in: {REPLAY_BUFFER_DIR_BASE}")
             if os.path.exists(REPLAY_BUFFER_DIR_BASE) and any(os.scandir(REPLAY_BUFFER_DIR_BASE)):
                 confirm = input(f"  Delete all contents of the replay buffer directory '{REPLAY_BUFFER_DIR_BASE}'? [y/N]: ").lower()
                 if confirm == 'y':
                     try:
                         shutil.rmtree(REPLAY_BUFFER_DIR_BASE)
                         os.makedirs(REPLAY_BUFFER_DIR_BASE) # Recreate the directory
                         print(f"  Buffer directory '{REPLAY_BUFFER_DIR_BASE}' cleared and recreated.")
                     except Exception as e_del:
                         print(f"  ERROR deleting buffer directory: {e_del}")
                 else:
                     print("  Buffer directory not deleted.")
             else:
                 print("  Buffer directory is empty or does not exist. No cleanup needed.")
        print("\n--- Script Finished ---")