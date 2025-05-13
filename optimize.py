# optimize.py

# ... (Imports, helper functions, config loading remain the same) ...
# --- Imports ---
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
    REPLAY_BUFFER_CAPACITY,
    EPISODES_PER_CYCLE, OPT_STEPS_PER_CYCLE, NUM_CYCLES, TOTAL_TARGET_EPISODES
)
# --- End Imports ---

# --- Global Paths (Set by args later) ---
CHECKPOINT_DIR_BASE = "checkpoints_fcn_dil"
REPLAY_BUFFER_DIR_BASE = "replay_buffer_data_fcn_dil"
PLOT_DIR_BASE = "trial_plots_fcn_dil"
RESULTS_FILE_BASE = "trial_results_fcn_dil.jsonl"

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

# --- Helper Functions (save_results_incrementally, load_previous_results, find_resumable_checkpoint, generate_dilation_factors) ---
# (These should be the same as the previous correct version)
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
                        except json.JSONDecodeError as je: print(f" Warn: JSON decode error in {filename} L{l_num}: {je}.")
        except Exception as e: print(f" Warn: Error loading results from {filename}: {e}.")
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
                if os.path.getsize(f_path) > 200: resumable_candidates.append((trial_num, f_path))
            except (ValueError, OSError): pass
    if not resumable_candidates: return None
    resumable_candidates.sort(key=lambda x: x[0])
    return resumable_candidates[0][1]

def generate_dilation_factors(strategy_name: str, num_blocks: int) -> list[int]:
    if strategy_name == "all_ones": return [1] * num_blocks
    elif strategy_name == "progressive_trim": return FCN_DILATION_PROGRESSIVE_PATTERN[:num_blocks]
    elif strategy_name == "cyclic_124": return [FCN_DILATION_CYCLIC_PATTERN[i % len(FCN_DILATION_CYCLIC_PATTERN)] for i in range(num_blocks)]
    else: print(f"Warning: Unknown dilation strategy '{strategy_name}'. Defaulting to 'all_ones'."); return [1] * num_blocks


def objective(trial: optuna.Trial) -> float:
    global CHECKPOINT_DIR_BASE, REPLAY_BUFFER_DIR_BASE, PLOT_DIR_BASE, TOTAL_TARGET_EPISODES

    # --- Device Selection ---
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = torch.device("mps")
    else: device = torch.device("cpu")

    trial_number = trial.number
    print(f"\n--- Starting Optuna Trial {trial_number} (FCN+Dilation+Success) | Device: {device} ---")

    # --- Resumption State Init ---
    start_cycle = 0; total_episodes_offset = 0; total_steps_offset = 0
    # No need for total_successful_episodes_offset here, calculated from flags later
    initial_rewards = []; initial_success_flags = []
    hyperparams_to_use = {}; loaded_agent_checkpoint_state = None
    loaded_trainer_loop_config = {}; is_resuming = False; resumed_from_path = None
    current_trial_checkpoint_path = os.path.join(CHECKPOINT_DIR_BASE, f"checkpoint_trial_{trial_number}.pth")

    # --- Resumption Logic ---
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
            # Check essential keys for successful resumption
            if loaded_agent_checkpoint_state and 'hyperparams_of_this_run' in trainer_chkpt_state and 'episode_success_flags' in trainer_chkpt_state:
                is_resuming = True; resumed_from_path = resumable_checkpoint_path
                hyperparams_to_use = trainer_chkpt_state['hyperparams_of_this_run']
                start_cycle = trainer_chkpt_state.get('current_cycle', -1) + 1
                total_episodes_offset = trainer_chkpt_state.get('total_episodes_run', 0)
                total_steps_offset = trainer_chkpt_state.get('total_steps_run', 0)
                initial_rewards = trainer_chkpt_state.get('episode_rewards', [])
                initial_success_flags = trainer_chkpt_state.get('episode_success_flags', []) # Load flags
                loaded_trainer_loop_config = trainer_chkpt_state.get('trainer_loop_config', {})
                # Calculate success count offset from loaded flags
                resumed_successful_episodes_count = sum(initial_success_flags)
                print(f"  Resuming from end of cycle: {start_cycle - 1}. Prev Eps: {total_episodes_offset}. Prev Successes: {resumed_successful_episodes_count}")
                if resumable_checkpoint_path != current_trial_checkpoint_path:
                    try: os.replace(resumable_checkpoint_path, current_trial_checkpoint_path); resumed_from_path = current_trial_checkpoint_path
                    except OSError as e:
                        print(f"ERROR renaming chkpt: {e}. Aborting resumption."); is_resuming=False
                        start_cycle = 0; total_episodes_offset = 0; total_steps_offset = 0; initial_rewards = []; initial_success_flags = []
                        hyperparams_to_use = {}; loaded_agent_checkpoint_state = None; loaded_trainer_loop_config = {}
            else:
                print("ERROR: Checkpoint incomplete (missing agent state, HPs, or success flags). Cannot resume.")


    # --- Determine HPs ---
    if is_resuming and hyperparams_to_use:
        print(f"Trial {trial_number} resuming with HPs loaded from checkpoint:")
        print(json.dumps(hyperparams_to_use, indent=2, default=repr))
        for key, value in hyperparams_to_use.items():
            try: trial.set_user_attr(f"loaded_{key}", value)
            except Exception: pass
        eps_per_cycle = loaded_trainer_loop_config.get('episodes_per_cycle', EPISODES_PER_CYCLE)
        opt_per_cycle = loaded_trainer_loop_config.get('opt_steps_per_cycle', OPT_STEPS_PER_CYCLE)
        num_cycles_target = loaded_trainer_loop_config.get('num_cycles_target', NUM_CYCLES)
    else:
        # ... (Suggest HPs logic remains the same, including generating fcn_dilation_factors) ...
        is_resuming = False
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
            "fcn_input_channels": FCN_INPUT_CHANNELS,
            "fcn_num_filters": trial.suggest_int("fcn_num_filters", 8, 32, step=8),
            "fcn_blocks": trial.suggest_int("fcn_blocks", 3, 8),
            "fcn_kernel_size": trial.suggest_categorical("fcn_kernel_size", [3, 5]),
            "dilation_strategy": trial.suggest_categorical("dilation_strategy", FCN_DILATION_STRATEGIES),
        }
        eps_per_cycle = hyperparams_to_use['eps_per_cycle']
        opt_per_cycle = hyperparams_to_use['opt_per_cycle']
        num_cycles_target = int(np.ceil(TOTAL_TARGET_EPISODES / eps_per_cycle)) if eps_per_cycle > 0 else NUM_CYCLES
        hyperparams_to_use['num_cycles_target'] = num_cycles_target
        hyperparams_to_use['lr_decay_cycles_calculated'] = int(np.ceil(num_cycles_target * hyperparams_to_use['lr_decay_cycle_fraction']))
        hyperparams_to_use['buffer_capacity'] = REPLAY_BUFFER_CAPACITY
        hyperparams_to_use['fcn_dilation_factors'] = generate_dilation_factors(
            hyperparams_to_use['dilation_strategy'], hyperparams_to_use['fcn_blocks']
        )
        print("--- Hyperparameters Suggested/Constructed ---")
        print(json.dumps(hyperparams_to_use, indent=2, default=repr))
        print("-------------------------------------------")

    # --- Setup Env & Agent ---
    env = CompositeDesignEnv()
    try:
        # ... (Agent initialization remains the same, using hyperparams_to_use) ...
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
            buffer_capacity=hyperparams_to_use['buffer_capacity'], buffer_dir=REPLAY_BUFFER_DIR_BASE
        )
    except Exception as e: print(f"ERROR: DQNAgent init failed: {e}"); traceback.print_exc(); raise optuna.TrialPruned(f"Agent init error: {e}")

    if is_resuming and loaded_agent_checkpoint_state:
        # ... (Load agent state remains the same) ...
        print(f"Loading agent state dict for Trial {trial_number}...")
        try: agent.load_state(loaded_agent_checkpoint_state)
        except Exception as e: print(f"ERROR loading agent state: {e}"); traceback.print_exc(); raise optuna.TrialPruned(f"Agent state load error: {e}")

    # --- Setup Trainer ---
    # Removed the incorrect offset parameter
    trainer = Trainer(
        env=env, agent=agent, device=device, trial_number=trial_number,
        episodes_per_cycle=eps_per_cycle,
        opt_steps_per_cycle=opt_per_cycle,
        num_cycles=num_cycles_target,
        checkpoint_dir=CHECKPOINT_DIR_BASE,
        start_cycle=start_cycle,
        total_episodes_offset=total_episodes_offset,
        total_steps_offset=total_steps_offset,
        # No successful offset needed here, it's derived from flags list
        initial_rewards=initial_rewards,
        initial_success_flags=initial_success_flags, # Pass the loaded flags
        hyperparams=hyperparams_to_use
    )

    # --- Run Training ---
    # ... (The try/except KeyboardInterrupt/finally block remains the same as the previous version) ...
    # It handles training, status updates, final result saving (including success rate),
    # and checkpoint deletion on interrupt/success.
    final_score = -float('inf'); training_status = "UNKNOWN_INIT_FAIL"; final_window_success_rate = 0.0
    try:
        final_score = trainer.train()
        training_status = "COMPLETE"
        final_window_success_rate = trainer.final_window_success_rate
        print(f"Trial {trial_number} COMPLETED. Score: {final_score:.4f}, Win Success: {final_window_success_rate:.2f}%")
        if os.path.exists(current_trial_checkpoint_path):
            try: os.remove(current_trial_checkpoint_path); print(f"  Deleted successful trial chkpt.")
            except OSError as e: print(f"  Warn: Failed to delete successful chkpt: {e}")
        plot_path = os.path.join(PLOT_DIR_BASE, f"trial_{trial_number}_rewards_COMPLETE.png"); os.makedirs(os.path.dirname(plot_path), exist_ok=True); trainer.plot_rewards(window=100, save_path=plot_path)
        return final_score
    except KeyboardInterrupt:
        training_status = "INTERRUPTED_USER"; print(f"\nTrial {trial_number} INTERRUPTED.")
        if os.path.exists(current_trial_checkpoint_path):
            try: os.remove(current_trial_checkpoint_path); print(f"  Deleted interrupted trial chkpt.")
            except OSError as e: print(f"  Warn: Failed to delete chkpt for interrupted trial: {e}")
        else: print(f"  No chkpt found to delete for interrupted T{trial_number}.")
        raise optuna.TrialPruned("Trial interrupted by user and checkpoint deleted.")
    except optuna.TrialPruned as e_prune:
        training_status = "PRUNED_OPTUNA"; print(f"Trial {trial_number} PRUNED: {e_prune}")
        plot_path = os.path.join(PLOT_DIR_BASE, f"trial_{trial_number}_rewards_PRUNED.png")
        if 'trainer' in locals() and trainer.episode_rewards: os.makedirs(os.path.dirname(plot_path),exist_ok=True); trainer.plot_rewards(window=100, save_path=plot_path)
        raise e_prune
    except Exception as e_train:
        training_status = "FAILED_RUNTIME"; print(f"Trial {trial_number} FAILED: {e_train}"); traceback.print_exc()
        plot_path = os.path.join(PLOT_DIR_BASE, f"trial_{trial_number}_rewards_FAILED.png")
        if 'trainer' in locals() and trainer.episode_rewards: os.makedirs(os.path.dirname(plot_path),exist_ok=True); trainer.plot_rewards(window=100, save_path=plot_path)
        raise optuna.TrialPruned(f"Runtime error in trial: {e_train}")
    finally:
        print(f"Executing 'finally' for Trial {trial_number}. Status: {training_status}")
        if 'agent' in locals() and agent: agent.close_buffer()
        if 'env' in locals() and env: env.close()
        if 'trainer' in locals() and hasattr(trainer, 'final_window_success_rate'): final_window_success_rate = trainer.final_window_success_rate
        current_result_summary = {
            "trial_number": trial_number, "status": training_status,
            "final_score_reported_to_optuna": final_score if training_status == "COMPLETE" else None,
            "final_window_success_rate_percent": round(final_window_success_rate, 2),
            "hyperparameters_used": hyperparams_to_use,
            "last_cycle_completed": trainer.current_cycle if 'trainer' in locals() else start_cycle -1,
            "total_episodes_run_in_trial": trainer.total_episodes_run - total_episodes_offset if 'trainer' in locals() else 0,
            "was_resumed_at_start": is_resuming, "resumed_from_file_path": os.path.basename(resumed_from_path) if resumed_from_path else None,
            "timestamp_end": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_results_incrementally(current_result_summary, RESULTS_FILE_BASE)
        print(f"--- Finished 'finally' for Trial {trial_number} ---")


# --- if __name__ == "__main__": block remains the same as previous ---
if __name__ == "__main__":
    # ... (Arg parsing same as previous) ...
    parser = argparse.ArgumentParser(description="Run Optuna FCN DQN with Dilation Strategies + Success Tracking")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--study-name", type=str, default="DQN_FCN_DilStrat")
    parser.add_argument("--storage", type=str, default="sqlite:///dqn_fcn_dilstrat.db")
    parser.add_argument("--results-file", type=str, default=RESULTS_FILE_BASE)
    parser.add_argument("--plot-dir", type=str, default=PLOT_DIR_BASE)
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR_BASE)
    parser.add_argument("--buffer-dir", type=str, default=REPLAY_BUFFER_DIR_BASE)
    parser.add_argument("--target-episodes", type=int, default=TOTAL_TARGET_EPISODES, help="Target episodes per trial.")
    parser.add_argument("--cleanup-buffer-ask", action="store_true")
    args = parser.parse_args()

    CHECKPOINT_DIR_BASE = args.checkpoint_dir; REPLAY_BUFFER_DIR_BASE = args.buffer_dir
    PLOT_DIR_BASE = args.plot_dir; RESULTS_FILE_BASE = args.results_file
    TOTAL_TARGET_EPISODES = args.target_episodes
    os.makedirs(CHECKPOINT_DIR_BASE, exist_ok=True); os.makedirs(REPLAY_BUFFER_DIR_BASE, exist_ok=True); os.makedirs(PLOT_DIR_BASE, exist_ok=True)
    initial_results_count = len(load_previous_results(RESULTS_FILE_BASE)); print(f"Found {initial_results_count} records in '{RESULTS_FILE_BASE}'.")

    print("\n--- Initializing Optuna Study (FCN Dilation+Success) ---")
    # ... (Study init) ...
    print(f"  Study: {args.study_name}, Storage: {args.storage}, Target Trials: {args.trials}")
    print(f"  Dirs: Checkpoint='{CHECKPOINT_DIR_BASE}', Buffer='{REPLAY_BUFFER_DIR_BASE}', Plots='{PLOT_DIR_BASE}'")
    print("-" * 30)
    study = optuna.create_study(study_name=args.study_name, storage=args.storage, load_if_exists=True, direction="maximize")

    try:
        study.optimize(objective, n_trials=args.trials, gc_after_trial=True, n_jobs=1, catch=(KeyboardInterrupt,))
    except KeyboardInterrupt: print("\nOptuna study.optimize loop INTERRUPTED by user.")
    except Exception as e: print(f"\nOptuna study.optimize loop FAILED: {e}"); traceback.print_exc()
    finally:
        # ... (Final reporting from study object and JSONL file remains the same) ...
        print("\n--- Optuna Study Session Concluded ---")
        print("\n--- Summary from Optuna Study ---"); print(f"Study: {study.study_name}")
        all_trials = study.get_trials(deepcopy=False)
        complete = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in all_trials if t.state == optuna.trial.TrialState.PRUNED]
        fail = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
        print(f"Total Optuna Trials: {len(all_trials)} (Complete: {len(complete)}, Pruned: {len(pruned)}, Fail: {len(fail)})")
        if study.best_trial: print(f"Best Optuna Trial (COMPLETE): T#{study.best_trial.number}, Value: {study.best_trial.value:.5f}, Params: {study.best_trial.params}")
        else: print("No trials completed successfully in Optuna study.")
        print(f"\n--- Summary from Results File ('{RESULTS_FILE_BASE}') ---")
        all_records = load_previous_results(RESULTS_FILE_BASE); print(f"Total execution attempts logged: {len(all_records)}")
        counts = {}; rates = []; scores = []
        for r in all_records: s = r.get("status","?"); counts[s] = counts.get(s,0)+1;
        if r.get("status")=="COMPLETE": rates.append(r.get("final_window_success_rate_percent", np.nan)); scores.append(r.get("final_score_reported_to_optuna", np.nan))
        print("Counts by recorded status:"); [print(f"  {s}: {c}") for s, c in sorted(counts.items())]
        if not all(np.isnan(rates)): print(f"Avg Window Success Rate (COMPLETE trials): {np.nanmean(rates):.2f}%")
        if not all(np.isnan(scores)): print(f"Avg Final Score (COMPLETE trials): {np.nanmean(scores):.4f}")
        if args.cleanup_buffer_ask:
            print(f"\nReplay buffer is in: {REPLAY_BUFFER_DIR_BASE}")
            if os.path.exists(REPLAY_BUFFER_DIR_BASE) and any(os.scandir(REPLAY_BUFFER_DIR_BASE)):
                confirm = input(f"Delete buffer contents? [y/N]: ").lower()
                if confirm == 'y':
                    try: shutil.rmtree(REPLAY_BUFFER_DIR_BASE); os.makedirs(REPLAY_BUFFER_DIR_BASE); print("  Deleted.")
                    except Exception as e: print(f"  ERR deleting: {e}")
                else: print("  Buffer not deleted.") # Added else
            else: print("  Buffer directory is empty or does not exist.") # Adjusted msg
        print("\n--- Script Finished ---")