# project_root/config.py

# ==============================================================================
#                       COMMON CONFIGURATIONS
# ==============================================================================

# -------------------------------
# Matrix configuration
# -------------------------------
MATRIX_SIZE = 8
MAX_STEPS = int(MATRIX_SIZE * MATRIX_SIZE * 0.5) # Max steps per episode

# -------------------------------
# FEM Material properties
# -------------------------------
E_STIFF = 1818.0
NU_STIFF = 0.33
E_COMP = 364.0
NU_COMP = 0.49

# -------------------------------
# Target Material properties (Examples, environment often resets these dynamically)
# -------------------------------
DESIRED_MODULUS = 1302.17 # Example, will be overridden
DESIRED_VOL_FRAC = 0.20   # Example, will be overridden

# -------------------------------
# Q-Network architecture (Common base parameters)
# -------------------------------
# FCN_INPUT_CHANNELS: 1 (grid) + 4 (broadcasted: current_VF, target_VF, current_E_scaled, target_E_scaled)
FCN_INPUT_CHANNELS = 5

# Dilation strategy definitions (used by Optuna and potentially single runs if generating)
FCN_DILATION_STRATEGIES = ["all_ones", "progressive_trim", "cyclic_124"]
FCN_DILATION_PROGRESSIVE_MAX_BLOCKS = 12 # Max blocks for progressive pattern
FCN_DILATION_PROGRESSIVE_PATTERN = [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4]
FCN_DILATION_CYCLIC_PATTERN = [1, 2, 4]


# ==============================================================================
#              PARAMETERS FOR OPTUNA HYPERPARAMETER SEARCH (optimize.py)
# ==============================================================================

# -------------------------------
# Optuna Default Training Parameters (can be overridden by Optuna suggestions)
# -------------------------------
OPTUNA_DEFAULT_LEARNING_RATE = 1e-4
OPTUNA_DEFAULT_TAU = 0.01
OPTUNA_DEFAULT_BATCH_SIZE = 1024 # Optuna might suggest from a list [256, 512, 1024]
OPTUNA_DEFAULT_GAMMA = 0.99
OPTUNA_DEFAULT_EPSILON_START = 1.0 # Usually fixed for new trials
OPTUNA_DEFAULT_EPSILON_DECAY = 0.99995 # Optuna will suggest a range
OPTUNA_DEFAULT_EPSILON_MIN = 0.02    # Optuna will suggest a range
OPTUNA_DEFAULT_CLIP_GRAD_NORM_MAX = 1.0 # Optuna will suggest a range
OPTUNA_DEFAULT_LOSS_FUNCTION_NAME = "smooth_l1" # Optuna can suggest from a list if configured

# -------------------------------
# Optuna Default Q-Network architecture (Optuna will suggest ranges/categories)
# -------------------------------
OPTUNA_DEFAULT_FCN_NUM_FILTERS_RESBLOCK = 8 # Optuna suggests e.g., 4 to 32
OPTUNA_DEFAULT_FCN_NUM_RES_BLOCKS = 4       # Optuna suggests e.g., 3 to 8
OPTUNA_DEFAULT_FCN_KERNEL_SIZE = 3          # Optuna suggests e.g., [3, 5]

# -------------------------------
# Optuna Default Training Loop Parameters (Optuna will suggest ranges)
# -------------------------------
OPTUNA_DEFAULT_NUM_CYCLES = 1000
OPTUNA_DEFAULT_EPISODES_PER_CYCLE = 100 # Optuna suggests e.g., 50 to 200
OPTUNA_DEFAULT_OPT_STEPS_PER_CYCLE = 500 # Optuna suggests e.g., 200 to 1000

# -------------------------------
# Optuna Default Replay Buffer Parameters
# -------------------------------
OPTUNA_DEFAULT_REPLAY_BUFFER_CAPACITY = 10_000_000 # Can be fixed or suggested

# -------------------------------
# Optuna Study Paths & Settings (used by optimize.py main block)
# -------------------------------
OPTUNA_CHECKPOINT_DIR_BASE = "checkpoints_fcn_optuna" # Base dir for all Optuna trial checkpoints
OPTUNA_REPLAY_BUFFER_DIR_BASE = "replay_buffer_data_fcn_optuna" # Base dir for Optuna trial buffers
OPTUNA_PLOT_DIR_BASE = "trial_plots_fcn_optuna"
OPTUNA_RESULTS_FILE_BASE = "trial_results_fcn_optuna.jsonl"
OPTUNA_TOTAL_TARGET_EPISODES = 20_000 # Target episodes for each Optuna trial


# ==============================================================================
#        PARAMETERS FOR A SPECIFIC SINGLE RUN (e.g., main_single_run.py)
#             (Based on the "best" trial JSON you provided)
# ==============================================================================

# -------------------------------
# Single Run DQN Training Parameters
# -------------------------------
SINGLE_RUN_LEARNING_RATE = 0.0006963266651039465
SINGLE_RUN_TAU = 0.0057097421485604195
SINGLE_RUN_BATCH_SIZE = 512
SINGLE_RUN_GAMMA = 0.9763769179789137
SINGLE_RUN_EPSILON_START = 1.0 # For a new single run
SINGLE_RUN_EPSILON_DECAY = 0.9995817186412947
SINGLE_RUN_EPSILON_MIN = 0.028569045452484964
SINGLE_RUN_CLIP_GRAD_NORM_MAX = 2.4310061277750767
SINGLE_RUN_LOSS_FUNCTION_NAME = "smooth_l1"

# --- Single Run LR Scheduler Specifics ---
SINGLE_RUN_USE_LR_SCHEDULER = True
SINGLE_RUN_LR_END_FACTOR = 0.034301965343095
SINGLE_RUN_LR_DECAY_CYCLES_CALCULATED = 87 # From JSON: "lr_decay_cycles_calculated": 87

# -------------------------------
# Single Run Q-Network architecture parameters
# -------------------------------
SINGLE_RUN_FCN_NUM_FILTERS_RESBLOCK = 16
SINGLE_RUN_FCN_NUM_RES_BLOCKS = 3
SINGLE_RUN_FCN_KERNEL_SIZE = 3
SINGLE_RUN_FCN_DILATION_STRATEGY_NAME = "progressive_trim" # For record-keeping
SINGLE_RUN_FCN_DILATION_FACTORS_LIST = [1, 1, 1] # The actual list to use

# -------------------------------
# Single Run Training Loop Parameters
# -------------------------------
SINGLE_RUN_NUM_CYCLES = 100 # From JSON: "num_cycles_target": 100
SINGLE_RUN_EPISODES_PER_CYCLE = 200
SINGLE_RUN_OPT_STEPS_PER_CYCLE = 300

# -------------------------------
# Single Run Replay Buffer Parameters
# -------------------------------
SINGLE_RUN_REPLAY_BUFFER_CAPACITY = 8_000_000

# -------------------------------
# Single Run Paths
# -------------------------------
SINGLE_RUN_ID_TAG = "trial57_best" # Helps in naming directories for this specific run
SINGLE_RUN_CHECKPOINT_DIR = f"checkpoints_single_run_{SINGLE_RUN_ID_TAG}"
SINGLE_RUN_PLOT_DIR = f"plots_single_run_{SINGLE_RUN_ID_TAG}"
SINGLE_RUN_REPLAY_BUFFER_DIR = f"replay_buffer_single_run_{SINGLE_RUN_ID_TAG}"