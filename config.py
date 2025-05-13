# project_root/config.py

# -------------------------------
# Matrix configuration
# -------------------------------
MATRIX_SIZE = 8

# -------------------------------
# FEM Material properties
# -------------------------------
E_STIFF = 1818.0
NU_STIFF = 0.33
E_COMP = 364.0
NU_COMP = 0.49

# -------------------------------
# Target Material properties
# -------------------------------
DESIRED_MODULUS = 1302.17
DESIRED_VOL_FRAC = 0.20

# -------------------------------
# DQN Training Parameters
# -------------------------------
LEARNING_RATE = 1e-4
TAU = 0.01
BATCH_SIZE = 1024 # Smaller batch might be needed if CHW states are large in GPU memory
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.99995
EPSILON_MIN = 0.02

# -------------------------------
# Q-Network architecture parameters
# -------------------------------
FCN_INPUT_CHANNELS = 4 # 1 (grid) + 3 (broadcasted: current_VF, scaled_step, scaled_dist_E)
FCN_NUM_FILTERS_RESBLOCK = 8 # Default number of filters in ResBlocks (can be made tunable)
FCN_NUM_RES_BLOCKS = 4 # Default number of ResBlocks (will be tuned by Optuna)
FCN_KERNEL_SIZE = 3 # Default kernel size for ResBlocks (will be tuned by Optuna)

# Define dilation strategy choices for Optuna
FCN_DILATION_STRATEGIES = ["all_ones", "progressive_trim", "cyclic_124"]
# Max number of blocks for which 'progressive_trim' is designed
FCN_DILATION_PROGRESSIVE_MAX_BLOCKS = 12
FCN_DILATION_PROGRESSIVE_PATTERN = [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4] # For up to 12 blocks
FCN_DILATION_CYCLIC_PATTERN = [1, 2, 4]

# -------------------------------
# Training Loop Parameters
# -------------------------------
NUM_CYCLES = 1000
EPISODES_PER_CYCLE = 100
MAX_STEPS = int(MATRIX_SIZE * MATRIX_SIZE * 0.5)
OPT_STEPS_PER_CYCLE = 500

# -------------------------------
# Replay Buffer Parameters
# -------------------------------
REPLAY_BUFFER_CAPACITY = 4_000_000 # Adjusted, CHW might take more if not careful, though grid is uint8
REPLAY_BUFFER_DIR = "replay_buffer_data_fcn" # New dir to avoid conflicts

# -------------------------------
# Optuna / Trainer Parameters
# -------------------------------
CHECKPOINT_DIR = "checkpoints_fcn"
TRIAL_PLOTS_DIR = "trial_plots_fcn"
RESULTS_FILENAME = "trial_results_fcn.jsonl"
TOTAL_TARGET_EPISODES = 20_000