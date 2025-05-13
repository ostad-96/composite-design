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
BATCH_SIZE = 1024
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.99995
EPSILON_MIN = 0.02
CLIP_GRAD_NORM_MAX = 1.0 # <<< Max norm for gradient clipping

# -------------------------------
# Q-Network architecture parameters
# -------------------------------
FCN_INPUT_CHANNELS = 4 # 1 (grid) + 3 (broadcasted: current_VF, scaled_step, scaled_dist_E)
FCN_NUM_FILTERS_RESBLOCK = 8
FCN_NUM_RES_BLOCKS = 4
FCN_KERNEL_SIZE = 3

FCN_DILATION_STRATEGIES = ["all_ones", "progressive_trim", "cyclic_124"]
FCN_DILATION_PROGRESSIVE_MAX_BLOCKS = 12
FCN_DILATION_PROGRESSIVE_PATTERN = [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4]
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
REPLAY_BUFFER_CAPACITY = 4_000_000
REPLAY_BUFFER_DIR = "replay_buffer_data_fcn"

# -------------------------------
# Optuna / Trainer Parameters
# -------------------------------
CHECKPOINT_DIR = "checkpoints_fcn"
TRIAL_PLOTS_DIR = "trial_plots_fcn"
RESULTS_FILENAME = "trial_results_fcn.jsonl"
TOTAL_TARGET_EPISODES = 20_000