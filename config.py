# -------------------------------
# Matrix configuration
# -------------------------------
MATRIX_SIZE = 5

# -------------------------------
# FEM Material properties
# -------------------------------
E_STIFF = 1818.0   # (MPa)
NU_STIFF = 0.33    # Poisson's ratio for stiff material
E_COMP = 364.0     # (MPa)
NU_COMP = 0.49     # Poisson's ratio for compliant material

# -------------------------------
# Target Material properties
# -------------------------------
DESIRED_MODULUS = 1302.17  # (MPa)
DESIRED_VOL_FRAC = 0.20    # Target volume fraction (compliant material ratio)

# -------------------------------
# DQN Training Parameters
# -------------------------------
LEARNING_RATE = 1e-3
BATCH_SIZE = 320
GAMMA = 0.99
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05

# -------------------------------
# Q-Network architecture parameters
# -------------------------------
FIRST_HIDDEN_NEURONS = 128
SECOND_HIDDEN_NEURONS = 64

# -------------------------------
# Training Loop Parameters
# -------------------------------
NUM_CYCLES = 750
EPISODES_PER_CYCLE = 50
MAX_STEPS = 25
OPT_STEPS_PER_CYCLE = 500
