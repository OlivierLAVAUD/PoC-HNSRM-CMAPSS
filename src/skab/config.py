"""SKAB configuration module.

Contains configuration constants and settings for the SKAB benchmark.
"""

# SKAB dataset configuration
SKAB_DATA_PATH = "data/SKAB"
SKAB_DEFAULT_CONFIG = "configs/config_skab_separate.yaml"

# Feature mapping for SKAB (8 features, different from FD001's 7 features)
SKAB_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "vibration_x",
    "vibration_y",
    "vibration_z",
]

# Model configuration
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 10
DEFAULT_RANDOM_STATE = 42

# Symbolic reasoning configuration
SYMBOLIC_MIN_SUPPORT = 3
SYMBOLIC_MAX_RULES = 50
SYMBOLIC_MIN_CONFIDENCE = 0.7

print("SKAB config loaded")
