"""Configuration settings for DFL tool"""

# Default simulation parameters
DEFAULT_NUM_PEERS = 5
DEFAULT_HOPS = {1}  # Ring with 1-hop neighbors
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cpu"
DEFAULT_LATENCY_MS = 0  # No latency by default
DEFAULT_DROP_PROB = 0.0  # No message drop by default
DEFAULT_DATASET = "bearing"  # Default to bearing dataset

# Model parameters - will be set dynamically based on dataset
MODEL_INPUT_DIM = 8  # Bearing dataset features (8 columns)
MODEL_HIDDEN_DIM = 64
MODEL_OUTPUT_DIM = 4  # Will be set based on number of classes in dataset

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TIMEOUT = 30  # seconds

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "dfl_tool.log"

# Data distribution types
DISTRIBUTION_IID = "iid"
DISTRIBUTION_NON_IID = "non_iid"
DISTRIBUTION_LABEL_SKEW = "label_skew"

# Aggregation methods
AGGREGATE_AVG = "FedAvg"
AGGREGATE_FEDPROX = "FedProx"
