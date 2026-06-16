"""Configuration for the debiasing audit experiments."""
import os

SEED = 42
MODEL_NAME = "gpt2"  # 124M params, CPU-feasible
DEVICE = "cpu"
PROJECT_ROOT = "/workspaces/internal-debiasing-audit-af4e"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Probe training
N_PROBE_SENTENCES = 500  # per gender
PROBE_TEST_RATIO = 0.3

# Intervention
TARGET_PROBE_ACCURACY = 0.55  # Target after intervention (near chance for binary)
INLP_MAX_ITERS = 35

# Evaluation
BBQ_MAX_EXAMPLES = 200
CROWS_MAX_EXAMPLES = 200

# Bootstrap
N_BOOTSTRAP = 1000
ALPHA = 0.05
