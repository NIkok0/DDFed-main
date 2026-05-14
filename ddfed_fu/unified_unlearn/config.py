"""Unified configuration for Federated Unlearning experiments."""

import os
import torch

# ── Dataset / Scenarios ──────────────────────────────────────────
ALPHA = 0.1  # default Dirichlet alpha, overridden by CLI

# ── Algorithm schemes ────────────────────────────────────────────
# Each scheme maps to a client-mode string used by the runner.
ALGORITHMS = {
    "Baseline_FedSGA":      "fedsga",
    "Baseline_Neurotoxin":  "neurotoxin",
    "Baseline_FedQuickDrop": "quickdrop",
    "Proposed_ddfu_client":  "ddfu_client",
    "Proposed_ddfu_sample":  "ddfu_sample",
}

# ── FL hyper-parameters ──────────────────────────────────────────
NUM_CLIENTS        = 20
SELECTED_PER_ROUND = 4
SELECTED_CLIENTS   = SELECTED_PER_ROUND
FRAC               = SELECTED_PER_ROUND / NUM_CLIENTS  # 0.2
LR                 = 0.01
MOMENTUM           = 0.5
BATCH_SIZE         = 64
LOCAL_EPOCHS       = 1
PRETRAIN_ROUNDS    = 30          # round at which forgetting phase begins
UNLEARN_ROUNDS     = 20          # rounds of forgetting (phase 2)
UNLEARN_TARGET_CLASS = 0        # target class for backdoor_loss construction

# ── Model ────────────────────────────────────────────────────────
MODEL_NAME   = "ConvNet"
NET_WIDTH    = 64
NET_DEPTH    = 3

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(os.path.dirname(BASE_DIR), "data")
RESULT_DIR   = os.path.join(os.path.dirname(BASE_DIR), "result", "unified_unlearn")
CKPT_DIR     = os.path.join(os.path.dirname(BASE_DIR), "save",   "unified_unlearn")
OUTPUT_DIR   = RESULT_DIR

os.makedirs(DATA_ROOT,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

# ── Neurotoxin coefficients ──────────────────────────────────────
NT_ALPHA = 1.0   # weight for total_loss (clean + gamma * backdoor)
GAMMA = 1.0   # weight for backdoor loss
BETA  = 0.1   # weight for gradient penalty

# ── Device ───────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# ── Global Seed ───────────────────────────────────────────────────
GLOBAL_SEED = 42

# ── Log ──────────────────────────────────────────────────────────
# CSV columns per round:
#   round, phase, test_acc, test_loss, forget_acc, forget_loss,
#   remain_acc, remain_loss
LOG_COLUMNS = [
    "round", "phase",
    "test_acc", "test_loss",
    "forget_acc", "forget_loss",
    "remain_acc", "remain_loss",
]


def scenario_id(dataset: str, alpha: float) -> str:
    """Short identifier for a scenario, e.g. fmnist-a100 or cifar10-a0.1"""
    short = "fmnist" if dataset == "FashionMNIST" else "cifar10"
    return f"{short}-a{str(alpha).replace('.','_')}"


def result_csv_path(dataset: str, alpha: float, algo: str) -> str:
    return os.path.join(RESULT_DIR, f"{scenario_id(dataset, alpha)}_{algo}.csv")


def ckpt_path(dataset: str, alpha: float) -> str:
    """Path to save/load Phase-1 checkpoint (shared across algorithms)."""
    return os.path.join(CKPT_DIR, f"{scenario_id(dataset, alpha)}_phase1.pt")