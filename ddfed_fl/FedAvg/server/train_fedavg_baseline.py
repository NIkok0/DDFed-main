"""
Standalone FedAvg baseline script (plaintext aggregation).

No dependency on any secure aggregation library (TMCFE / DDFed / Lepcat).
Produces ``results/fedavg_baseline.csv`` with per-round metrics.

Usage::

    python -m FedAvg.server.train_fedavg_baseline \
        --device cuda:0 \
        --env_path "./env" \
        --env affine-fashionmnist-seed42-u20-alpha0.1-scale0.01 \
        --num_rounds 5 --local_epochs 1 --batch_size 64 \
        --participation_rate 0.2 --seed 0
"""

import argparse
import copy
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
FL_ROOT = CURRENT_DIR.parent.parent
if str(FL_ROOT) not in sys.path:
    sys.path.append(str(FL_ROOT))

from FedAvg.server.server_fedavg import FedAvgServer
from FedAvg.utils import setup_seed
from utils.fed_utils import get_time


# ---------------------------------------------------------------------------
# Pure plaintext operations (no crypto imports)
# ---------------------------------------------------------------------------

def compute_model_delta(
    global_model: torch.nn.Module,
    local_model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """Compute local update delta = local - global."""
    g_state = global_model.state_dict()
    l_state = local_model.state_dict()
    delta: Dict[str, torch.Tensor] = {}
    for key in g_state:
        delta[key] = (l_state[key].detach().cpu() - g_state[key].detach().cpu()).float()
    return delta


def apply_model_delta(
    global_model: torch.nn.Module,
    delta_global: Dict[str, torch.Tensor],
) -> None:
    """Apply aggregated delta to global model in-place."""
    state = global_model.state_dict()
    for key in state:
        state[key] = state[key] + delta_global[key].to(
            state[key].device, dtype=state[key].dtype
        )
    global_model.load_state_dict(state)


def plaintext_fedavg_aggregate(
    client_deltas: Dict[int, Dict[str, torch.Tensor]],
    client_weights: Dict[int, int],
) -> Dict[str, torch.Tensor]:
    """FedAvg weighted average in plaintext."""
    if not client_deltas:
        raise ValueError("client_deltas is empty")
    active_clients = list(client_deltas.keys())
    total = float(sum(client_weights[uid] for uid in active_clients))
    if total <= 0:
        raise ValueError("Sum of client weights must be positive")

    template = client_deltas[active_clients[0]]
    out = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in template.items()}
    for uid in active_clients:
        weight = float(client_weights[uid]) / total
        for key in out:
            out[key] += client_deltas[uid][key].float() * weight
    return out


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def str2bool(value):
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"true", "1", "yes", "y", "t"}:
        return True
    if val in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FedAvg Baseline (plaintext aggregation, no crypto deps)"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--env_path", type=str, default="./env")
    parser.add_argument(
        "--env",
        type=str,
        default="affine-mnist-seed42-u20-alpha0.1-scale0.01",
    )
    parser.add_argument("--data_path", type=str, default="../../data")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--strategy", type=str, default="pretrained_affine")
    parser.add_argument("--model", type=str, default="ConvNet")
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--participation_rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=str2bool, default=False)
    parser.add_argument("--persistent_workers", type=str2bool, default=False)
    parser.add_argument("--results_dir", type=str, default="outputs/results")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Server args builder
# ---------------------------------------------------------------------------

def _build_server_args(cfg):
    return argparse.Namespace(
        device=cfg.device,
        data_path=cfg.data_path,
        dataset=cfg.dataset,
        env_path=cfg.env_path,
        strategy=cfg.strategy,
        env=cfg.env,
        model=cfg.model,
        communication_round=cfg.num_rounds,
        local_epoch=cfg.local_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        secure_agg_backend="none",
        dmcfe_project_root="",
        dmcfe_scale=100000,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_fedavg_baseline(cfg):
    """Run FedAvg baseline with plaintext aggregation."""
    setup_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if "cuda" in cfg.device and not torch.cuda.is_available():
        print(f"{get_time()} CUDA unavailable, fallback to cpu")
        cfg.device = "cpu"

    server = FedAvgServer(_build_server_args(cfg))
    users = sorted(list(server.users))
    n_total = len(users)
    n_participate = max(1, int(round(n_total * float(cfg.participation_rate))))

    records = []
    for r in range(1, int(cfg.num_rounds) + 1):
        round_start = time.time()

        participants = (
            random.sample(users, n_participate) if n_participate < n_total else users[:]
        )
        participants = sorted(participants)

        print(
            f"{get_time()} Round {r}: {len(participants)} participants / {n_total} total"
        )

        # Snapshot global model before local training
        global_snapshot = copy.deepcopy(server.global_model)

        # Execute local training on participants
        server.execute_update(participants)

        # Compute per-client deltas and weights
        client_deltas: Dict[int, Dict[str, torch.Tensor]] = {}
        client_weights: Dict[int, int] = {}
        local_losses = []
        for uid in participants:
            client_obj = server.client_instances[uid]
            client_deltas[uid] = compute_model_delta(
                global_snapshot, client_obj.local_model
            )
            client_weights[uid] = len(client_obj.trainset)
            if client_obj.last_eval_loss is not None:
                local_losses.append(float(client_obj.last_eval_loss))

        # Plaintext weighted average aggregation
        delta_global = plaintext_fedavg_aggregate(client_deltas, client_weights)

        # Apply aggregated delta to global model
        apply_model_delta(server.global_model, delta_global)

        # Evaluate
        test_acc, test_loss = server.test_global_model()
        train_loss = float(np.mean(local_losses)) if local_losses else float("nan")
        round_time = time.time() - round_start

        records.append(
            {
                "round": r,
                "train_loss": train_loss,
                "test_accuracy": float(test_acc),
                "test_loss": float(test_loss),
                "round_time": round_time,
            }
        )
        print(
            f"{get_time()} round={r} "
            f"acc={test_acc:.4f} loss={test_loss:.6f} "
            f"time={round_time:.2f}s"
        )

    # Save results
    df = pd.DataFrame(records)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "fedavg_baseline.csv"
    df.to_csv(out_path, index=False)
    print(f"{get_time()} Results saved to {out_path}")
    return df, out_path


def main():
    cfg = parse_args()
    train_fedavg_baseline(cfg)


if __name__ == "__main__":
    main()