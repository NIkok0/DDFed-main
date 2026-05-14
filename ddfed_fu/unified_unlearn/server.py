"""Unified federated unlearning server.

Phase 1 (rounds 0 .. PRETRAIN_ROUNDS-1): standard FedAvg.
Phase 2 (rounds PRETRAIN_ROUNDS .. PRETRAIN_ROUNDS+UNLEARN_ROUNDS-1):
  - One designated *forget client* runs the unlearn algorithm.
  - The other selected clients perform normal local training.
  - Server aggregates all updates via FedAvg.
"""

import copy
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Allow importing from parent package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import (
    DEVICE, PRETRAIN_ROUNDS, UNLEARN_ROUNDS, LR, MOMENTUM,
    NUM_CLIENTS, SELECTED_PER_ROUND, BATCH_SIZE, LOCAL_EPOCHS,
    CKPT_DIR, GLOBAL_SEED, UNLEARN_TARGET_CLASS,
)
from .model import create_model
from .dataset import get_dataloaders, split_by_class
from .client import run_client, evaluate
from .log_utils import Logger


# ────────────────────────────────────────────────────────────────
#  Utilities
# ────────────────────────────────────────────────────────────────

def fedavg_aggregate(local_models: list, global_model: nn.Module):
    """Simple FedAvg: average parameters across local_models."""
    gsd = global_model.state_dict()
    keys = list(gsd.keys())
    for k in keys:
        gsd[k] = sum(m.state_dict()[k].float() for m in local_models) / len(local_models)
    global_model.load_state_dict(gsd)


# ────────────────────────────────────────────────────────────────
#  Phase 1 – Standard FedAvg Pretraining
# ────────────────────────────────────────────────────────────────

def pretrain_phase(global_model: nn.Module,
                   user_loaders: list,
                   test_loader: DataLoader,
                   num_clients: int = NUM_CLIENTS,
                   num_selected: int = SELECTED_PER_ROUND,
                   rounds: int = PRETRAIN_ROUNDS,
                   logger: Logger = None) -> nn.Module:
    """Standard FedAvg for *rounds* rounds. Logs test accuracy."""
    global_model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    for rnd in range(rounds):
        selected = torch.randperm(num_clients, generator=torch.Generator().manual_seed(GLOBAL_SEED + rnd + 1000))[:num_selected]
        local_models = []

        for uid in selected.tolist():
            m = copy.deepcopy(global_model).to(DEVICE)
            m.train()
            opt = optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM)
            for _ in range(LOCAL_EPOCHS):
                for x, y in user_loaders[uid]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    loss = criterion(m(x), y)
                    loss.backward()
                    opt.step()
            local_models.append(m)

        fedavg_aggregate(local_models, global_model)

        # evaluate
        acc, loss = evaluate(global_model, test_loader)
        if logger:
            logger.log_round(rnd, "pretrain", acc, loss, 0, 0, 0, 0,
                             cur=rnd + 1, total=rounds)

    return global_model


# ────────────────────────────────────────────────────────────────
#  Phase 2 – Federated Unlearning
# ────────────────────────────────────────────────────────────────

def unlearn_phase(global_model: nn.Module,
                  user_loaders: list,
                  user_datasets: list,
                  test_loader: DataLoader,
                  forget_uid: int,
                  algo: str,
                  num_clients: int = NUM_CLIENTS,
                  num_selected: int = SELECTED_PER_ROUND,
                  rounds: int = UNLEARN_ROUNDS,
                  logger: Logger = None,
                  dsa: bool = True) -> nn.Module:
    """
    Phase 2: Federated unlearning.

    Each round:
      1. Select *num_selected* normal users + always include *forget_uid*.
      2. For the forget user: run the specified unlearn algorithm.
      3. For each normal user: standard local FedAvg training.
      4. FedAvg aggregate all (num_selected + 1) models.
    """
    global_model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Build unlearn-set for fedsga (single-class loader)
    unlearn_loader = None
    if algo == "fedsga":
        unlearn_ds = split_by_class(user_datasets[forget_uid], UNLEARN_TARGET_CLASS,
                                     num_classes=10)
        if unlearn_ds is not None:
            unlearn_loader = DataLoader(unlearn_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Fixed random sequence for reproducibility across algorithms
    # (but forget_uid is always included)
    total_selected = num_selected  # normal clients per round (forget is extra)

    for rnd in range(rounds):
        g = torch.Generator().manual_seed(GLOBAL_SEED + PRETRAIN_ROUNDS + rnd + 2000)
        # pick normal users excluding forget_uid
        candidates = [u for u in range(num_clients) if u != forget_uid]
        perm = torch.randperm(len(candidates), generator=g)[:total_selected]
        normal_uids = [candidates[i] for i in perm.tolist()]

        local_models = []

        # ── 1. Unlearn client ──
        fm = run_client(algo, global_model, user_loaders[forget_uid],
                         unlearn_loader=unlearn_loader,
                         num_classes=10, epochs=LOCAL_EPOCHS, lr=LR, use_dsa=dsa)
        local_models.append(fm)

        # ── 2. Normal clients ──
        for uid in normal_uids:
            m = copy.deepcopy(global_model).to(DEVICE)
            m.train()
            opt = optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM)
            for _ in range(LOCAL_EPOCHS):
                for x, y in user_loaders[uid]:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    loss = criterion(m(x), y)
                    loss.backward()
                    opt.step()
            local_models.append(m)

        # ── 3. Aggregate ──
        fedavg_aggregate(local_models, global_model)

        # ── 4. Logging ──
        test_acc, test_loss = evaluate(global_model, test_loader)
        # forget accuracy on forget user's data
        forget_acc, forget_loss = evaluate(global_model, user_loaders[forget_uid])
        # remain accuracy on a held-out normal user (use uid=0 unless it's the forget user)
        remain_uid = 1 if forget_uid == 0 else 0
        remain_acc, remain_loss = evaluate(global_model, user_loaders[remain_uid])

        rnd_offset = PRETRAIN_ROUNDS + rnd
        if logger:
            logger.log_round(rnd_offset, algo, test_acc, test_loss,
                             forget_acc, forget_loss, remain_acc, remain_loss,
                             cur=rnd + 1, total=rounds)

    return global_model


# ────────────────────────────────────────────────────────────────
#  Full Server Runner
# ────────────────────────────────────────────────────────────────

def run_server(dataset: str, alpha: float, algo: str,
               num_clients: int = NUM_CLIENTS,
               pretrain_rounds: int = PRETRAIN_ROUNDS,
               unlearn_rounds: int = UNLEARN_ROUNDS,
               dsa: bool = True,
               resume_phase1: bool = True):
    """
    Full two-phase server execution.

    Parameters
    ----------
    resume_phase1 : bool
        If True, load a previously saved Phase‑1 checkpoint (sharing across algos).
        If False, run Phase‑1 from scratch and save the checkpoint.
    """
    from .config import ckpt_path, result_csv_path, NUM_CLIENTS

    # ── Data ──
    user_loaders, user_datasets, test_loader = get_dataloaders(
        dataset=dataset, num_clients=num_clients, alpha=alpha,
        batch_size=BATCH_SIZE, seed=GLOBAL_SEED,
    )

    # ── Model ──
    global_model = create_model(dataset, num_classes=10)

    # ── Logger ──
    csv_file = result_csv_path(dataset, alpha, algo)
    logger = Logger(csv_file)

    # ── Phase 1 ──
    ckpt_f = ckpt_path(dataset, alpha)
    if resume_phase1 and os.path.exists(ckpt_f):
        global_model.load_state_dict(torch.load(ckpt_f, map_location=DEVICE))
    else:
        global_model = pretrain_phase(global_model, user_loaders, test_loader,
                                       num_clients=NUM_CLIENTS,
                                       rounds=pretrain_rounds,
                                       logger=logger)
        torch.save(global_model.state_dict(), ckpt_f)

    # ── Determine forget user (fixed seed, same across algorithms) ──
    g_fid = torch.Generator().manual_seed(GLOBAL_SEED)
    forget_uid = torch.randint(0, NUM_CLIENTS, (1,), generator=g_fid).item()

    # ── Phase 2 ──
    global_model = unlearn_phase(
        global_model, user_loaders, user_datasets, test_loader,
        forget_uid=forget_uid, algo=algo,
        rounds=unlearn_rounds, logger=logger, dsa=dsa,
    )

    # ── Close logger ──
    logger.close()

    return global_model, forget_uid