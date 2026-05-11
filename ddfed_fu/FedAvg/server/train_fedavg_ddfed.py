import argparse
import copy
import random
import time
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from FedAvg.server.server_fedavg import FedAvgServer
from FedAvg.utils import setup_seed
from utils.fed_utils import get_time


def compute_model_delta(global_model: torch.nn.Module, local_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Compute local update delta = local - global."""
    g_state = global_model.state_dict()
    l_state = local_model.state_dict()
    delta = {}
    for k in g_state.keys():
        delta[k] = (l_state[k].detach().cpu() - g_state[k].detach().cpu()).float()
    return delta


def apply_model_delta(global_model: torch.nn.Module, delta_global: Dict[str, torch.Tensor]) -> None:
    """Apply aggregated delta to global model in-place."""
    state = global_model.state_dict()
    for k in state.keys():
        state[k] = state[k] + delta_global[k].to(state[k].device, dtype=state[k].dtype)
    global_model.load_state_dict(state)


def flatten_model_update(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a model delta dict into one 1D vector."""
    flat_parts = []
    for k in delta.keys():
        flat_parts.append(delta[k].reshape(-1))
    return torch.cat(flat_parts, dim=0)


def unflatten_model_update(vector: torch.Tensor, model_template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Unflatten a 1D vector back to delta dict using template shapes."""
    rebuilt = {}
    cursor = 0
    for k, t in model_template.items():
        n = t.numel()
        rebuilt[k] = vector[cursor: cursor + n].reshape(t.shape)
        cursor += n
    return rebuilt


def _weighted_average_deltas(client_deltas: List[Dict[str, torch.Tensor]], client_weights: List[int]) -> Dict[str, torch.Tensor]:
    """Plaintext weighted average for baseline and sanity comparison."""
    if not client_deltas:
        raise ValueError("client_deltas is empty")
    w = torch.tensor(client_weights, dtype=torch.float32)
    w = w / w.sum()
    keys = list(client_deltas[0].keys())
    out = {}
    for k in keys:
        acc = torch.zeros_like(client_deltas[0][k], dtype=torch.float32)
        for i in range(len(client_deltas)):
            acc += client_deltas[i][k].float() * float(w[i])
        out[k] = acc
    return out


def _secure_scalar_weighted_sum_rodot(
    rodot,
    sk_dict: Dict[int, object],
    dk_j_dict: Dict[int, Dict[int, object]],
    k_dict: Dict[int, int],
    x_dict: Dict[int, int],
    label: str,
    decryptor_ids: List[int],
) -> Tuple[int, float, float, float]:
    """Securely aggregate one scalar coordinate via Rodot+, returning weighted sum."""
    t0 = time.time()
    ct_dict = {i: rodot.enc(sk_dict[i], x_dict[i], label) for i in x_dict.keys()}
    t_enc = time.time() - t0

    t0 = time.time()
    y_dict = {}
    for j in decryptor_ids:
        y_dict[j] = rodot.pardec(dk_j_dict[j], k_dict, ct_dict, label)
    t_pardec = time.time() - t0

    t0 = time.time()
    weighted_sum = int(rodot.comdec(y_dict, k_dict, set(x_dict.keys())))
    t_comdec = time.time() - t0
    return weighted_sum, t_enc, t_pardec, t_comdec


def _pack_digits(digits: List[int], base: int) -> int:
    val = 0
    mul = 1
    for d in digits:
        val += int(d) * mul
        mul *= base
    return val


def _unpack_digits(value: int, base: int, count: int) -> List[int]:
    out = []
    x = int(value)
    for _ in range(count):
        out.append(x % base)
        x //= base
    return out


def secure_aggregate_ddfed(
    client_deltas: List[Dict[str, torch.Tensor]],
    client_weights: List[int],
    ddfed_config: Dict,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Securely aggregate weighted model deltas via DDFed Rodot+.
    Aggregator only receives the final weighted-average delta.
    """
    ddfed_root = Path(ddfed_config["ddfed_project_root"]).resolve()
    if str(ddfed_root) not in sys.path:
        sys.path.append(str(ddfed_root))

    from ddfed_crypto.rodot_plus import RodotPlus  # noqa: WPS433

    quant_scale = int(ddfed_config.get("quantization_scale", 100000))
    lam = int(ddfed_config.get("lambda_sec", 128))
    threshold = int(ddfed_config["threshold"])
    num_decryptors = int(ddfed_config["num_decryptors"])

    if len(client_deltas) < 2:
        raise ValueError("Need at least 2 clients for secure aggregation.")
    if num_decryptors < threshold:
        raise ValueError("num_decryptors must be >= threshold.")

    template = client_deltas[0]
    flat_deltas = [flatten_model_update(d).float() for d in client_deltas]
    vec_len = int(flat_deltas[0].numel())
    for v in flat_deltas[1:]:
        if int(v.numel()) != vec_len:
            raise ValueError("All deltas must have the same flattened length.")

    # Signed encoding by decomposition: x = x_pos - x_neg
    q_vectors = [torch.round(v * quant_scale).to(torch.int64) for v in flat_deltas]
    pos_vectors = [torch.clamp(v, min=0) for v in q_vectors]
    neg_vectors = [torch.clamp(-v, min=0) for v in q_vectors]

    n_clients = len(client_deltas)
    client_ids = list(range(1, n_clients + 1))
    decryptor_ids = list(range(1, num_decryptors + 1))
    k_dict = {i: int(client_weights[i - 1]) for i in client_ids}

    rodot = RodotPlus()
    rodot.setup(lam=lam, n=num_decryptors, t=threshold)

    sk_dict = {i: rodot.kgen(i) for i in client_ids}
    all_dk_shares = {i: rodot.dkshare(sk_dict[i], k_dict[i]) for i in client_ids}
    dk_j_dict = {}
    for j in decryptor_ids:
        shares_for_j = {i: all_dk_shares[i][j] for i in client_ids}
        dk_j_dict[j] = rodot.dkcom(shares_for_j, k_dict)

    sum_weights = float(sum(client_weights))
    agg_vec = torch.zeros(vec_len, dtype=torch.float32)
    t_enc_total = 0.0
    t_pardec_total = 0.0
    t_comdec_total = 0.0

    pack_size = int(ddfed_config.get("secure_pack_size", 16))
    max_plain_bits = int(ddfed_config.get("max_plaintext_bits", 120))
    if pack_size < 1:
        pack_size = 1

    idx = 0
    block_id = 0
    sum_w_int = int(sum(client_weights))
    while idx < vec_len:
        remaining = vec_len - idx
        # Bound per-coordinate digit to avoid carry on weighted sums.
        local_max = 0
        probe_len = min(pack_size, remaining)
        for i in range(len(client_ids)):
            vpos = pos_vectors[i][idx: idx + probe_len]
            vneg = neg_vectors[i][idx: idx + probe_len]
            vmax = int(torch.max(torch.maximum(vpos, vneg)).item())
            if vmax > local_max:
                local_max = vmax
        base = max(2, sum_w_int * local_max + 1)
        # Keep packed plaintext under a conservative bit-length.
        bits_per_digit = max(1.0, math.log2(base))
        k_max = max(1, int(max_plain_bits // bits_per_digit))
        block_len = min(pack_size, remaining, k_max)

        x_pos = {}
        x_neg = {}
        for i in client_ids:
            pos_digits = [int(v.item()) for v in pos_vectors[i - 1][idx: idx + block_len]]
            neg_digits = [int(v.item()) for v in neg_vectors[i - 1][idx: idx + block_len]]
            x_pos[i] = _pack_digits(pos_digits, base)
            x_neg[i] = _pack_digits(neg_digits, base)

        pos_sum, t_enc, t_par, t_com = _secure_scalar_weighted_sum_rodot(
            rodot, sk_dict, dk_j_dict, k_dict, x_pos, f"wpos_blk_{block_id}", decryptor_ids
        )
        t_enc_total += t_enc
        t_pardec_total += t_par
        t_comdec_total += t_com

        neg_sum, t_enc, t_par, t_com = _secure_scalar_weighted_sum_rodot(
            rodot, sk_dict, dk_j_dict, k_dict, x_neg, f"wneg_blk_{block_id}", decryptor_ids
        )
        t_enc_total += t_enc
        t_pardec_total += t_par
        t_comdec_total += t_com

        pos_digits_sum = _unpack_digits(pos_sum, base, block_len)
        neg_digits_sum = _unpack_digits(neg_sum, base, block_len)
        for off in range(block_len):
            weighted_avg = (pos_digits_sum[off] - neg_digits_sum[off]) / (quant_scale * sum_weights)
            agg_vec[idx + off] = float(weighted_avg)

        idx += block_len
        block_id += 1

    delta_global = unflatten_model_update(agg_vec, template)
    timing = {
        "encryption_time": t_enc_total,
        "partial_decryption_time": t_pardec_total,
        "combine_decryption_time": t_comdec_total,
    }
    return delta_global, timing


def _build_server_args(cfg):
    # Reuse existing FedAvgServer setup, keep training logic unchanged.
    args = argparse.Namespace(
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
        # Keep compatibility with existing server options if present.
        secure_agg_backend="none",
        dmcfe_project_root="../../DDFed-main",
        dmcfe_scale=100000,
    )
    return args


def train_fedavg_ddfed(cfg):
    """Train FedAvg baseline or FedAvg+DDFed with identical training setup."""
    setup_seed(cfg.seed)
    if "cuda" in cfg.device and not torch.cuda.is_available():
        print(f"{get_time()} CUDA requested but unavailable, fallback to cpu.")
        cfg.device = "cpu"
    if "cuda" in cfg.device and torch.cuda.is_available():
        print(f"{get_time()} Using GPU device: {cfg.device}")
    server_args = _build_server_args(cfg)
    server = FedAvgServer(server_args)

    users = list(server.users)
    n_total = len(users)
    n_participate = max(1, int(round(n_total * cfg.participation_rate)))

    records = []
    train_start = time.time()
    for r in range(1, cfg.num_rounds + 1):
        round_start = time.time()
        participants = random.sample(users, n_participate) if n_participate < n_total else users
        print(f"{get_time()} Round {r}: participants={participants}")

        global_snapshot = copy.deepcopy(server.global_model)
        comm_start = time.time()
        server.execute_update(participants)
        communication_time = time.time() - comm_start

        client_deltas = []
        client_weights = []
        local_losses = []
        for uid in participants:
            client = server.client_instances[uid]
            client_deltas.append(compute_model_delta(global_snapshot, client.local_model))
            client_weights.append(len(client.trainset))
            if client.last_eval_loss is not None:
                local_losses.append(float(client.last_eval_loss))

        if cfg.method == "fedavg":
            delta_global = _weighted_average_deltas(client_deltas, client_weights)
            enc_t = 0.0
            par_t = 0.0
            com_t = 0.0
        else:
            delta_global, tinfo = secure_aggregate_ddfed(
                client_deltas,
                client_weights,
                {
                    "ddfed_project_root": cfg.ddfed_project_root,
                    "threshold": cfg.threshold,
                    "num_decryptors": cfg.num_decryptors,
                    "quantization_scale": cfg.quantization_scale,
                    "lambda_sec": cfg.lambda_sec,
                    "secure_pack_size": cfg.secure_pack_size,
                    "max_plaintext_bits": cfg.max_plaintext_bits,
                },
            )
            enc_t = tinfo["encryption_time"]
            par_t = tinfo["partial_decryption_time"]
            com_t = tinfo["combine_decryption_time"]

        apply_model_delta(server.global_model, delta_global)
        test_acc, test_loss = server.test_global_model()
        train_loss = float(np.mean(local_losses)) if local_losses else float("nan")
        round_total = time.time() - round_start

        records.append(
            {
                "round": r,
                "method": cfg.method,
                "num_clients": n_total,
                "participation_rate": cfg.participation_rate,
                "train_loss": train_loss,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "communication_time": communication_time,
                "encryption_time": enc_t,
                "partial_decryption_time": par_t,
                "combine_decryption_time": com_t,
                "round_total_time": round_total,
            }
        )
        print(
            f"{get_time()} Round {r} | train_loss={train_loss:.6f} "
            f"test_acc={test_acc:.4f} test_loss={test_loss:.6f} "
            f"time(comm/enc/par/com/total)=({communication_time:.3f}/{enc_t:.3f}/{par_t:.3f}/{com_t:.3f}/{round_total:.3f})"
        )

    total_train_time = time.time() - train_start
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / ("fedavg_baseline.csv" if cfg.method == "fedavg" else "fedavg_ddfed.csv")
    df = pd.DataFrame(records)
    df["total_training_time"] = total_train_time
    df.to_csv(out_csv, index=False)
    print(f"{get_time()} Saved results to: {out_csv}")
    return df, out_csv


def sanity_check_secure_aggregation(cfg):
    """Compare plaintext weighted average vs DDFed secure aggregation."""
    torch.manual_seed(cfg.seed)
    n_clients = 3
    template = {
        "w1": torch.zeros(7, dtype=torch.float32),
        "w2": torch.zeros(5, dtype=torch.float32),
    }
    client_deltas = []
    for _ in range(n_clients):
        d = {
            "w1": torch.randn(7) * 0.01,
            "w2": torch.randn(5) * 0.01,
        }
        client_deltas.append(d)
    client_weights = [5, 3, 2]

    plain = _weighted_average_deltas(client_deltas, client_weights)
    secure, _ = secure_aggregate_ddfed(
        client_deltas,
        client_weights,
        {
            "ddfed_project_root": cfg.ddfed_project_root,
            "threshold": min(cfg.threshold, cfg.num_decryptors),
            "num_decryptors": max(cfg.num_decryptors, 3),
            "quantization_scale": cfg.quantization_scale,
            "lambda_sec": cfg.lambda_sec,
            "secure_pack_size": cfg.secure_pack_size,
            "max_plaintext_bits": cfg.max_plaintext_bits,
        },
    )

    plain_vec = flatten_model_update(plain)
    secure_vec = flatten_model_update(secure)
    max_abs_error = torch.max(torch.abs(plain_vec - secure_vec)).item()
    print(f"{get_time()} sanity_check max_abs_error={max_abs_error:.8f}")
    return max_abs_error


def parse_args():
    parser = argparse.ArgumentParser(description="FedAvg baseline and FedAvg+DDFed trainer")
    parser.add_argument("--method", type=str, default="fedavg", choices=["fedavg", "fedavg_ddfed"])
    parser.add_argument("--num_clients", type=int, default=20, help="Reserved for compatibility; env controls actual clients")
    parser.add_argument("--participation_rate", type=float, default=0.4)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--num_decryptors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    # Existing training/env configs
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_path", type=str, default="../../data")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--env_path", type=str, default="./env")
    parser.add_argument("--strategy", type=str, default="quickdrop-affine")
    parser.add_argument("--env", type=str, default="affine-mnist-seed42-u20-alpha0.1-scale0.01")
    parser.add_argument("--model", type=str, default="ConvNet")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--persistent_workers", type=bool, default=False)

    # DDFed adapter configs
    parser.add_argument("--ddfed_project_root", type=str, default="../../DDFed-main")
    parser.add_argument("--lambda_sec", type=int, default=128)
    parser.add_argument("--quantization_scale", type=int, default=100000)
    parser.add_argument("--secure_pack_size", type=int, default=1, help="number of coordinates packed per secure ciphertext (1 is numerically safest)")
    parser.add_argument("--max_plaintext_bits", type=int, default=120, help="upper bound for packed plaintext bit-length")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_sanity_check", action="store_true")
    return parser.parse_args()


def main():
    cfg = parse_args()
    if cfg.run_sanity_check:
        sanity_check_secure_aggregation(cfg)
    train_fedavg_ddfed(cfg)


if __name__ == "__main__":
    main()
