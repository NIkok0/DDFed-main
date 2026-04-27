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
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from FedAvg.server.server_fedavg import FedAvgServer
from FedAvg.server.secure_packing import (
    choose_block_len,
    compute_slot_bits,
    compute_safe_base,
    decode_slot_to_signed,
    encode_signed_to_slot,
    estimate_block_qmax,
    pack_digits,
    pack_client_update_vector,
    pack_plaintexts,
    unpack_digits,
    unpack_aggregated_vector,
    unpack_plaintext,
)
from FedAvg.utils import setup_seed
from utils.fed_utils import get_time


def str2bool(value):
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"true", "1", "yes", "y", "t"}:
        return True
    if val in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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


def _build_block_label(label: str, block_id: int) -> str:
    return f"{label}|blk={int(block_id)}"


def secure_aggregate_packed(
    client_deltas: List[Dict[str, torch.Tensor]],
    client_weights: List[int],
    active_clients: List[int],
    active_decryptors: List[int],
    label: str,
    crypto_ctx: Dict,
    pack_size: int,
    slot_bits: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Packed secure aggregation using integer-weight FedAvg with slot encoding.

    Decrypted value per slot is: sum_i n_i * encoded_i.
    Then recover q_sum = decoded - offset * sum_i n_i.
    """
    template = client_deltas[0]
    quant_scale = int(crypto_ctx["quantization_scale"])
    max_plain_bits = int(crypto_ctx["max_plaintext_bits"])
    packing_value_bits = int(crypto_ctx["packing_value_bits"])
    sum_w_int = int(sum(client_weights))
    if sum_w_int <= 0:
        raise ValueError("sum of client weights must be positive")

    flat_deltas = [flatten_model_update(d).float() for d in client_deltas]
    vec_len = int(flat_deltas[0].numel())
    for vec in flat_deltas[1:]:
        if int(vec.numel()) != vec_len:
            raise ValueError("All deltas must share the same flattened length")

    q_vectors = [torch.round(v * quant_scale).to(torch.int64) for v in flat_deltas]
    offset = 1 << (max(2, int(packing_value_bits)) - 1)
    slot_upper = 1 << int(slot_bits)

    max_slots_by_bits = max(1, int(max_plain_bits) // max(1, int(slot_bits)))
    effective_pack_size = max(1, min(int(pack_size), int(max_slots_by_bits)))
    packed_vectors = []
    packing_overflow = False
    encoded_vectors = []
    for q_vec in q_vectors:
        blocks, overflow = pack_client_update_vector(q_vec, int(effective_pack_size), int(slot_bits), int(offset))
        packed_vectors.append(blocks)
        packing_overflow = packing_overflow or bool(overflow)
        encoded_vectors.append(q_vec + int(offset))
    # Cross-slot carry protection for weighted aggregated slot sums.
    for idx in range(vec_len):
        weighted_encoded_sum = 0
        for i, w in enumerate(client_weights):
            weighted_encoded_sum += int(w) * int(encoded_vectors[i][idx].item())
        if weighted_encoded_sum >= slot_upper:
            packing_overflow = True
            break
    if packing_overflow:
        raise OverflowError(
            "Packing overflow detected. Increase --packing_value_bits or decrease --quantization_scale."
        )

    n_blocks = len(packed_vectors[0])
    for blocks in packed_vectors[1:]:
        if len(blocks) != n_blocks:
            raise ValueError("Packed block length mismatch across clients")

    rodot = crypto_ctx["rodot"]
    sk_dict = crypto_ctx["sk_dict"]
    dk_j_dict = crypto_ctx["dk_j_dict"]
    k_dict = crypto_ctx["k_dict"]

    # Optional safety cap from bit budget.
    max_packed_bits = max(1, int(max_plain_bits))

    packed_sums = []
    t_enc_total = 0.0
    t_pardec_total = 0.0
    t_comdec_total = 0.0
    max_abs_update = 0

    for block_id in range(n_blocks):
        x_block = {}
        for i in active_clients:
            packed_val = int(packed_vectors[i - 1][block_id])
            if packed_val.bit_length() > max_packed_bits:
                raise OverflowError(
                    f"Packed plaintext bit-length overflow at block {block_id}: "
                    f"{packed_val.bit_length()} > {max_packed_bits}"
                )
            x_block[i] = packed_val
        weighted_sum, t_enc, t_par, t_com = _secure_scalar_weighted_sum_rodot(
            rodot,
            sk_dict,
            dk_j_dict,
            k_dict,
            x_block,
            _build_block_label(label, block_id),
            active_decryptors,
        )
        packed_sums.append(int(weighted_sum))
        t_enc_total += t_enc
        t_pardec_total += t_par
        t_comdec_total += t_com

    decoded_qsum = unpack_aggregated_vector(
        packed_values=packed_sums,
        original_length=vec_len,
        pack_size=int(effective_pack_size),
        slot_bits=int(slot_bits),
        offset=0,  # here values are encoded sums, subtract offset*sum_w later
    )
    q_avg = torch.zeros(vec_len, dtype=torch.float32)
    upper = slot_upper
    for idx, encoded_sum in enumerate(decoded_qsum):
        # encoded_sum is slot residue after unpack, must be in slot range.
        if encoded_sum < 0 or encoded_sum >= upper:
            raise OverflowError(
                f"Aggregated slot overflow at idx={idx}, value={encoded_sum}, slot_bits={slot_bits}"
            )
        q_sum = int(encoded_sum) - int(offset * sum_w_int)
        max_abs_update = max(max_abs_update, abs(q_sum))
        q_avg[idx] = float(q_sum) / float(sum_w_int)

    agg_vec = q_avg / float(quant_scale)
    delta_global = unflatten_model_update(agg_vec, template)
    n_params = int(vec_len)
    n_ciphertexts = int(n_blocks) * int(len(active_clients))
    timing = {
        "encryption_time": t_enc_total,
        "partial_decryption_time": t_pardec_total,
        "combine_decryption_time": t_comdec_total,
        "slot_bits": int(slot_bits),
        "secure_pack_size": int(pack_size),
        "effective_pack_size": int(effective_pack_size),
        "pack_block_count": int(n_blocks),
        "pack_avg_size": float(vec_len) / float(max(1, n_blocks)),
        "packing_overflow": False,
        "num_model_params": n_params,
        "num_ciphertexts": n_ciphertexts,
        "compression_ratio": float(n_params) / float(max(1, n_blocks)),
        "pack_max_abs_qsum": int(max_abs_update),
    }
    return delta_global, timing


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
    pack_size = max(1, int(ddfed_config.get("secure_pack_size", 1)))
    packing_value_bits = int(ddfed_config.get("packing_value_bits", 32))
    force_packed_mode = bool(ddfed_config.get("force_packed_mode", False))

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

    slot_bits, padding_bits = compute_slot_bits(
        num_clients=n_clients,
        quantization_scale=quant_scale,
        packing_value_bits=packing_value_bits,
    )
    # Integer-weight aggregation enlarges slot sums; reserve extra bits for sum(weights).
    sum_w_int = int(sum(client_weights))
    slot_bits += int(math.ceil(math.log2(max(2, sum_w_int + 1))))
    if pack_size > 1 or force_packed_mode:
        packed_delta, packed_timing = secure_aggregate_packed(
            client_deltas=client_deltas,
            client_weights=client_weights,
            active_clients=client_ids,
            active_decryptors=decryptor_ids,
            label=ddfed_config.get("label_base", "fedavg_ddfed_round"),
            crypto_ctx={
                "rodot": rodot,
                "sk_dict": sk_dict,
                "dk_j_dict": dk_j_dict,
                "k_dict": k_dict,
                "quantization_scale": quant_scale,
                "max_plaintext_bits": ddfed_config.get("max_plaintext_bits", 120),
                "packing_value_bits": packing_value_bits,
            },
            pack_size=pack_size,
            slot_bits=slot_bits,
        )
        packed_timing["packing_value_bits"] = int(packing_value_bits)
        packed_timing["padding_bits"] = int(padding_bits)
        return packed_delta, packed_timing

    sum_weights = float(sum(client_weights))
    agg_vec = torch.zeros(vec_len, dtype=torch.float32)
    t_enc_total = 0.0
    t_pardec_total = 0.0
    t_comdec_total = 0.0
    block_count = 0
    packed_coords_total = 0
    max_base = 2
    max_qmax = 0
    skipped_zero_blocks = 0

    pack_size = int(ddfed_config.get("secure_pack_size", 1))
    max_plain_bits = int(ddfed_config.get("max_plaintext_bits", 120))
    skip_zero_blocks = bool(ddfed_config.get("skip_zero_blocks", True))
    if pack_size < 1:
        pack_size = 1

    idx = 0
    block_id = 0
    sum_w_int = int(sum(client_weights))
    while idx < vec_len:
        remaining = vec_len - idx
        # Bound per-coordinate digit to avoid carry on weighted sums.
        probe_len = min(pack_size, remaining)
        local_max = estimate_block_qmax(pos_vectors + neg_vectors, idx, probe_len)
        base = compute_safe_base(sum_w_int, local_max, margin=1)
        max_base = max(max_base, int(base))
        max_qmax = max(max_qmax, int(local_max))
        block_len = choose_block_len(
            requested_pack_size=pack_size,
            remaining=remaining,
            base=base,
            max_plain_bits=max_plain_bits,
        )
        if skip_zero_blocks and local_max == 0:
            idx += block_len
            block_id += 1
            block_count += 1
            packed_coords_total += int(block_len)
            skipped_zero_blocks += 1
            continue

        x_pos = {}
        x_neg = {}
        for i in client_ids:
            pos_digits = [int(v.item()) for v in pos_vectors[i - 1][idx: idx + block_len]]
            neg_digits = [int(v.item()) for v in neg_vectors[i - 1][idx: idx + block_len]]
            x_pos[i] = pack_digits(pos_digits, base)
            x_neg[i] = pack_digits(neg_digits, base)

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

        pos_digits_sum = unpack_digits(pos_sum, base, block_len)
        neg_digits_sum = unpack_digits(neg_sum, base, block_len)
        for off in range(block_len):
            weighted_avg = (pos_digits_sum[off] - neg_digits_sum[off]) / (quant_scale * sum_weights)
            agg_vec[idx + off] = float(weighted_avg)

        idx += block_len
        block_id += 1
        block_count += 1
        packed_coords_total += int(block_len)

    delta_global = unflatten_model_update(agg_vec, template)
    avg_pack_size = float(packed_coords_total) / float(max(1, block_count))
    timing = {
        "encryption_time": t_enc_total,
        "partial_decryption_time": t_pardec_total,
        "combine_decryption_time": t_comdec_total,
        "pack_block_count": block_count,
        "pack_avg_size": avg_pack_size,
        "pack_max_base": max_base,
        "pack_max_qmax": max_qmax,
        "pack_skipped_zero_blocks": skipped_zero_blocks,
        "slot_bits": int(slot_bits),
        "packing_value_bits": int(packing_value_bits),
        "padding_bits": int(padding_bits),
        "packing_overflow": False,
        "num_model_params": int(vec_len),
        "num_ciphertexts": int(block_count) * int(len(client_ids)),
        "compression_ratio": float(vec_len) / float(max(1, block_count)),
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
            plain_delta = delta_global
            enc_t = 0.0
            par_t = 0.0
            com_t = 0.0
            total_crypto_t = 0.0
            pack_block_count = 0
            pack_avg_size = 1.0
            pack_max_base = 0
            pack_max_qmax = 0
            pack_skipped_zero_blocks = 0
            slot_bits = 0
            packing_value_bits = int(cfg.packing_value_bits)
            padding_bits = 0
            packing_overflow = False
            num_model_params = int(flatten_model_update(client_deltas[0]).numel()) if client_deltas else 0
            num_ciphertexts = 0
            compression_ratio = 1.0
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
                    "packing_value_bits": cfg.packing_value_bits,
                    "skip_zero_blocks": cfg.skip_zero_blocks,
                    "label_base": f"round_{r}",
                },
            )
            enc_t = tinfo["encryption_time"]
            par_t = tinfo["partial_decryption_time"]
            com_t = tinfo["combine_decryption_time"]
            total_crypto_t = enc_t + par_t + com_t
            pack_block_count = int(tinfo.get("pack_block_count", 0))
            pack_avg_size = float(tinfo.get("pack_avg_size", 1.0))
            pack_max_base = int(tinfo.get("pack_max_base", 0))
            pack_max_qmax = int(tinfo.get("pack_max_qmax", 0))
            pack_skipped_zero_blocks = int(tinfo.get("pack_skipped_zero_blocks", 0))
            slot_bits = int(tinfo.get("slot_bits", 0))
            packing_value_bits = int(tinfo.get("packing_value_bits", cfg.packing_value_bits))
            padding_bits = int(tinfo.get("padding_bits", 0))
            packing_overflow = bool(tinfo.get("packing_overflow", False))
            num_model_params = int(tinfo.get("num_model_params", 0))
            num_ciphertexts = int(tinfo.get("num_ciphertexts", 0))
            compression_ratio = float(tinfo.get("compression_ratio", 1.0))
            plain_delta = _weighted_average_deltas(client_deltas, client_weights)

        max_abs_error = torch.max(
            torch.abs(flatten_model_update(delta_global) - flatten_model_update(plain_delta))
        ).item()

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
                "enc_time": enc_t,
                "partial_decryption_time": par_t,
                "par_dec_time": par_t,
                "combine_decryption_time": com_t,
                "com_dec_time": com_t,
                "total_crypto_time": total_crypto_t,
                "secure_pack_size": int(cfg.secure_pack_size),
                "slot_bits": slot_bits,
                "packing_value_bits": packing_value_bits,
                "padding_bits": padding_bits,
                "packing_overflow": packing_overflow,
                "num_model_params": num_model_params,
                "num_ciphertexts": num_ciphertexts,
                "compression_ratio": compression_ratio,
                "max_abs_error": float(max_abs_error),
                "pack_block_count": pack_block_count,
                "pack_avg_size": pack_avg_size,
                "pack_max_base": pack_max_base,
                "pack_max_qmax": pack_max_qmax,
                "pack_skipped_zero_blocks": pack_skipped_zero_blocks,
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
    if cfg.method == "fedavg":
        out_csv = results_dir / "fedavg_baseline.csv"
    else:
        out_csv = results_dir / "fedavg_ddfed.csv"
        pack_csv = results_dir / f"fedavg_ddfed_packsize_{int(cfg.secure_pack_size)}.csv"
        df = pd.DataFrame(records)
        df["total_training_time"] = total_train_time
        df.to_csv(pack_csv, index=False)
        print(f"{get_time()} Saved results to: {pack_csv}")
    df = pd.DataFrame(records)
    df["total_training_time"] = total_train_time
    df.to_csv(out_csv, index=False)
    print(f"{get_time()} Saved results to: {out_csv}")
    return df, out_csv


def _plot_packsize_summary(results_dir: Path, summary_df: pd.DataFrame):
    fig_dir = results_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return
    agg = (
        summary_df.groupby("secure_pack_size", as_index=False)[
            ["total_crypto_time", "round_total_time", "max_abs_error", "num_ciphertexts"]
        ]
        .mean()
        .sort_values("secure_pack_size")
    )

    x = agg["secure_pack_size"].astype(int).values
    plt.figure(figsize=(8, 5))
    plt.plot(x, agg["total_crypto_time"].values, marker="o")
    plt.xlabel("secure_pack_size")
    plt.ylabel("total_crypto_time")
    plt.title("Crypto time vs pack size")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(fig_dir / "packing_crypto_time_vs_pack_size.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, agg["round_total_time"].values, marker="o")
    plt.xlabel("secure_pack_size")
    plt.ylabel("round_time")
    plt.title("Round time vs pack size")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(fig_dir / "packing_round_time_vs_pack_size.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, agg["max_abs_error"].values, marker="o")
    plt.xlabel("secure_pack_size")
    plt.ylabel("max_abs_error")
    plt.title("Max abs error vs pack size")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(fig_dir / "packing_max_abs_error_vs_pack_size.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, agg["num_ciphertexts"].values, marker="o")
    plt.xlabel("secure_pack_size")
    plt.ylabel("num_ciphertexts")
    plt.title("Num ciphertexts vs pack size")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(fig_dir / "packing_num_ciphertexts_vs_pack_size.png", dpi=160)
    plt.close()

    # accuracy vs round from all runs
    plt.figure(figsize=(8, 5))
    for psize, g in summary_df.groupby("secure_pack_size"):
        plt.plot(g["round"], g["test_accuracy"], marker="o", label=f"pack={int(psize)}")
    plt.xlabel("round")
    plt.ylabel("test_accuracy")
    plt.title("Accuracy vs round")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "packing_accuracy_vs_round.png", dpi=160)
    plt.close()


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


def run_sanity_packing(cfg):
    """
    Compare plaintext, secure pack_size=1, and packed secure aggregation.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    n_clients = 3
    vec_len = 20
    client_weights = [10, 20, 30]
    client_deltas = []
    for _ in range(n_clients):
        vals = torch.randn(vec_len) * 0.01
        client_deltas.append({"w": vals})

    plain = _weighted_average_deltas(client_deltas, client_weights)
    pack_sizes = [1, 4, 8, 16]
    if int(cfg.secure_pack_size) not in pack_sizes:
        pack_sizes.append(int(cfg.secure_pack_size))
    pack_sizes = sorted(set(pack_sizes))
    rows = []
    for psize in pack_sizes:
        secure, tinfo = secure_aggregate_ddfed(
            client_deltas=client_deltas,
            client_weights=client_weights,
            ddfed_config={
                "ddfed_project_root": cfg.ddfed_project_root,
                "threshold": min(cfg.threshold, cfg.num_decryptors),
                "num_decryptors": max(cfg.num_decryptors, 3),
                "quantization_scale": cfg.quantization_scale,
                "lambda_sec": cfg.lambda_sec,
                "secure_pack_size": int(psize),
                "max_plaintext_bits": cfg.max_plaintext_bits,
                "packing_value_bits": cfg.packing_value_bits,
                "skip_zero_blocks": cfg.skip_zero_blocks,
                "label_base": f"sanity_pack_{psize}",
            },
        )
        diff = torch.abs(flatten_model_update(plain) - flatten_model_update(secure))
        max_abs_error = float(torch.max(diff).item())
        mean_abs_error = float(torch.mean(diff).item())
        passed = max_abs_error <= (6.0 / float(cfg.quantization_scale))
        rows.append(
            {
                "pack_size": int(psize),
                "effective_pack_size": int(tinfo.get("effective_pack_size", psize)),
                "slot_bits": int(tinfo.get("slot_bits", 0)),
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "packing_overflow": bool(tinfo.get("packing_overflow", False)),
                "passed": bool(passed),
            }
        )
    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))
    return out_df


def parse_args():
    parser = argparse.ArgumentParser(description="FedAvg baseline and FedAvg+DDFed trainer")
    parser.add_argument("--method", type=str, default="fedavg", choices=["fedavg", "fedavg_ddfed", "sanity_packing"])
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
    parser.add_argument("--packing_value_bits", type=int, default=32)
    parser.add_argument("--max_plaintext_bits", type=int, default=120, help="upper bound for packed plaintext bit-length")
    parser.add_argument("--skip_zero_blocks", type=str2bool, default=True, help="skip all-zero packed blocks to reduce crypto work")
    parser.add_argument("--packsize_sweep", type=str, default="", help="comma-separated pack sizes, e.g. 1,4,8,16,32")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_sanity_check", action="store_true")
    return parser.parse_args()


def main():
    cfg = parse_args()
    if cfg.method == "sanity_packing":
        run_sanity_packing(cfg)
        return
    if cfg.run_sanity_check:
        sanity_check_secure_aggregation(cfg)
    if str(cfg.packsize_sweep).strip():
        pack_sizes = [int(x.strip()) for x in str(cfg.packsize_sweep).split(",") if x.strip()]
        all_rows = []
        for psize in pack_sizes:
            cfg_one = copy.deepcopy(cfg)
            cfg_one.secure_pack_size = int(psize)
            df, _ = train_fedavg_ddfed(cfg_one)
            all_rows.append(df)
        merged = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        if not merged.empty:
            _plot_packsize_summary(Path(cfg.results_dir), merged)
        return
    train_fedavg_ddfed(cfg)


if __name__ == "__main__":
    main()
