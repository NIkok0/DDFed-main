import argparse
import copy
import hashlib
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from FedAvg.server.server_fedavg import FedAvgServer
from FedAvg.server.secure_packing import (
    compute_slot_bits,
    pack_client_update_vector,
    unpack_aggregated_vector,
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


def parse_int_csv(value: str) -> List[int]:
    text = str(value or "").strip()
    if not text:
        return []
    out = []
    for item in text.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def compute_model_delta(global_model: torch.nn.Module, local_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    g_state = global_model.state_dict()
    l_state = local_model.state_dict()
    delta = {}
    for key in g_state.keys():
        delta[key] = (l_state[key].detach().cpu() - g_state[key].detach().cpu()).float()
    return delta


def apply_model_delta(global_model: torch.nn.Module, delta_global: Dict[str, torch.Tensor]) -> None:
    state = global_model.state_dict()
    for key in state.keys():
        state[key] = state[key] + delta_global[key].to(state[key].device, dtype=state[key].dtype)
    global_model.load_state_dict(state)


def flatten_model_update(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([delta[key].reshape(-1) for key in delta.keys()], dim=0)


def unflatten_model_update(vector: torch.Tensor, model_template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    restored = {}
    cursor = 0
    for key, tensor in model_template.items():
        n_elem = tensor.numel()
        restored[key] = vector[cursor: cursor + n_elem].reshape(tensor.shape)
        cursor += n_elem
    return restored


def plaintext_fedavg_aggregate(
    client_deltas: Dict[str, Dict[str, torch.Tensor]],
    client_weights: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    if not client_deltas:
        raise ValueError("client_deltas is empty")
    active_clients = list(client_deltas.keys())
    total_weight = float(sum(int(client_weights[uid]) for uid in active_clients))
    if total_weight <= 0:
        raise ValueError("sum of client weights must be positive")
    template = client_deltas[active_clients[0]]
    output = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in template.items()}
    for uid in active_clients:
        w = float(int(client_weights[uid])) / total_weight
        for key in output.keys():
            output[key] += client_deltas[uid][key].float() * w
    return output


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
        dmcfe_project_root=cfg.lepcat_project_root,
        dmcfe_scale=cfg.quantization_scale,
    )


def _compute_model_version(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _build_block_label(
    experiment_id: str,
    round_id: int,
    active_client_count: int,
    selected_enc_ids: List[int],
    model_version: str,
    secure_pack_size: int,
    block_id: int,
) -> str:
    payload = (
        str(experiment_id),
        int(round_id),
        int(active_client_count),
        tuple(sorted(int(x) for x in selected_enc_ids)),
        str(model_version),
        int(secure_pack_size),
        int(block_id),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _resolve_active_client_count(cfg, total_clients: int) -> int:
    if int(cfg.active_client_count) > 0:
        return min(int(cfg.active_client_count), int(total_clients))
    return max(1, min(int(total_clients), int(round(float(cfg.participation_rate) * float(total_clients)))))


def _sanitize_participant_counts(requested: List[int], total_clients: int) -> List[int]:
    out = []
    for x in requested:
        if x <= 0:
            continue
        if x <= int(total_clients):
            out.append(int(x))
    out = sorted(set(out))
    if out:
        return out
    default_counts = [5, 10, 20, 50]
    filtered = [x for x in default_counts if x <= int(total_clients)]
    if filtered:
        return filtered
    return [max(1, int(total_clients))]


class LepcatContext:
    def __init__(self, cfg, all_users: List[str]):
        self.cfg = cfg
        self.all_users = sorted(list(all_users))
        self.uid_to_enc_id = {uid: idx + 1 for idx, uid in enumerate(self.all_users)}
        self.enc_id_to_uid = {idx + 1: uid for idx, uid in enumerate(self.all_users)}
        self.lepcat = None
        self.pp = None
        self.client_sks = {}
        self.client_pks = {}
        self.client_shares = {}

    def setup(self):
        root = Path(self.cfg.lepcat_project_root).resolve()
        if str(root) not in sys.path:
            sys.path.append(str(root))
        from ddfed_crypto.baselines.dmcfe_ip import DMCFE_IP  # noqa: WPS433

        self.lepcat = DMCFE_IP()
        self.pp = self.lepcat.GlobalSetup(
            lam=int(self.cfg.lambda_sec),
            n_encryptors=len(self.all_users),
            t=int(self.cfg.threshold),
        )
        for enc_id in range(1, len(self.all_users) + 1):
            sk_i, pk_i = self.lepcat.ClientSetup()
            shares_i = self.lepcat.KeySharing(sk_i)
            self.client_sks[enc_id] = sk_i
            self.client_pks[enc_id] = pk_i
            self.client_shares[enc_id] = shares_i


def secure_aggregate_lepcat(
    cfg,
    lepcat_ctx: LepcatContext,
    client_deltas: Dict[str, Dict[str, torch.Tensor]],
    client_weights: Dict[str, int],
    selected_users: List[str],
    round_id: int,
    experiment_id: str,
    model_version: str,
):
    selected_users = sorted(list(selected_users))
    selected_enc_ids = [lepcat_ctx.uid_to_enc_id[uid] for uid in selected_users]
    selected_pk_dict = {enc_id: lepcat_ctx.client_pks[enc_id] for enc_id in selected_enc_ids}
    selected_shares_dh_dict = {enc_id: lepcat_ctx.client_shares[enc_id]["shares_dh"] for enc_id in selected_enc_ids}
    y_dict = {lepcat_ctx.uid_to_enc_id[uid]: int(client_weights[uid]) for uid in selected_users}
    sum_weights = int(sum(int(client_weights[uid]) for uid in selected_users))
    if sum_weights <= 0:
        raise ValueError("sum_weights must be positive")

    t0 = time.time()
    for enc_id in selected_enc_ids:
        payload = lepcat_ctx.lepcat.AgreeOnWeightY_Sign(y_dict[enc_id], lepcat_ctx.client_sks[enc_id])
        ok = lepcat_ctx.lepcat.AgreeOnWeightY_Verify(payload, lepcat_ctx.client_pks[enc_id]["sig_pk_i"])
        if not ok:
            return None, {
                "aggregation_success": False,
                "failure_reason": "weight signature verification failed",
                "weight_sign_verify_time": time.time() - t0,
            }
    weight_sign_verify_time = time.time() - t0

    template = client_deltas[selected_users[0]]
    flat_vectors = {uid: flatten_model_update(client_deltas[uid]).float() for uid in selected_users}
    vec_len = int(flat_vectors[selected_users[0]].numel())
    quantization_scale = int(cfg.quantization_scale)
    q_vectors = {uid: torch.round(flat_vectors[uid] * float(quantization_scale)).to(torch.int64) for uid in selected_users}

    slot_bits, padding_bits = compute_slot_bits(
        num_clients=len(selected_users),
        quantization_scale=quantization_scale,
        packing_value_bits=int(cfg.packing_value_bits),
    )
    slot_bits += int(math.ceil(math.log2(max(2, sum_weights + 1))))
    offset = 1 << max(1, int(cfg.packing_value_bits) - 1)
    slot_upper = 1 << int(slot_bits)
    max_slots_by_bits = max(1, int(cfg.max_plaintext_bits) // max(1, int(slot_bits)))
    effective_pack_size = max(1, min(int(cfg.secure_pack_size), int(max_slots_by_bits)))

    t0 = time.time()
    packed_blocks = {}
    packing_overflow = False
    for uid in selected_users:
        blocks, overflow = pack_client_update_vector(
            q_vectors[uid],
            pack_size=int(effective_pack_size),
            slot_bits=int(slot_bits),
            offset=int(offset),
        )
        packed_blocks[uid] = blocks
        packing_overflow = packing_overflow or bool(overflow)

    for idx in range(vec_len):
        weighted_encoded_sum = 0
        for uid in selected_users:
            weighted_encoded_sum += int(client_weights[uid]) * int(q_vectors[uid][idx].item() + int(offset))
        if weighted_encoded_sum < 0 or weighted_encoded_sum >= slot_upper:
            packing_overflow = True
            break
    pack_time = time.time() - t0
    if packing_overflow:
        return None, {
            "aggregation_success": False,
            "failure_reason": "packing overflow",
            "weight_sign_verify_time": weight_sign_verify_time,
            "pack_time": pack_time,
            "packing_overflow": True,
        }

    n_blocks = len(next(iter(packed_blocks.values())))
    for uid in selected_users[1:]:
        if len(packed_blocks[uid]) != n_blocks:
            raise ValueError("packed block length mismatch across clients")

    enc_time = 0.0
    aggregation_time = 0.0
    agg_packed_blocks = []
    for block_id in range(n_blocks):
        block_label = _build_block_label(
            experiment_id=experiment_id,
            round_id=round_id,
            active_client_count=len(selected_users),
            selected_enc_ids=selected_enc_ids,
            model_version=model_version,
            secure_pack_size=int(cfg.secure_pack_size),
            block_id=block_id,
        )
        enc_payloads = {}
        t1 = time.time()
        for uid in selected_users:
            enc_id = lepcat_ctx.uid_to_enc_id[uid]
            enc_payloads[enc_id] = lepcat_ctx.lepcat.Encryption(
                i=enc_id,
                sk_i=lepcat_ctx.client_sks[enc_id],
                x_i=int(packed_blocks[uid][block_id]),
                y_i=int(client_weights[uid]),
                pk_dict=selected_pk_dict,
                label_l=block_label,
            )
        enc_time += time.time() - t1

        t1 = time.time()
        agg_packed = lepcat_ctx.lepcat.Aggregation(
            U=selected_enc_ids,
            all_encryptors=selected_enc_ids,
            enc_payloads=enc_payloads,
            shares_dh_dict=selected_shares_dh_dict,
            pk_dict=selected_pk_dict,
            label_l=block_label,
            y_dict=y_dict,
        )
        aggregation_time += time.time() - t1
        agg_packed_blocks.append(int(agg_packed))

    t0 = time.time()
    decoded_weighted_encoded = unpack_aggregated_vector(
        packed_values=agg_packed_blocks,
        original_length=int(vec_len),
        pack_size=int(effective_pack_size),
        slot_bits=int(slot_bits),
        offset=0,
    )
    q_sum_vec = torch.zeros(vec_len, dtype=torch.float32)
    max_abs_qsum = 0
    for idx, val in enumerate(decoded_weighted_encoded):
        q_sum = int(val) - int(offset * sum_weights)
        max_abs_qsum = max(max_abs_qsum, abs(q_sum))
        q_sum_vec[idx] = float(q_sum)
    delta_vec = q_sum_vec / float(sum_weights) / float(quantization_scale)
    unpack_time = time.time() - t0

    delta_global = unflatten_model_update(delta_vec, template)
    plain_delta = plaintext_fedavg_aggregate(client_deltas, client_weights)
    max_abs_error = float(
        torch.max(torch.abs(flatten_model_update(delta_global) - flatten_model_update(plain_delta))).item()
    )
    total_crypto_time = weight_sign_verify_time + pack_time + enc_time + aggregation_time + unpack_time
    meta = {
        "aggregation_success": True,
        "failure_reason": "",
        "weight_sign_verify_time": weight_sign_verify_time,
        "pack_time": pack_time,
        "encrypt_time": enc_time,
        "aggregation_time": aggregation_time,
        "unpack_time": unpack_time,
        "total_crypto_time": total_crypto_time,
        "slot_bits": int(slot_bits),
        "packing_value_bits": int(cfg.packing_value_bits),
        "padding_bits": int(padding_bits),
        "packing_overflow": False,
        "num_model_params": int(vec_len),
        "num_plain_values": int(vec_len),
        "num_packed_blocks": int(n_blocks),
        "num_ciphertexts": int(n_blocks) * int(len(selected_users)),
        "compression_ratio": float(vec_len) / float(max(1, n_blocks)),
        "sum_weights": int(sum_weights),
        "max_abs_error": max_abs_error,
        "effective_pack_size": int(effective_pack_size),
        "pack_max_abs_qsum": int(max_abs_qsum),
    }
    return delta_global, meta


def _run_single_training(cfg, method: str, active_client_count: int, secure_pack_size: int):
    setup_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if "cuda" in cfg.device and not torch.cuda.is_available():
        print(f"{get_time()} CUDA requested but unavailable, fallback to cpu.")
        cfg.device = "cpu"

    run_cfg = copy.deepcopy(cfg)
    run_cfg.method = method
    run_cfg.secure_pack_size = int(secure_pack_size)

    server = FedAvgServer(_build_server_args(run_cfg))
    users = sorted(list(server.users))
    n_total = len(users)
    n_active = min(int(active_client_count), int(n_total))
    experiment_id = hashlib.sha256(
        f"{run_cfg.env}|seed={run_cfg.seed}|method={method}|active={n_active}|pack={secure_pack_size}".encode("utf-8")
    ).hexdigest()[:16]

    lepcat_ctx = None
    if method == "fedavg_lepcat":
        lepcat_ctx = LepcatContext(run_cfg, users)
        lepcat_ctx.setup()

    rows = []
    for round_id in range(1, int(run_cfg.num_rounds) + 1):
        round_start = time.time()
        selected_users = random.sample(users, n_active) if n_active < n_total else users[:]
        selected_users = sorted(selected_users)
        global_snapshot = copy.deepcopy(server.global_model)
        comm_start = time.time()
        server.execute_update(selected_users)
        communication_time = time.time() - comm_start

        client_deltas = {}
        client_weights = {}
        local_losses = []
        for uid in selected_users:
            client = server.client_instances[uid]
            client_deltas[uid] = compute_model_delta(global_snapshot, client.local_model)
            client_weights[uid] = int(len(client.trainset))
            if client.last_eval_loss is not None:
                local_losses.append(float(client.last_eval_loss))
        sum_weights = int(sum(client_weights.values()))

        aggregation_success = True
        failure_reason = ""
        weight_sign_verify_time = 0.0
        pack_time = 0.0
        encrypt_time = 0.0
        aggregation_time = 0.0
        unpack_time = 0.0
        total_crypto_time = 0.0
        encryption_time = 0.0
        enc_time = 0.0
        partial_decryption_time = 0.0
        par_dec_time = 0.0
        combine_decryption_time = 0.0
        com_dec_time = 0.0
        slot_bits = 0
        packing_value_bits = int(run_cfg.packing_value_bits)
        padding_bits = 0
        num_model_params = int(flatten_model_update(client_deltas[selected_users[0]]).numel())
        num_plain_values = num_model_params
        num_packed_blocks = 0
        num_ciphertexts = 0
        compression_ratio = 1.0
        packing_overflow = False
        max_abs_error = 0.0

        if method == "fedavg":
            delta_global = plaintext_fedavg_aggregate(client_deltas, client_weights)
            encryption_time = 0.0
            enc_time = 0.0
            partial_decryption_time = 0.0
            par_dec_time = 0.0
            combine_decryption_time = 0.0
            com_dec_time = 0.0
        else:
            model_version = _compute_model_version(global_snapshot)
            delta_global, meta = secure_aggregate_lepcat(
                cfg=run_cfg,
                lepcat_ctx=lepcat_ctx,
                client_deltas=client_deltas,
                client_weights=client_weights,
                selected_users=selected_users,
                round_id=round_id,
                experiment_id=experiment_id,
                model_version=model_version,
            )
            aggregation_success = bool(meta.get("aggregation_success", False))
            failure_reason = str(meta.get("failure_reason", ""))
            weight_sign_verify_time = float(meta.get("weight_sign_verify_time", 0.0))
            pack_time = float(meta.get("pack_time", 0.0))
            encrypt_time = float(meta.get("encrypt_time", 0.0))
            aggregation_time = float(meta.get("aggregation_time", 0.0))
            unpack_time = float(meta.get("unpack_time", 0.0))
            total_crypto_time = float(meta.get("total_crypto_time", 0.0))
            encryption_time = float(encrypt_time)
            enc_time = float(encrypt_time)
            combine_decryption_time = float(aggregation_time + unpack_time)
            com_dec_time = float(combine_decryption_time)
            slot_bits = int(meta.get("slot_bits", 0))
            packing_value_bits = int(meta.get("packing_value_bits", packing_value_bits))
            padding_bits = int(meta.get("padding_bits", 0))
            num_model_params = int(meta.get("num_model_params", num_model_params))
            num_plain_values = int(meta.get("num_plain_values", num_plain_values))
            num_packed_blocks = int(meta.get("num_packed_blocks", 0))
            num_ciphertexts = int(meta.get("num_ciphertexts", 0))
            compression_ratio = float(meta.get("compression_ratio", 1.0))
            packing_overflow = bool(meta.get("packing_overflow", False))
            max_abs_error = float(meta.get("max_abs_error", np.nan))

        if aggregation_success:
            apply_model_delta(server.global_model, delta_global)
        test_acc, test_loss = server.test_global_model()
        train_loss = float(np.mean(local_losses)) if local_losses else float("nan")
        round_time = time.time() - round_start
        round_total_time = float(round_time)
        pack_block_count = int(num_packed_blocks)
        pack_avg_size = float(num_model_params) / float(max(1, num_packed_blocks))
        pack_max_base = 0
        pack_max_qmax = 0
        pack_skipped_zero_blocks = 0

        rows.append(
            {
                "round": int(round_id),
                "method": str(method),
                "num_clients": int(n_total),
                "participation_rate": float(run_cfg.participation_rate),
                "communication_time": float(communication_time),
                "encryption_time": float(encryption_time),
                "enc_time": float(enc_time),
                "partial_decryption_time": float(partial_decryption_time),
                "par_dec_time": float(par_dec_time),
                "combine_decryption_time": float(combine_decryption_time),
                "com_dec_time": float(com_dec_time),
                "pack_block_count": int(pack_block_count),
                "pack_avg_size": float(pack_avg_size),
                "pack_max_base": int(pack_max_base),
                "pack_max_qmax": int(pack_max_qmax),
                "pack_skipped_zero_blocks": int(pack_skipped_zero_blocks),
                "round_total_time": float(round_total_time),
                "active_client_count": int(n_active),
                "num_participants": int(n_active),
                "selected_client_count": int(len(selected_users)),
                "selected_clients": "|".join(str(x) for x in selected_users),
                "sum_weights": int(sum_weights),
                "secure_pack_size": int(run_cfg.secure_pack_size),
                "slot_bits": int(slot_bits),
                "packing_value_bits": int(packing_value_bits),
                "padding_bits": int(padding_bits),
                "num_model_params": int(num_model_params),
                "num_plain_values": int(num_plain_values),
                "num_packed_blocks": int(num_packed_blocks),
                "num_ciphertexts": int(num_ciphertexts),
                "compression_ratio": float(compression_ratio),
                "packing_overflow": bool(packing_overflow),
                "aggregation_success": bool(aggregation_success),
                "failure_reason": str(failure_reason),
                "max_abs_error": float(max_abs_error),
                "train_loss": float(train_loss),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "weight_sign_verify_time": float(weight_sign_verify_time),
                "pack_time": float(pack_time),
                "encrypt_time": float(encrypt_time),
                "aggregation_time": float(aggregation_time),
                "unpack_time": float(unpack_time),
                "total_crypto_time": float(total_crypto_time),
                "round_time": float(round_time),
            }
        )
        print(
            f"{get_time()} round={round_id} method={method} active={n_active} "
            f"success={aggregation_success} acc={float(test_acc):.4f} "
            f"crypto_total={float(total_crypto_time):.4f}s round_time={float(round_time):.4f}s"
        )

    df = pd.DataFrame(rows)
    return df


def _save_round_results(df: pd.DataFrame, results_dir: Path, method: str, active_client_count: int, secure_pack_size: int):
    results_dir.mkdir(parents=True, exist_ok=True)
    if method == "fedavg_lepcat":
        out_path = results_dir / f"fedavg_lepcat_participants_{int(active_client_count)}_pack_{int(secure_pack_size)}.csv"
    else:
        out_path = results_dir / f"fedavg_participants_{int(active_client_count)}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _make_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for (active_count, pack_size), part in df_all.groupby(["active_client_count", "secure_pack_size"], dropna=False):
        success_rate = float(part["aggregation_success"].astype(float).mean()) if len(part) else 0.0
        grouped.append(
            {
                "active_client_count": int(active_count),
                "secure_pack_size": int(pack_size),
                "avg_test_accuracy": float(part["test_accuracy"].mean()),
                "final_test_accuracy": float(part["test_accuracy"].iloc[-1]),
                "avg_encrypt_time": float(part["encrypt_time"].mean()),
                "avg_aggregation_time": float(part["aggregation_time"].mean()),
                "avg_total_crypto_time": float(part["total_crypto_time"].mean()),
                "avg_round_time": float(part["round_time"].mean()),
                "avg_num_ciphertexts": float(part["num_ciphertexts"].mean()),
                "avg_compression_ratio": float(part["compression_ratio"].mean()),
                "avg_max_abs_error": float(part["max_abs_error"].mean()),
                "success_rate": float(success_rate),
            }
        )
    out = pd.DataFrame(grouped)
    if not out.empty:
        out = out.sort_values(["secure_pack_size", "active_client_count"]).reset_index(drop=True)
    return out


def _plot_participant_scalability(summary_df: pd.DataFrame, figures_dir: Path):
    if summary_df.empty:
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    metrics = [
        ("avg_total_crypto_time", "lepcat_participants_crypto_time.png"),
        ("avg_round_time", "lepcat_participants_round_time.png"),
        ("avg_num_ciphertexts", "lepcat_participants_num_ciphertexts.png"),
        ("final_test_accuracy", "lepcat_participants_accuracy.png"),
        ("avg_max_abs_error", "lepcat_participants_error.png"),
    ]
    for metric, filename in metrics:
        plt.figure(figsize=(8, 5))
        for psize, part in summary_df.groupby("secure_pack_size"):
            part = part.sort_values("active_client_count")
            plt.plot(
                part["active_client_count"].values,
                part[metric].values,
                marker="o",
                label=f"secure_pack_size={int(psize)}",
            )
        plt.xlabel("active_client_count")
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        out_path = figures_dir / filename
        plt.savefig(out_path, dpi=160)
        plt.close()
        outputs.append(out_path)
    return outputs


def run_sanity_lepcat(cfg):
    setup_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    participant_counts = parse_int_csv(cfg.participant_counts)
    if not participant_counts:
        participant_counts = [int(cfg.active_client_count)] if int(cfg.active_client_count) > 0 else [5, 10, 20]
    participant_counts = sorted(set(x for x in participant_counts if x > 0))
    if not participant_counts:
        participant_counts = [5]
    pack_sizes = parse_int_csv(cfg.pack_sizes)
    if not pack_sizes:
        pack_sizes = sorted(set([1, int(cfg.secure_pack_size), 4, 8, 16]))
    pack_sizes = [x for x in pack_sizes if x > 0]

    max_clients = max(participant_counts)
    fake_users = [f"u_{idx:03d}" for idx in range(max_clients)]
    base_vectors = {}
    base_weights = {}
    for uid in fake_users:
        base_vectors[uid] = torch.randn(64) * 0.01
        base_weights[uid] = int(np.random.randint(5, 25))

    ctx_cfg = copy.deepcopy(cfg)
    ctx_cfg.threshold = min(int(cfg.threshold), int(max_clients))
    lepcat_ctx = LepcatContext(ctx_cfg, fake_users)
    lepcat_ctx.setup()

    rows = []
    for active_client_count in participant_counts:
        selected_users = fake_users[: int(active_client_count)]
        deltas = {uid: {"w": base_vectors[uid].clone()} for uid in selected_users}
        weights = {uid: int(base_weights[uid]) for uid in selected_users}
        plain = plaintext_fedavg_aggregate(deltas, weights)
        plain_vec = flatten_model_update(plain)
        for psize in pack_sizes:
            run_cfg = copy.deepcopy(cfg)
            run_cfg.secure_pack_size = int(psize)
            secure_delta, meta = secure_aggregate_lepcat(
                cfg=run_cfg,
                lepcat_ctx=lepcat_ctx,
                client_deltas=deltas,
                client_weights=weights,
                selected_users=selected_users,
                round_id=1,
                experiment_id="sanity_lepcat",
                model_version="sanity_v1",
            )
            if secure_delta is None:
                rows.append(
                    {
                        "active_client_count": int(active_client_count),
                        "secure_pack_size": int(psize),
                        "slot_bits": int(meta.get("slot_bits", 0)),
                        "max_abs_error": np.nan,
                        "mean_abs_error": np.nan,
                        "packing_overflow": bool(meta.get("packing_overflow", False)),
                        "passed": False,
                    }
                )
                continue

            secure_vec = flatten_model_update(secure_delta)
            err = torch.abs(plain_vec - secure_vec)
            max_abs_error = float(torch.max(err).item())
            mean_abs_error = float(torch.mean(err).item())
            passed = bool(max_abs_error <= (6.0 / float(cfg.quantization_scale)))
            rows.append(
                {
                    "active_client_count": int(active_client_count),
                    "secure_pack_size": int(psize),
                    "slot_bits": int(meta.get("slot_bits", 0)),
                    "max_abs_error": max_abs_error,
                    "mean_abs_error": mean_abs_error,
                    "packing_overflow": bool(meta.get("packing_overflow", False)),
                    "passed": passed,
                }
            )

    df = pd.DataFrame(rows).sort_values(["active_client_count", "secure_pack_size"]).reset_index(drop=True)
    print(df.to_string(index=False))
    if bool(cfg.save_results):
        out_dir = Path(cfg.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sanity_lepcat.csv"
        df.to_csv(out_path, index=False)
        print(f"{get_time()} saved sanity results: {out_path}")
    return df


def probe_total_clients(cfg) -> int:
    probe_server = FedAvgServer(_build_server_args(cfg))
    return int(len(probe_server.users))


def run_main(cfg):
    if cfg.method == "sanity_lepcat":
        run_sanity_lepcat(cfg)
        return

    total_clients = probe_total_clients(cfg)
    if cfg.experiment == "participant_scalability":
        requested = parse_int_csv(cfg.participant_counts)
        participant_counts = _sanitize_participant_counts(requested, total_clients)
        pack_sizes = parse_int_csv(cfg.pack_sizes) or [int(cfg.secure_pack_size)]
    else:
        participant_counts = [_resolve_active_client_count(cfg, total_clients)]
        pack_sizes = [int(cfg.secure_pack_size)]

    all_frames = []
    result_paths = []
    for pack_size in pack_sizes:
        for active_count in participant_counts:
            df = _run_single_training(
                cfg=cfg,
                method=cfg.method,
                active_client_count=int(active_count),
                secure_pack_size=int(pack_size),
            )
            all_frames.append(df)
            if bool(cfg.save_results):
                result_paths.append(
                    _save_round_results(
                        df=df,
                        results_dir=Path(cfg.results_dir),
                        method=cfg.method,
                        active_client_count=int(active_count),
                        secure_pack_size=int(pack_size),
                    )
                )

    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True)
    else:
        merged = pd.DataFrame()

    if bool(cfg.save_results) and not merged.empty:
        results_dir = Path(cfg.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        if cfg.method == "fedavg_lepcat":
            main_path = results_dir / "fedavg_lepcat.csv"
        elif cfg.method == "fedavg":
            main_path = results_dir / "fedavg_baseline.csv"
        else:
            main_path = results_dir / f"{cfg.method}.csv"
        merged.to_csv(main_path, index=False)
        result_paths.append(main_path)
        print(f"{get_time()} saved main result: {main_path}")
        if cfg.method == "fedavg_lepcat":
            if bool(getattr(cfg, "simulate_client_dropout", False)):
                dropout_path = results_dir / "fedavg_lepcat_dropout.csv"
                merged.to_csv(dropout_path, index=False)
                result_paths.append(dropout_path)
                print(f"{get_time()} saved dropout result: {dropout_path}")
            if bool(getattr(cfg, "simulate_replay_attack", False)):
                replay_path = results_dir / "fedavg_lepcat_replay.csv"
                merged.to_csv(replay_path, index=False)
                result_paths.append(replay_path)
                print(f"{get_time()} saved replay result: {replay_path}")

        if cfg.method == "fedavg_lepcat":
            summary = _make_summary(merged)
            summary_path = results_dir / "fedavg_lepcat_participant_scalability_summary.csv"
            summary.to_csv(summary_path, index=False)
            result_paths.append(summary_path)
            print(f"{get_time()} saved summary: {summary_path}")
            if bool(cfg.save_figures):
                fig_paths = _plot_participant_scalability(summary, Path(cfg.figures_dir))
                for fig_path in fig_paths:
                    print(f"{get_time()} saved figure: {fig_path}")

    for path in result_paths:
        print(f"{get_time()} saved result: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="FedAvg and FedAvg-Lepcat experiment runner")
    parser.add_argument("--method", type=str, default="fedavg", choices=["fedavg", "fedavg_lepcat", "sanity_lepcat"])
    parser.add_argument("--experiment", type=str, default="", choices=["", "participant_scalability"])

    parser.add_argument("--active_client_count", type=int, default=-1)
    parser.add_argument("--participant_counts", type=str, default="")
    parser.add_argument("--pack_sizes", type=str, default="")

    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--participation_rate", type=float, default=0.2)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--env_path", type=str, default="./env")
    parser.add_argument("--env", type=str, default="affine-mnist-seed42-u20-alpha0.1-scale0.01")
    parser.add_argument("--strategy", type=str, default="quickdrop-affine")
    parser.add_argument("--data_path", type=str, default="../../data")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--model", type=str, default="ConvNet")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=str2bool, default=False)
    parser.add_argument("--persistent_workers", type=str2bool, default=False)

    parser.add_argument("--lepcat_project_root", type=str, default="../../DDFed-main")
    parser.add_argument("--lambda_sec", type=int, default=128)
    parser.add_argument("--quantization_scale", type=int, default=100000)
    parser.add_argument("--secure_pack_size", type=int, default=1)
    parser.add_argument("--packing_value_bits", type=int, default=32)
    parser.add_argument("--max_plaintext_bits", type=int, default=120)
    parser.add_argument("--simulate_client_dropout", type=str2bool, default=False)
    parser.add_argument("--client_dropout_rate", type=float, default=0.0)
    parser.add_argument("--simulate_replay_attack", type=str2bool, default=False)
    parser.add_argument(
        "--replay_attack_type",
        type=str,
        default="client_ciphertext",
        choices=["client_ciphertext", "partial_decryption", "cross_set"],
    )
    parser.add_argument("--replay_ratio", type=float, default=0.2)
    parser.add_argument("--replay_source_round", type=int, default=0)
    parser.add_argument("--replay_target_round", type=int, default=0)

    parser.add_argument("--save_results", type=str2bool, default=True)
    parser.add_argument("--save_figures", type=str2bool, default=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--figures_dir", type=str, default="figures")
    return parser.parse_args()


def main():
    cfg = parse_args()
    run_main(cfg)


if __name__ == "__main__":
    main()
