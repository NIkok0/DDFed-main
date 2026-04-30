import argparse
import copy
import hashlib
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
FL_ROOT = CURRENT_DIR.parent.parent
PROJECT_ROOT = FL_ROOT.parent
if str(FL_ROOT) not in sys.path:
    sys.path.append(str(FL_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from FedAvg.server.server_fedavg import FedAvgServer
from FedAvg.server.secure_packing import (
    choose_block_len,
    compute_slot_bits,
    compute_safe_base,
    estimate_block_qmax,
    pack_digits,
    unpack_digits,
)
from FedAvg.utils import setup_seed
from utils.fed_utils import get_time


class AggregationFailure(RuntimeError):
    """Raised when secure aggregation cannot proceed safely."""


class ReplayAttackDetected(AggregationFailure):
    """Raised when stale or cross-set data is detected."""


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
    for key in g_state.keys():
        delta[key] = (l_state[key].detach().cpu() - g_state[key].detach().cpu()).float()
    return delta


def apply_model_delta(global_model: torch.nn.Module, delta_global: Dict[str, torch.Tensor]) -> None:
    """Apply aggregated delta to global model in-place."""
    state = global_model.state_dict()
    for key in state.keys():
        state[key] = state[key] + delta_global[key].to(state[key].device, dtype=state[key].dtype)
    global_model.load_state_dict(state)


def flatten_model_update(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a model-delta dict into one 1D vector."""
    return torch.cat([delta[key].reshape(-1) for key in delta.keys()], dim=0)


def unflatten_model_update(vector: torch.Tensor, model_template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Restore flattened delta vector into model-shaped tensors."""
    cursor = 0
    restored = {}
    for key, tensor in model_template.items():
        n_elem = tensor.numel()
        restored[key] = vector[cursor: cursor + n_elem].reshape(tensor.shape)
        cursor += n_elem
    if cursor != int(vector.numel()):
        raise ValueError(f"Unflatten mismatch: consumed {cursor}, total {int(vector.numel())}")
    return restored


def quantize_update(vector: torch.Tensor, quantization_scale: int) -> torch.Tensor:
    """Float -> fixed-point integer."""
    return torch.round(vector.float() * float(quantization_scale)).to(torch.int64)


def dequantize_update(vector: torch.Tensor, quantization_scale: int) -> torch.Tensor:
    """Fixed-point integer -> float."""
    return vector.float() / float(quantization_scale)


def plaintext_fedavg_aggregate(
    client_deltas: Dict[int, Dict[str, torch.Tensor]],
    client_weights: Dict[int, int],
) -> Dict[str, torch.Tensor]:
    """FedAvg weighted average in plaintext for baseline and sanity-check."""
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
        for key in out.keys():
            out[key] += client_deltas[uid][key].float() * weight
    return out


def _set_fingerprint(values: List[int]) -> str:
    msg = ",".join(str(v) for v in sorted(values))
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()


def _hash_label_payload(payload: Dict) -> str:
    serial = repr(sorted(payload.items()))
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()


def _compute_model_version(model: torch.nn.Module) -> str:
    """Hash model state to bind labels to a specific global-model version."""
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _build_round_label(
    experiment_id: str,
    round_id: int,
    epoch_id: int,
    active_client_set: List[int],
    active_decryptor_set: List[int],
    model_version: str,
) -> str:
    payload = {
        "experiment_id": experiment_id,
        "round_id": int(round_id),
        "epoch_id": int(epoch_id),
        "active_client_set": tuple(sorted(active_client_set)),
        "active_decryptor_set": tuple(sorted(active_decryptor_set)),
        "model_version": model_version,
    }
    return _hash_label_payload(payload)


@dataclass
class RoundMeta:
    label: str
    active_client_enc_ids: List[int]
    active_decryptors: List[int]


class TMCFEContext:
    """Holds TMCFE setup artifacts and replay metadata across rounds."""

    def __init__(self, cfg, all_client_ids: List[int]):
        self.cfg = cfg
        # FedAvg environments often use string client ids like "f_00012".
        self.all_client_ids = sorted(all_client_ids)
        self.uid_to_enc_id = {uid: idx + 1 for idx, uid in enumerate(self.all_client_ids)}
        self.enc_id_to_uid = {v: k for k, v in self.uid_to_enc_id.items()}
        self.num_encryptors = len(self.all_client_ids)
        self.num_decryptors = int(cfg.num_decryptors)
        self.threshold = int(cfg.threshold)
        self.quantization_scale = int(cfg.quantization_scale)
        self.secure_pack_size = int(cfg.secure_pack_size)
        self.max_dlog = int(cfg.max_dlog)
        self.setup_once = bool(cfg.setup_once)

        self.tmcfe = None
        self.public_params = None
        self.client_sk_by_enc_id: Dict[int, dict] = {}
        self.round_meta: Dict[int, RoundMeta] = {}
        self.round_partial_cache: Dict[int, Dict[int, dict]] = {}
        self.current_round = 0

        ddfed_root = Path(cfg.ddfed_project_root).resolve()
        if str(ddfed_root) not in sys.path:
            sys.path.append(str(ddfed_root))
        from ddfed_crypto.baselines.tmcfe import TMCFE  # noqa: WPS433
        self._tmcfe_cls = TMCFE

    def setup_system(self):
        if self.tmcfe is not None:
            return
        # Server(??) ->> Client, Decryptor(??): one-time TMCFE setup before FL round 1.
        self.tmcfe = self._tmcfe_cls()
        self.public_params = self.tmcfe.setup(
            lam=int(self.cfg.lambda_sec),
            n_encryptors=self.num_encryptors,
            n_decryptors=self.num_decryptors,
            t=self.threshold,
        )
        self.client_sk_by_enc_id = {}
        # Server(??) ->> Client(??): distribute persistent sk_i to every potential client once.
        for enc_id in range(1, self.num_encryptors + 1):
            self.client_sk_by_enc_id[enc_id] = self.tmcfe.sk_distribute(enc_id)

    def ensure_setup(self):
        # Phase 1 must be one-time. Do not re-run setup/sk_distribute inside FL rounds.
        if self.tmcfe is None:
            self.setup_system()

    def require_ready(self):
        if self.tmcfe is None or not self.client_sk_by_enc_id:
            raise RuntimeError("TMCFEContext.setup_system() must be called once before FL rounds")


def _choose_replay_source_round(cfg, ctx: TMCFEContext) -> int:
    if int(cfg.replay_source_round) > 0:
        return int(cfg.replay_source_round)
    return max(1, int(ctx.current_round) - 1)


def _should_apply_replay_attack(cfg, ctx: TMCFEContext) -> bool:
    if not bool(cfg.simulate_replay_attack):
        return False
    target = int(cfg.replay_target_round)
    if target > 0 and int(ctx.current_round) != target:
        return False
    return int(ctx.current_round) >= 2


def _validate_ciphertexts(
    ct_payloads: Dict[int, dict],
    expected_label: str,
    expected_enc_client_ids: List[int],
    expected_decryptors: List[int],
):
    expected_client_fingerprint = _set_fingerprint(expected_enc_client_ids)
    expected_dec_fingerprint = _set_fingerprint(expected_decryptors)
    if sorted(ct_payloads.keys()) != sorted(expected_enc_client_ids):
        raise ReplayAttackDetected("Ciphertext client set mismatch with current active set")
    for enc_id, payload in ct_payloads.items():
        meta = payload["meta"]
        if meta["label"] != expected_label:
            raise ReplayAttackDetected(f"Ciphertext label mismatch for enc_id={enc_id}")
        if meta["client_set_fp"] != expected_client_fingerprint:
            raise ReplayAttackDetected(f"Ciphertext active-client-set mismatch for enc_id={enc_id}")
        if meta["decryptor_set_fp"] != expected_dec_fingerprint:
            raise ReplayAttackDetected(f"Ciphertext decryptor-set mismatch for enc_id={enc_id}")


def _validate_partials(
    partial_payloads: Dict[int, dict],
    expected_label: str,
    expected_enc_client_ids: List[int],
    expected_decryptors: List[int],
):
    expected_client_fingerprint = _set_fingerprint(expected_enc_client_ids)
    expected_dec_fingerprint = _set_fingerprint(expected_decryptors)
    if sorted(partial_payloads.keys()) != sorted(expected_decryptors):
        raise ReplayAttackDetected("Partial decryption set mismatch with current active decryptors")
    for dec_id, payload in partial_payloads.items():
        meta = payload["meta"]
        if meta["label"] != expected_label:
            raise ReplayAttackDetected(f"Partial decryption label mismatch for decryptor={dec_id}")
        if meta["client_set_fp"] != expected_client_fingerprint:
            raise ReplayAttackDetected(f"Partial decryption active-client-set mismatch for decryptor={dec_id}")
        if meta["decryptor_set_fp"] != expected_dec_fingerprint:
            raise ReplayAttackDetected(f"Partial decryption decryptor-set mismatch for decryptor={dec_id}")


def _aggregate_single_scalar_tmcfe(
    x_dict_by_uid: Dict[int, int],
    y_dict_by_enc_id: Dict[int, int],
    active_clients: List[int],
    active_decryptors: List[int],
    label: str,
    ctx: TMCFEContext,
    cfg,
    coord_idx: int,
) -> Tuple[int, Dict[str, float], bool]:
    """Securely aggregate one scalar with TMCFE and replay checks."""
    replay_detected = False
    active_enc_ids = [ctx.uid_to_enc_id[uid] for uid in active_clients]
    client_set_fp = _set_fingerprint(active_enc_ids)
    decryptor_set_fp = _set_fingerprint(active_decryptors)

    # Client(??) ->> Server(??) ->> Decryptor(??): active clients encrypt uploads.
    t0 = time.time()
    ct_payloads = {}
    for uid in active_clients:
        enc_id = ctx.uid_to_enc_id[uid]
        xi = int(x_dict_by_uid[uid])
        ct = ctx.tmcfe.encrypt(ctx.client_sk_by_enc_id[enc_id], xi, label)
        ct_payloads[enc_id] = {
            "ct": ct,
            "meta": {
                "label": label,
                "client_set_fp": client_set_fp,
                "decryptor_set_fp": decryptor_set_fp,
            },
        }

    if _should_apply_replay_attack(cfg, ctx) and cfg.replay_attack_type == "client_ciphertext" and coord_idx == 0:
        source_round = _choose_replay_source_round(cfg, ctx)
        if source_round in ctx.round_meta:
            src = ctx.round_meta[source_round]
            src_label = src.label
            src_client_fp = _set_fingerprint(src.active_client_enc_ids)
            num_attack = max(1, int(np.ceil(len(active_clients) * float(cfg.replay_ratio))))
            attacked_uids = sorted(active_clients)[:num_attack]
            for uid in attacked_uids:
                enc_id = ctx.uid_to_enc_id[uid]
                xi = int(x_dict_by_uid[uid])
                # Replayed ciphertext is equivalent to encrypting current value with stale label context.
                old_ct = ctx.tmcfe.encrypt(ctx.client_sk_by_enc_id[enc_id], xi, src_label)
                ct_payloads[enc_id] = {
                    "ct": old_ct,
                    "meta": {
                        "label": src_label,
                        "client_set_fp": src_client_fp,
                        "decryptor_set_fp": _set_fingerprint(src.active_decryptors),
                    },
                }

    if _should_apply_replay_attack(cfg, ctx) and cfg.replay_attack_type == "cross_set" and coord_idx == 0:
        source_round = _choose_replay_source_round(cfg, ctx)
        if source_round in ctx.round_meta:
            src = ctx.round_meta[source_round]
            src_label = src.label
            src_client_fp = _set_fingerprint(src.active_client_enc_ids)
            for old_enc in src.active_client_enc_ids:
                if old_enc not in active_enc_ids:
                    x_dummy = 0
                    old_ct = ctx.tmcfe.encrypt(ctx.client_sk_by_enc_id[old_enc], x_dummy, src_label)
                    ct_payloads[old_enc] = {
                        "ct": old_ct,
                        "meta": {
                            "label": src_label,
                            "client_set_fp": src_client_fp,
                            "decryptor_set_fp": _set_fingerprint(src.active_decryptors),
                        },
                    }
                    break

    t_enc = time.time() - t0

    _validate_ciphertexts(ct_payloads, label, active_enc_ids, active_decryptors)
    ct_dict = {enc_id: payload["ct"] for enc_id, payload in ct_payloads.items()}

    # Decryptor(??) ->> Server(??): independent share_decrypt from each decryptor.
    t0 = time.time()
    partial_payloads = {}
    for dec_id in active_decryptors:
        part = ctx.tmcfe.share_decrypt(
            ctx.dk_dict[dec_id],
            dec_id,
            active_decryptors,
            y_dict_by_enc_id,
            ct_dict,
        )
        partial_payloads[dec_id] = {
            "partial": part,
            "meta": {
                "label": label,
                "client_set_fp": client_set_fp,
                "decryptor_set_fp": decryptor_set_fp,
            },
        }
    t_pardec = time.time() - t0

    # Store first-coordinate partials for replay tests in future rounds.
    if coord_idx == 0:
        ctx.round_partial_cache[ctx.current_round] = copy.deepcopy(partial_payloads)

    if _should_apply_replay_attack(cfg, ctx) and cfg.replay_attack_type == "partial_decryption" and coord_idx == 0:
        source_round = _choose_replay_source_round(cfg, ctx)
        if source_round in ctx.round_partial_cache:
            cached = ctx.round_partial_cache[source_round]
            num_attack = max(1, int(np.ceil(len(active_decryptors) * float(cfg.replay_ratio))))
            attacked_dec = sorted(active_decryptors)[:num_attack]
            for dec_id in attacked_dec:
                if dec_id in cached:
                    partial_payloads[dec_id] = cached[dec_id]

    _validate_partials(partial_payloads, label, active_enc_ids, active_decryptors)
    pardec_dict = {dec_id: payload["partial"] for dec_id, payload in partial_payloads.items()}

    # Server(??): combine threshold shares to recover the weighted plaintext sum.
    t0 = time.time()
    scalar_sum = int(ctx.tmcfe.combine_decrypt(pardec_dict))
    t_comdec = time.time() - t0

    timing = {
        "enc_time": t_enc,
        "par_dec_time": t_pardec,
        "com_dec_time": t_comdec,
    }
    return scalar_sum, timing, replay_detected


def secure_aggregate_tmcfe(
    client_deltas: Dict[int, Dict[str, torch.Tensor]],
    client_weights: Dict[int, int],
    active_clients: List[int],
    active_decryptors: List[int],
    label: str,
    tmcfe_ctx: TMCFEContext,
):
    """
    Securely aggregate weighted deltas using existing TMCFE interfaces.

    Returns:
      - aggregated delta dict
      - timing dict
      - metadata dict
    """
    cfg = tmcfe_ctx.cfg
    if len(active_clients) == 0:
        raise AggregationFailure("No active clients after dropout")
    if len(active_decryptors) < int(cfg.threshold):
        raise AggregationFailure("Not enough active decryptors to satisfy threshold")

    # Phase 2 only: setup/sk_distribute are owned by TMCFEContext before the FL loop.
    tmcfe_ctx.require_ready()
    active_clients = sorted(active_clients)
    active_decryptors = sorted(int(j) for j in active_decryptors)
    tmcfe_ctx.round_meta[tmcfe_ctx.current_round] = RoundMeta(
        label=label,
        active_client_enc_ids=[tmcfe_ctx.uid_to_enc_id[uid] for uid in active_clients],
        active_decryptors=active_decryptors,
    )

    template = client_deltas[active_clients[0]]
    flat_updates = {uid: flatten_model_update(client_deltas[uid]).float() for uid in active_clients}
    vec_len = int(next(iter(flat_updates.values())).numel())
    for uid, vec in flat_updates.items():
        if int(vec.numel()) != vec_len:
            raise AggregationFailure(f"Flatten length mismatch for client {uid}")

    q_updates = {uid: quantize_update(vec, int(cfg.quantization_scale)) for uid, vec in flat_updates.items()}

    total_weight = int(sum(int(client_weights[uid]) for uid in active_clients))
    if total_weight <= 0:
        raise AggregationFailure("Total client weight is not positive")
    y_dict_by_enc_id = {tmcfe_ctx.uid_to_enc_id[uid]: int(client_weights[uid]) for uid in active_clients}

    # Server(??) ->> Decryptor(??): generate this round's dynamic dkj for active clients/label only.
    t0 = time.time()
    tmcfe_ctx.dk_dict = tmcfe_ctx.tmcfe.dk_generate(y_dict_by_enc_id, label)
    dk_generate_time = time.time() - t0

    agg_int = torch.zeros(vec_len, dtype=torch.int64)
    enc_time = 0.0
    par_dec_time = 0.0
    com_dec_time = 0.0
    replay_detected = False
    requested_pack_size = max(1, int(cfg.secure_pack_size))
    idx = 0
    block_id = 0
    block_count = 0
    packed_coords_total = 0
    max_base = 2
    max_qmax = 0
    skipped_zero_blocks = 0
    skip_zero_blocks = bool(getattr(cfg, "skip_zero_blocks", True))
    pos_q_vectors = [torch.clamp(q_updates[uid], min=0) for uid in active_clients]
    neg_q_vectors = [torch.clamp(-q_updates[uid], min=0) for uid in active_clients]

    while idx < vec_len:
        remaining = vec_len - idx
        probe_len = min(requested_pack_size, remaining)
        local_qmax = estimate_block_qmax(pos_q_vectors + neg_q_vectors, idx, probe_len)
        base = compute_safe_base(total_weight, local_qmax, margin=1)
        initial_block_len = choose_block_len(
            requested_pack_size=requested_pack_size,
            remaining=remaining,
            base=base,
            max_dlog=int(cfg.max_dlog),
        )
        block_len = initial_block_len
        max_base = max(max_base, int(base))
        max_qmax = max(max_qmax, int(local_qmax))
        if skip_zero_blocks and local_qmax == 0:
            idx += block_len
            block_id += 1
            block_count += 1
            packed_coords_total += int(block_len)
            skipped_zero_blocks += 1
            continue

        # Shrink block_len on overflow so packed scalar stays in TMCFE dlog range.
        x_pos = {}
        x_neg = {}
        upper_pos = 0
        upper_neg = 0
        while True:
            x_pos = {}
            x_neg = {}
            for uid in active_clients:
                q_slice = q_updates[uid][idx: idx + block_len]
                pos_digits = [max(int(v.item()), 0) for v in q_slice]
                neg_digits = [max(-int(v.item()), 0) for v in q_slice]
                x_pos[uid] = pack_digits(pos_digits, base)
                x_neg[uid] = pack_digits(neg_digits, base)
            upper_pos = sum(int(client_weights[uid]) * int(x_pos[uid]) for uid in active_clients)
            upper_neg = sum(int(client_weights[uid]) * int(x_neg[uid]) for uid in active_clients)
            if (upper_pos <= int(cfg.max_dlog) and upper_neg <= int(cfg.max_dlog)) or block_len <= 1:
                break
            block_len -= 1

        if upper_pos > int(cfg.max_dlog) or upper_neg > int(cfg.max_dlog):
            raise AggregationFailure(
                f"Potential dlog overflow at block {block_id}: "
                f"upper_pos={upper_pos}, upper_neg={upper_neg}, max_dlog={int(cfg.max_dlog)}"
            )

        pos_sum, tinfo, detected = _aggregate_single_scalar_tmcfe(
            x_pos, y_dict_by_enc_id, active_clients, active_decryptors, label, tmcfe_ctx, cfg, block_id
        )
        replay_detected = replay_detected or detected
        enc_time += tinfo["enc_time"]
        par_dec_time += tinfo["par_dec_time"]
        com_dec_time += tinfo["com_dec_time"]

        neg_sum, tinfo, detected = _aggregate_single_scalar_tmcfe(
            x_neg, y_dict_by_enc_id, active_clients, active_decryptors, label, tmcfe_ctx, cfg, block_id
        )
        replay_detected = replay_detected or detected
        enc_time += tinfo["enc_time"]
        par_dec_time += tinfo["par_dec_time"]
        com_dec_time += tinfo["com_dec_time"]

        pos_digits_sum = unpack_digits(pos_sum, base, block_len)
        neg_digits_sum = unpack_digits(neg_sum, base, block_len)
        for off in range(block_len):
            signed_sum = int(pos_digits_sum[off] - neg_digits_sum[off])
            if abs(signed_sum) > int(cfg.max_dlog):
                raise AggregationFailure(
                    f"Signed aggregate integer out of supported dlog range at block {block_id}, "
                    f"offset={off}: {signed_sum}"
                )
            agg_int[idx + off] = signed_sum

        idx += block_len
        block_id += 1
        block_count += 1
        packed_coords_total += int(block_len)

    delta_vec = dequantize_update(agg_int.float(), int(cfg.quantization_scale)) / float(total_weight)
    delta_global = unflatten_model_update(delta_vec, template)
    avg_pack_size = float(packed_coords_total) / float(max(1, block_count))
    slot_bits, padding_bits = compute_slot_bits(
        num_clients=len(active_clients),
        quantization_scale=int(cfg.quantization_scale),
        packing_value_bits=int(getattr(cfg, "packing_value_bits", 32)),
    )
    n_params = int(vec_len)
    n_ciphertexts = int(block_count) * int(len(active_clients))
    timing = {
        "dk_generate_time": dk_generate_time,
        "enc_time": enc_time,
        "share_decrypt_time": par_dec_time,
        "combine_decrypt_time": com_dec_time,
        "total_crypto_time": dk_generate_time + enc_time + par_dec_time + com_dec_time,
        "secure_pack_size": int(requested_pack_size),
        "slot_bits": int(slot_bits),
        "packing_value_bits": int(getattr(cfg, "packing_value_bits", 32)),
        "padding_bits": int(padding_bits),
        "packing_overflow": False,
        "num_model_params": n_params,
        "num_ciphertexts": n_ciphertexts,
        "compression_ratio": float(n_params) / float(max(1, block_count)),
        "pack_block_count": block_count,
        "pack_avg_size": avg_pack_size,
        "pack_max_base": max_base,
        "pack_max_qmax": max_qmax,
        "pack_skipped_zero_blocks": skipped_zero_blocks,
    }
    metadata = {
        "aggregation_success": True,
        "failure_reason": "",
        "replay_detected": replay_detected,
    }
    return delta_global, timing, metadata


def _build_server_args(cfg):
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
        secure_agg_backend="none",
        dmcfe_project_root=cfg.ddfed_project_root,
        dmcfe_scale=cfg.quantization_scale,
    )
    return args


def _sample_dropout(active_items: List[int], dropout_rate: float) -> Tuple[List[int], List[int]]:
    if not active_items:
        return [], []
    n_drop = int(np.floor(len(active_items) * max(0.0, min(1.0, dropout_rate))))
    if n_drop <= 0:
        return sorted(active_items), []
    dropped = sorted(random.sample(active_items, n_drop))
    remain = sorted([x for x in active_items if x not in set(dropped)])
    return remain, dropped


def sanity_check_secure_aggregation(cfg):
    """Compare plaintext FedAvg aggregation and secure TMCFE aggregation."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    fake_clients = [1, 2, 3]
    template = {"w1": torch.zeros(6), "w2": torch.zeros(4)}
    client_deltas = {}
    for uid in fake_clients:
        client_deltas[uid] = {
            "w1": torch.randn_like(template["w1"]) * 0.01,
            "w2": torch.randn_like(template["w2"]) * 0.01,
        }
    client_weights = {1: 5, 2: 3, 3: 2}
    active_decryptors = list(range(1, max(3, int(cfg.num_decryptors)) + 1))

    sane_cfg = copy.deepcopy(cfg)
    sane_cfg.num_decryptors = len(active_decryptors)
    sane_cfg.threshold = min(int(cfg.threshold), len(active_decryptors))
    sane_cfg.setup_once = True
    sane_cfg.simulate_replay_attack = False

    ctx = TMCFEContext(sane_cfg, fake_clients)
    ctx.setup_system()
    ctx.current_round = 1
    label = _build_round_label(
        experiment_id="sanity_check",
        round_id=1,
        epoch_id=1,
        active_client_set=fake_clients,
        active_decryptor_set=active_decryptors,
        model_version="sanity_v1",
    )

    plain = plaintext_fedavg_aggregate(client_deltas, client_weights)
    secure, _, _ = secure_aggregate_tmcfe(
        client_deltas=client_deltas,
        client_weights=client_weights,
        active_clients=fake_clients,
        active_decryptors=active_decryptors,
        label=label,
        tmcfe_ctx=ctx,
    )
    plain_vec = flatten_model_update(plain)
    secure_vec = flatten_model_update(secure)
    abs_err = torch.abs(plain_vec - secure_vec)
    max_abs_error = float(torch.max(abs_err).item())
    mean_abs_error = float(torch.mean(abs_err).item())
    pass_threshold = 5.0 / float(cfg.quantization_scale)
    passed = max_abs_error <= pass_threshold
    print(
        f"{get_time()} sanity_check: max_abs_error={max_abs_error:.10f}, "
        f"mean_abs_error={mean_abs_error:.10f}, passed={passed}"
    )
    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "passed": passed,
    }


def _save_results(df: pd.DataFrame, cfg):
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if cfg.method == "fedavg":
        base_path = results_dir / "fedavg_baseline.csv"
        df.to_csv(base_path, index=False)
        outputs.append(base_path)
    else:
        main_path = results_dir / "fedavg_tmcfe.csv"
        df.to_csv(main_path, index=False)
        outputs.append(main_path)
        if bool(cfg.simulate_client_dropout) or bool(cfg.simulate_decryptor_dropout):
            dropout_path = results_dir / "fedavg_tmcfe_dropout.csv"
            df.to_csv(dropout_path, index=False)
            outputs.append(dropout_path)
            # Compatibility output names.
            df.to_csv(results_dir / "dropout_client.csv", index=False)
            df.to_csv(results_dir / "dropout_decryptor.csv", index=False)
        if bool(cfg.simulate_replay_attack):
            replay_path = results_dir / "fedavg_tmcfe_replay.csv"
            df.to_csv(replay_path, index=False)
            outputs.append(replay_path)
            df.to_csv(results_dir / "replay_attack.csv", index=False)
    return outputs


def train_fedavg_tmcfe(cfg):
    """Main trainer for both fedavg baseline and fedavg_tmcfe."""
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
    experiment_id = hashlib.sha256(f"{cfg.env}|seed={cfg.seed}|method={cfg.method}".encode("utf-8")).hexdigest()[:16]

    tmcfe_ctx = None
    if cfg.method == "fedavg_tmcfe":
        tmcfe_ctx = TMCFEContext(cfg, users)
        # Phase 1 (one-time): setup and sk_distribute for all potential clients before round 1.
        tmcfe_ctx.setup_system()

    records = []
    for r in range(1, int(cfg.num_rounds) + 1):
        round_start = time.time()
        participants = random.sample(users, n_participate) if n_participate < n_total else users[:]
        participants = sorted(participants)

        global_snapshot = copy.deepcopy(server.global_model)
        server.execute_update(participants)

        # Client local training is unchanged; dropout is applied after local updates.
        active_clients = participants[:]
        dropped_clients = []
        if bool(cfg.simulate_client_dropout):
            active_clients, dropped_clients = _sample_dropout(active_clients, float(cfg.client_dropout_rate))

        active_decryptors = list(range(1, int(cfg.num_decryptors) + 1))
        dropped_decryptors = []
        if bool(cfg.simulate_decryptor_dropout):
            active_decryptors, dropped_decryptors = _sample_dropout(active_decryptors, float(cfg.decryptor_dropout_rate))

        model_version = _compute_model_version(global_snapshot)
        label = _build_round_label(
            experiment_id=experiment_id,
            round_id=r,
            epoch_id=int(cfg.local_epochs),
            active_client_set=active_clients,
            active_decryptor_set=active_decryptors,
            model_version=model_version,
        )

        client_deltas = {}
        client_weights = {}
        local_losses = []
        for uid in participants:
            client_obj = server.client_instances[uid]
            client_deltas[uid] = compute_model_delta(global_snapshot, client_obj.local_model)
            client_weights[uid] = len(client_obj.trainset)
            if client_obj.last_eval_loss is not None:
                local_losses.append(float(client_obj.last_eval_loss))

        aggregation_success = False
        failure_reason = ""
        replay_detected = False
        max_abs_error = np.nan
        enc_t = 0.0
        dk_t = 0.0
        par_t = 0.0
        com_t = 0.0
        total_crypto_t = 0.0
        pack_block_count = 0
        pack_avg_size = 1.0
        pack_max_base = 0
        pack_max_qmax = 0
        pack_skipped_zero_blocks = 0
        slot_bits = 0
        packing_value_bits = int(getattr(cfg, "packing_value_bits", 32))
        padding_bits = 0
        packing_overflow = False
        num_model_params = 0
        num_ciphertexts = 0
        compression_ratio = 1.0

        try:
            if len(active_clients) == 0:
                raise AggregationFailure("All selected clients dropped before upload")

            active_delta = {uid: client_deltas[uid] for uid in active_clients}
            active_weight = {uid: client_weights[uid] for uid in active_clients}

            if cfg.method == "fedavg":
                delta_global = plaintext_fedavg_aggregate(active_delta, active_weight)
                aggregation_success = True
            else:
                tmcfe_ctx.current_round = r
                delta_global, timing, meta = secure_aggregate_tmcfe(
                    client_deltas=active_delta,
                    client_weights=active_weight,
                    active_clients=active_clients,
                    active_decryptors=active_decryptors,
                    label=label,
                    tmcfe_ctx=tmcfe_ctx,
                )
                dk_t = float(timing["dk_generate_time"])
                enc_t = float(timing["enc_time"])
                par_t = float(timing["share_decrypt_time"])
                com_t = float(timing["combine_decrypt_time"])
                total_crypto_t = float(timing["total_crypto_time"])
                pack_block_count = int(timing.get("pack_block_count", 0))
                pack_avg_size = float(timing.get("pack_avg_size", 1.0))
                pack_max_base = int(timing.get("pack_max_base", 0))
                pack_max_qmax = int(timing.get("pack_max_qmax", 0))
                pack_skipped_zero_blocks = int(timing.get("pack_skipped_zero_blocks", 0))
                slot_bits = int(timing.get("slot_bits", 0))
                packing_value_bits = int(timing.get("packing_value_bits", packing_value_bits))
                padding_bits = int(timing.get("padding_bits", 0))
                packing_overflow = bool(timing.get("packing_overflow", False))
                num_model_params = int(timing.get("num_model_params", 0))
                num_ciphertexts = int(timing.get("num_ciphertexts", 0))
                compression_ratio = float(timing.get("compression_ratio", 1.0))
                replay_detected = bool(meta["replay_detected"])
                aggregation_success = bool(meta["aggregation_success"])

                plain_delta = plaintext_fedavg_aggregate(active_delta, active_weight)
                err = torch.abs(flatten_model_update(plain_delta) - flatten_model_update(delta_global))
                max_abs_error = float(torch.max(err).item())

            if aggregation_success:
                apply_model_delta(server.global_model, delta_global)
        except ReplayAttackDetected as exc:
            aggregation_success = False
            replay_detected = True
            failure_reason = f"replay_detected: {exc}"
        except Exception as exc:  # noqa: BLE001
            aggregation_success = False
            failure_reason = str(exc)

        test_acc, test_loss = server.test_global_model()
        train_loss = float(np.mean(local_losses)) if local_losses else float("nan")
        round_time = time.time() - round_start

        records.append(
            {
                "round": r,
                "epoch": int(cfg.local_epochs),
                "method": cfg.method,
                "active_clients": active_clients,
                "dropped_clients": dropped_clients,
                "active_decryptors": active_decryptors,
                "dropped_decryptors": dropped_decryptors,
                "label": label,
                "aggregation_success": aggregation_success,
                "failure_reason": failure_reason,
                "replay_attack_enabled": bool(cfg.simulate_replay_attack),
                "replay_attack_type": cfg.replay_attack_type if bool(cfg.simulate_replay_attack) else "",
                "replay_detected": replay_detected,
                "max_abs_error": max_abs_error,
                "train_loss": train_loss,
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "enc_time": enc_t,
                "dk_generate_time": dk_t,
                "par_dec_time": par_t,
                "share_decrypt_time": par_t,
                "com_dec_time": com_t,
                "combine_decrypt_time": com_t,
                "total_crypto_time": total_crypto_t,
                "secure_pack_size": int(cfg.secure_pack_size),
                "slot_bits": slot_bits,
                "packing_value_bits": packing_value_bits,
                "padding_bits": padding_bits,
                "packing_overflow": packing_overflow,
                "num_model_params": num_model_params,
                "num_ciphertexts": num_ciphertexts,
                "compression_ratio": compression_ratio,
                "pack_block_count": pack_block_count,
                "pack_avg_size": pack_avg_size,
                "pack_max_base": pack_max_base,
                "pack_max_qmax": pack_max_qmax,
                "pack_skipped_zero_blocks": pack_skipped_zero_blocks,
                "round_time": round_time,
            }
        )
        print(
            f"{get_time()} round={r} "
            f"success={aggregation_success} replay_detected={replay_detected} "
            f"acc={test_acc:.4f} loss={test_loss:.6f} "
            f"crypto(enc/dk/par/com/total)=({enc_t:.3f}/{dk_t:.3f}/{par_t:.3f}/{com_t:.3f}/{total_crypto_t:.3f})"
        )

    df = pd.DataFrame(records)
    outputs = _save_results(df, cfg)
    for out_path in outputs:
        print(f"{get_time()} saved: {out_path}")
    return df, outputs


def parse_args():
    parser = argparse.ArgumentParser(description="FedAvg baseline and FedAvg+TMCFE secure aggregation")
    parser.add_argument("--method", type=str, default="fedavg", choices=["fedavg", "fedavg_tmcfe"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--env_path", type=str, default="./env")
    parser.add_argument("--env", type=str, default="affine-mnist-seed42-u20-alpha0.1-scale0.01")
    parser.add_argument("--data_path", type=str, default="../../data")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--strategy", type=str, default="quickdrop-affine")
    parser.add_argument("--model", type=str, default="ConvNet")
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--participation_rate", type=float, default=0.2)
    parser.add_argument("--num_clients", type=int, default=20, help="For compatibility; env controls actual clients.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=str2bool, default=False)
    parser.add_argument("--persistent_workers", type=str2bool, default=False)

    # TMCFE / secure aggregation parameters.
    parser.add_argument("--ddfed_project_root", type=str, default="../../DDFed-main")
    parser.add_argument("--lambda_sec", type=int, default=128)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--num_decryptors", type=int, default=10)
    parser.add_argument("--setup_once", type=str2bool, default=True)
    parser.add_argument("--quantization_scale", type=int, default=100000)
    parser.add_argument("--secure_pack_size", type=int, default=1)
    parser.add_argument("--packing_value_bits", type=int, default=32)
    parser.add_argument("--max_dlog", type=int, default=5000000)
    parser.add_argument("--skip_zero_blocks", type=str2bool, default=True)

    # Dynamic dropout experiments.
    parser.add_argument("--simulate_client_dropout", type=str2bool, default=False)
    parser.add_argument("--client_dropout_rate", type=float, default=0.0)
    parser.add_argument("--simulate_decryptor_dropout", type=str2bool, default=False)
    parser.add_argument("--decryptor_dropout_rate", type=float, default=0.0)

    # Replay attack experiments.
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

    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_sanity_check", action="store_true")
    parser.add_argument("--sanity_check_only", action="store_true")
    return parser.parse_args()


def main():
    cfg = parse_args()
    if cfg.run_sanity_check:
        sanity_check_secure_aggregation(cfg)
        if cfg.sanity_check_only:
            return
    train_fedavg_tmcfe(cfg)


if __name__ == "__main__":
    main()
