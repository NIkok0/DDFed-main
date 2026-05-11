# -*- coding: utf-8 -*-
"""
DDFed-FU Unified CLI Parser
A single command-line interface for all three federated unlearning algorithms.

Usage:
    # FedEraser
    python -m ddfed_fu.common.cli_parser --method federaser \\
        --data_name mnist --n_clients 20 --forget_client_idx 3

    # Backdoor Unlearning
    python -m ddfed_fu.common.cli_parser --method backdoor \\
        --data_name cifar10 --poison --unlearn

    # QuickDrop
    python -m ddfed_fu.common.cli_parser --method quickdrop \\
        --dataset CIFAR10 --env cifar10-seed42-u20-alpha0.1 --scale 0.01
"""

import argparse
import sys
from typing import Union

from .base_config import (
    BaseFLConfig,
    FedEraserConfig,
    BackdoorUnlearnConfig,
    QuickDropConfig,
)

MethodConfig = Union[FedEraserConfig, BackdoorUnlearnConfig, QuickDropConfig]


# ═══════════════════════════════════════════════════════════════════════════
# Shared argument registrations
# ═══════════════════════════════════════════════════════════════════════════

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register arguments shared by all three algorithms."""
    common = parser.add_argument_group("Common FL Parameters")
    common.add_argument(
        "--data_name", type=str, default="mnist",
        choices=["mnist", "fashion-mnist", "cifar10", "imagenet", "purchase", "adult",
                 "MNIST", "FashionMNIST", "Fashion-MNIST", "CIFAR10", "CIFAR-10",
                 "ImageNet", "Purchase", "Adult"],
        help="Dataset name."
    )
    common.add_argument(
        "--n_clients", type=int, default=10,
        help="Number of participating clients per round."
    )
    common.add_argument(
        "--n_total_clients", type=int, default=100,
        help="Total number of clients in the FL system."
    )
    common.add_argument(
        "--global_epoch", type=int, default=20,
        help="Number of global communication rounds."
    )
    common.add_argument(
        "--local_epoch", type=int, default=10,
        help="Number of local training epochs per round."
    )
    common.add_argument(
        "--local_lr", type=float, default=0.005,
        help="Local learning rate."
    )
    common.add_argument(
        "--local_batch_size", type=int, default=64,
        help="Local batch size for client training."
    )
    common.add_argument(
        "--seed", type=int, default=1,
        help="Random seed for reproducibility."
    )
    common.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cpu', 'cuda', or 'cuda:0'."
    )
    common.add_argument(
        "--use_gpu", type=lambda x: x.lower() == "true", default=True,
        help="Whether to use GPU if available (true/false)."
    )
    common.add_argument(
        "--save_all_model", type=lambda x: x.lower() == "true", default=True,
        help="Whether to save all intermediate global models."
    )
    common.add_argument(
        "--forget_client_idx", type=int, default=2,
        help="Client index to be forgotten. Set to -1 to skip."
    )


def _add_federaser_args(parser: argparse.ArgumentParser) -> None:
    """Register FedEraser-specific arguments."""
    fe = parser.add_argument_group("FedEraser Parameters")
    fe.add_argument(
        "--unlearn_interval", type=int, default=1,
        help="Model parameter save frequency (1 = every round)."
    )
    fe.add_argument(
        "--forget_local_epoch_ratio", type=float, default=0.5,
        help="Ratio of local epochs for direction estimation."
    )
    fe.add_argument(
        "--if_retrain", action="store_true",
        help="Run FL-Retrain as gold-standard baseline."
    )
    fe.add_argument(
        "--if_unlearning", action="store_true",
        help="Enable unlearning during global_train_once."
    )
    fe.add_argument(
        "--skip_mia", action="store_true",
        help="Skip Step4 Membership Inference Attack."
    )
    fe.add_argument(
        "--train_with_test", action="store_true",
        help="Include test set during training."
    )


def _add_backdoor_args(parser: argparse.ArgumentParser) -> None:
    """Register Backdoor Unlearning-specific arguments."""
    bd = parser.add_argument_group("Backdoor Unlearning Parameters")
    bd.add_argument(
        "--poison", action="store_true",
        help="Enable backdoor poisoning phase."
    )
    bd.add_argument(
        "--unlearn", action="store_true",
        help="Enable unlearning phase after poisoning."
    )
    bd.add_argument(
        "--poison_strategy", type=str, default="normal",
        help="Client selection strategy during poisoning."
    )
    bd.add_argument(
        "--poison_start_round", type=int, default=50,
        help="Round when poisoning starts."
    )
    bd.add_argument(
        "--poison_duration", type=int, default=50,
        help="Number of poisoning rounds."
    )
    bd.add_argument(
        "--unlearn_duration", type=int, default=10,
        help="Number of unlearning rounds."
    )
    bd.add_argument(
        "--attacker_id", type=int, default=42,
        help="ID of the attacking client."
    )
    bd.add_argument(
        "--target_class", type=int, default=0,
        help="Target class for backdoor attack."
    )
    bd.add_argument(
        "--num_poison", type=int, default=250,
        help="Number of poisoned samples."
    )
    bd.add_argument(
        "--fixed_frequency", type=int, default=5,
        help="Frequency of attacker client selection."
    )
    bd.add_argument(
        "--alpha_bd", type=float, default=1.0,
        help="Weight for total training loss (unlearning)."
    )
    bd.add_argument(
        "--beta", type=float, default=1.0,
        help="Scaling factor for penalty."
    )
    bd.add_argument(
        "--gamma_bd", type=float, default=3.0,
        help="Weight for backdoor loss."
    )


def _add_quickdrop_args(parser: argparse.ArgumentParser) -> None:
    """Register QuickDrop-specific arguments."""
    qd = parser.add_argument_group("QuickDrop Parameters")
    qd.add_argument(
        "--strategy", type=str, default="dilichlet",
        help="FL data distribution strategy."
    )
    qd.add_argument(
        "--env", type=str, default="",
        help="FL environment name (e.g., cifar10-seed42-u20-alpha0.1)."
    )
    qd.add_argument(
        "--env_path", type=str, default=None,
        help="Path to saved FL environments."
    )
    qd.add_argument(
        "--data_path", type=str, default="../../data",
        help="Path to dataset directory."
    )
    qd.add_argument(
        "--model", type=str, default="ConvNet",
        help="Model architecture name."
    )
    qd.add_argument(
        "--method_qd", type=str, default="DC",
        choices=["DC", "DSA"],
        help="Synthetic data method: DC or DSA."
    )
    qd.add_argument(
        "--lr_img", type=float, default=0.1,
        help="Learning rate for synthetic image optimization."
    )
    qd.add_argument(
        "--lr_net", type=float, default=0.01,
        help="Learning rate for network parameter optimization."
    )
    qd.add_argument(
        "--batch_real", type=int, default=256,
        help="Batch size for real data."
    )
    qd.add_argument(
        "--batch_train", type=int, default=256,
        help="Batch size for training networks."
    )
    qd.add_argument(
        "--init_qd", type=str, default="real",
        choices=["noise", "real"],
        help="Synthetic image initialization: noise or real."
    )
    qd.add_argument(
        "--dsa_strategy", type=str, default="None",
        help="Differentiable Siamese augmentation strategy."
    )
    qd.add_argument(
        "--dis_metric", type=str, default="ours",
        help="Distance metric for synthetic data matching."
    )
    qd.add_argument(
        "--scale", type=float, default=0.01,
        help="Synthetic images scale ratio per class."
    )
    qd.add_argument(
        "--directly_update", action="store_true",
        help="Update local model via synthetic loss directly."
    )
    qd.add_argument(
        "--affine_path", type=str, default="quickdrop-affine",
        help="Path to save affine dataset results."
    )
    qd.add_argument(
        "--momentum", type=float, default=0.9,
        help="Momentum for optimizer."
    )
    qd.add_argument(
        "--weight_decay", type=float, default=0.001,
        help="Weight decay for optimizer."
    )
    qd.add_argument(
        "--forgetting_epoch", type=int, default=0,
        help="Number of forgetting epochs."
    )
    qd.add_argument(
        "--forgetting_rate", type=float, default=0.0,
        help="Forgetting rate."
    )
    qd.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of DataLoader workers."
    )
    qd.add_argument(
        "--pin_memory", action="store_true",
        help="Pin memory for DataLoader."
    )
    qd.add_argument(
        "--persistent_workers", action="store_true",
        help="Keep DataLoader workers alive."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Config builder (factory)
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_data_name(raw: str) -> str:
    """Normalize dataset names to lowercase for consistency."""
    mapping = {
        "fashion-mnist": "fashion-mnist",
        "fashionmnist": "fashion-mnist",
        "cifar10": "cifar10",
        "cifar-10": "cifar10",
        "mnist": "mnist",
        "imagenet": "imagenet",
    }
    return mapping.get(raw.lower(), raw.lower())


def _extract_common_kwargs(args: argparse.Namespace) -> dict:
    """Extract BaseFLConfig constructor kwargs from parsed args."""
    forget_idx = args.forget_client_idx
    if forget_idx is not None and forget_idx < 0:
        forget_idx = None

    return {
        "data_name": _normalize_data_name(args.data_name),
        "n_clients": args.n_clients,
        "n_total_clients": args.n_total_clients,
        "global_epoch": args.global_epoch,
        "local_epoch": args.local_epoch,
        "local_lr": args.local_lr,
        "local_batch_size": args.local_batch_size,
        "test_batch_size": getattr(args, "test_batch_size", 64),
        "seed": args.seed,
        "device": args.device,
        "use_gpu": args.use_gpu,
        "save_all_model": args.save_all_model,
        "forget_client_idx": forget_idx,
    }


def build_config(args: argparse.Namespace) -> MethodConfig:
    """
    Build the appropriate config object based on --method.

    Args:
        args: Parsed command-line arguments (must contain --method).

    Returns:
        FedEraserConfig, BackdoorUnlearnConfig, or QuickDropConfig.

    Raises:
        ValueError: If --method is not one of the supported values.
    """
    method = args.method.lower()
    common_kwargs = _extract_common_kwargs(args)

    # ── FedEraser ────────────────────────────────────────────────────
    if method == "federaser":
        return FedEraserConfig(
            **common_kwargs,
            unlearn_interval=args.unlearn_interval,
            forget_local_epoch_ratio=args.forget_local_epoch_ratio,
            if_retrain=args.if_retrain,
            if_unlearning=args.if_unlearning,
            skip_mia=args.skip_mia,
            train_with_test=getattr(args, "train_with_test", False),
        )

    # ── Backdoor Unlearning ──────────────────────────────────────────
    elif method == "backdoor":
        alpha_val = getattr(args, "alpha_bd", getattr(args, "alpha", 1.0))
        gamma_val = getattr(args, "gamma_bd", getattr(args, "gamma", 3.0))
        return BackdoorUnlearnConfig(
            **common_kwargs,
            poison=args.poison,
            unlearn=args.unlearn,
            poison_strategy=args.poison_strategy,
            poison_start_round=args.poison_start_round,
            poison_duration=args.poison_duration,
            unlearn_duration=args.unlearn_duration,
            attacker_id=getattr(args, "attacker_id", 42),
            target_class=getattr(args, "target_class", 0),
            num_poison=getattr(args, "num_poison", 250),
            fixed_frequency=getattr(args, "fixed_frequency", 5),
            alpha=alpha_val,
            beta=args.beta,
            gamma=gamma_val,
        )

    # ── QuickDrop ────────────────────────────────────────────────────
    elif method == "quickdrop":
        init_val = getattr(args, "init_qd", getattr(args, "init", "real"))
        method_val = getattr(args, "method_qd", getattr(args, "method", "DC"))
        return QuickDropConfig(
            **common_kwargs,
            strategy=getattr(args, "strategy", "dilichlet"),
            env=getattr(args, "env", ""),
            env_path=getattr(args, "env_path", None),
            data_path=getattr(args, "data_path", "../../data"),
            model=getattr(args, "model", "ConvNet"),
            method=method_val,
            lr_img=getattr(args, "lr_img", 0.1),
            lr_net=getattr(args, "lr_net", 0.01),
            batch_real=getattr(args, "batch_real", 256),
            batch_train=getattr(args, "batch_train", 256),
            init=init_val,
            dsa_strategy=getattr(args, "dsa_strategy", "None"),
            dis_metric=getattr(args, "dis_metric", "ours"),
            scale=getattr(args, "scale", 0.01),
            directly_update=getattr(args, "directly_update", False),
            affine_path=getattr(args, "affine_path", "quickdrop-affine"),
            momentum=getattr(args, "momentum", 0.9),
            weight_decay=getattr(args, "weight_decay", 0.001),
            forgetting_epoch=getattr(args, "forgetting_epoch", 0),
            forgetting_rate=getattr(args, "forgetting_rate", 0.0),
            num_workers=getattr(args, "num_workers", 0),
            pin_memory=getattr(args, "pin_memory", False),
            persistent_workers=getattr(args, "persistent_workers", False),
        )

    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Supported methods: federaser, backdoor, quickdrop."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main parser entry point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv=None) -> MethodConfig:
    """
    Parse command-line arguments and return a method-specific config object.

    This is the single entry point for all three unlearning algorithms.
    The --method flag determines which config class is returned.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        FedEraserConfig, BackdoorUnlearnConfig, or QuickDropConfig.
    """
    parser = argparse.ArgumentParser(
        description="DDFed-FU: Unified Federated Unlearning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FedEraser - generic client unlearning
  %(prog)s --method federaser --data_name mnist --n_clients 20

  # Backdoor Unlearning - remove backdoor traces
  %(prog)s --method backdoor --data_name cifar10 --poison --unlearn

  # QuickDrop - synthetic-data-based fast forgetting
  %(prog)s --method quickdrop --dataset CIFAR10 --env myenv --scale 0.01
        """,
    )

    # ── Method selection ─────────────────────────────────────────────
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["federaser", "backdoor", "quickdrop"],
        help="Federated unlearning algorithm to use."
    )

    # ── Register all parameter groups ────────────────────────────────
    _add_common_args(parser)
    _add_federaser_args(parser)
    _add_backdoor_args(parser)
    _add_quickdrop_args(parser)

    args = parser.parse_args(argv if argv is not None else None)
    return build_config(args)


# ═══════════════════════════════════════════════════════════════════════════
# Direct invocation
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config = parse_args()
    print(f"Method: {type(config).__name__}")
    print(f"Config:\n")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")