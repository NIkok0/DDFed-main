# -*- coding: utf-8 -*-
"""
DDFed-FU Unified Configuration Layer
Base class + method-specific configs for FedEraser, Backdoor Unlearning, and QuickDrop.

All three algorithms share common FL parameters defined in BaseFLConfig,
while each algorithm extends it with its own unique parameters.

Usage:
    from ddfed_fu.common import FedEraserConfig
    config = FedEraserConfig(data_name="mnist", n_clients=20, forget_client_idx=3)
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class BaseFLConfig:
    """
    Common federated learning parameters shared by all three unlearning algorithms.
    All subclasses inherit these fields automatically.
    """

    # ── Dataset ──────────────────────────────────────────────────────────
    data_name: str = "mnist"
    """Dataset name: mnist, fashion-mnist, cifar10, imagenet, purchase, adult"""

    # ── Clients ──────────────────────────────────────────────────────────
    n_clients: int = 10
    """Number of participating clients per round"""

    n_total_clients: int = 100
    """Total number of clients in the federated learning system"""

    # ── Training ─────────────────────────────────────────────────────────
    global_epoch: int = 20
    """Number of global communication rounds (multi-round training)"""

    local_epoch: int = 10
    """Number of local training epochs per round"""

    local_lr: float = 0.005
    """Local learning rate"""

    local_batch_size: int = 64
    """Local batch size for client training"""

    test_batch_size: int = 64
    """Batch size for evaluation"""

    # ── Reproducibility ──────────────────────────────────────────────────
    seed: int = 1
    """Random seed for reproducibility"""

    # ── Device ───────────────────────────────────────────────────────────
    device: str = "cpu"
    """Device: 'auto', 'cpu', 'cuda', or 'cuda:0' etc."""

    use_gpu: bool = True
    """Whether to use GPU if available"""

    save_all_model: bool = True
    """Whether to save all intermediate global models during training"""

    # ── Unlearning Common ────────────────────────────────────────────────
    forget_client_idx: Optional[int] = 2
    """Client index to be forgotten. Set to None to skip unlearning."""

    # ── Computed properties ──────────────────────────────────────────────

    @property
    def resolved_device(self) -> str:
        """Resolve the actual torch device string."""
        if self.device != "auto":
            return self.device
        return "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"

    def to_dict(self) -> dict:
        """Export all config fields as a flat dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            result[field_name] = getattr(self, field_name)
        result["resolved_device"] = self.resolved_device
        return result


# ═══════════════════════════════════════════════════════════════════════════
# FedEraser 特有配置
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FedEraserConfig(BaseFLConfig):
    """
    Configuration for the FedEraser federated unlearning algorithm.

    FedEraser removes a target client's influence using calibration
    with historical model parameters.

    Extra Parameters (beyond BaseFLConfig):
        unlearn_interval: Model parameter save frequency (1 = every round)
        forget_local_epoch_ratio: Ratio of local epochs used for direction estimation
        if_retrain: Whether to also run FL-Retrain as a gold-standard baseline
        if_unlearning: Whether to enable unlearning during global_train_once
        skip_mia: Skip Step4 membership inference attack (for smoke tests)
        train_with_test: Whether to include test set during training
    """

    # ── FedEraser Unlearning Settings ────────────────────────────────────
    unlearn_interval: int = 1
    """
    How many rounds the model parameters are saved.
    1 = save parameters every round (N_itv in the paper).
    """

    forget_local_epoch_ratio: float = 0.5
    """
    When a user is selected to be forgotten, other users need to train
    several rounds locally to determine the general direction of model
    convergence.  forget_local_epoch_ratio * local_epoch is the number
    of local training rounds for direction estimation.
    """

    if_retrain: bool = False
    """
    If True, the global model is retrained using FL-Retrain,
    discarding data from the forget_client_idx.
    """

    if_unlearning: bool = False
    """
    If False, global_train_once will NOT skip the forgotten user.
    If True, global_train_once skips the forgotten user during training.
    """

    skip_mia: bool = False
    """Skip Step4 Membership Inference Attack (for smoke tests / XGBoost native crashes)."""

    train_with_test: bool = False
    """Whether to include test set during training."""

    # ── Post-init compatibility aliases ──────────────────────────────
    def __post_init__(self):
        """Create backward-compatible attribute aliases for legacy code."""
        # Legacy code uses N_total_client / N_client
        object.__setattr__(self, 'N_total_client', self.n_total_clients)
        object.__setattr__(self, 'N_client', self.n_clients)
        # Legacy code checks cuda_state
        object.__setattr__(self, 'cuda_state', torch.cuda.is_available())


# ═══════════════════════════════════════════════════════════════════════════
# Backdoor Unlearning (Neurotoxin) 特有配置
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BackdoorUnlearnConfig(BaseFLConfig):
    """
    Configuration for the Federated Backdoor Unlearning algorithm (Neurotoxin).

    Three-phase process:
      1. Normal FL training
      2. Backdoor poisoning (attacker injects backdoor)
      3. Unlearning (erase backdoor traces)

    Extra Parameters (beyond BaseFLConfig):
        poison: Enable backdoor poisoning phase
        unlearn: Enable unlearning phase
        poison_strategy: Client selection strategy during poisoning
        poison_start_round: Round when poisoning starts
        poison_duration: Number of poisoning rounds
        unlearn_duration: Number of unlearning rounds
        attacker_id: ID of the attacking client
        target_class: Target class for the backdoor attack
        num_poison: Number of poisoned samples
        fixed_frequency: Frequency at which attacker client is selected
        alpha: Weight for total training loss in unlearning
        beta: Scaling factor for penalty in unlearning
        gamma: Weight for backdoor loss in unlearning
    """

    # ── Phase Control ────────────────────────────────────────────────────
    poison: bool = False
    """Enable backdoor poisoning during training."""

    unlearn: bool = False
    """Enable unlearning after poisoning."""

    # ── Poisoning Strategy ───────────────────────────────────────────────
    poison_strategy: str = "normal"
    """Strategy used for backdoor poisoning client selection."""

    poison_start_round: int = 50
    """The round number when backdoor poisoning should start."""

    poison_duration: int = 50
    """Number of rounds for which the backdoor poisoning is active."""

    # ── Unlearning Parameters ────────────────────────────────────────────
    unlearn_duration: int = 10
    """Number of rounds for which unlearning is applied after poisoning."""

    # ── Attack Configuration ─────────────────────────────────────────────
    attacker_id: int = 42
    """The ID of the client designated as the attacker."""

    target_class: int = 0
    """The target class for the backdoor attack."""

    num_poison: int = 250
    """Number of poisoned samples introduced into the training process."""

    fixed_frequency: int = 5
    """Frequency at which the attacker client is selected."""

    # ── Unlearning Loss Weights ─────────────────────────────────────────
    alpha: float = 1.0
    """Weight for total training loss in unlearning."""

    beta: float = 1.0
    """Scaling factor for the penalty in unlearning."""

    gamma: float = 3.0
    """Weight for the backdoor loss in unlearning."""

    # ── Post-init compatibility aliases ──────────────────────────────
    def __post_init__(self):
        """Create backward-compatible attribute aliases for legacy code."""
        # Legacy config.py uses num_clients / num_selected / batch_size
        object.__setattr__(self, 'num_clients', self.n_total_clients)
        object.__setattr__(self, 'num_selected', self.n_clients)
        object.__setattr__(self, 'batch_size', self.local_batch_size)


# ═══════════════════════════════════════════════════════════════════════════
# QuickDrop 特有配置
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QuickDropConfig(BaseFLConfig):
    """
    Configuration for the QuickDrop federated unlearning algorithm.

    QuickDrop uses synthetic data distillation (DC/DSA) and affine dataset
    operations to quickly remove a client's influence.

    Extra Parameters (beyond BaseFLConfig):
        strategy: FL data distribution strategy
        env: FL environment name (e.g., 'cifar10-seed42-u20-alpha0.1')
        env_path: Path to saved FL environments
        affine_path: Path to save affine results
        method: Synthetic data method ('DC' or 'DSA')
        lr_img: Learning rate for synthetic image optimization
        lr_net: Learning rate for network parameter optimization
        batch_real: Batch size for real data
        batch_train: Batch size for training
        init: Synthetic image initialization ('noise' or 'real')
        dsa_strategy: Differentiable Siamese augmentation strategy
        dis_metric: Distance metric for synthetic data
        scale: Synthetic images scale ratio per class
        directly_update: Whether to update via synthetic loss directly
        momentum: Momentum for optimizer
        weight_decay: Weight decay for optimizer
        forgetting_epoch: Number of forgetting epochs
        forgetting_rate: Forgetting rate
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory for DataLoader
        persistent_workers: Whether to keep DataLoader workers alive
    """

    # ── Environment ──────────────────────────────────────────────────────
    strategy: str = "dilichlet"
    """FL data distribution strategy."""

    env: str = ""
    """FL environment name, e.g., 'cifar10-seed42-u20-alpha0.1'."""

    env_path: Optional[str] = None
    """Path to the saved FL environments directory."""

    data_path: str = "../../data"
    """Path to the dataset directory."""

    # ── Model Architecture ───────────────────────────────────────────────
    model: str = "ConvNet"
    """Model architecture name."""

    # ── Synthetic Data Distillation ──────────────────────────────────────
    method: str = "DC"
    """Synthetic data method: 'DC' (DeepInversion-style) or 'DSA'."""

    lr_img: float = 0.1
    """Learning rate for updating synthetic images."""

    lr_net: float = 0.01
    """Learning rate for updating network parameters during distillation."""

    batch_real: int = 256
    """Batch size for real data during distillation."""

    batch_train: int = 256
    """Batch size for training networks."""

    init: str = "real"
    """
    Synthetic image initialization:
    'noise' = from random noise,
    'real' = from randomly sampled real images.
    """

    dsa_strategy: str = "None"
    """Differentiable Siamese augmentation strategy."""

    dis_metric: str = "ours"
    """Distance metric for synthetic data matching."""

    scale: float = 0.01
    """
    For each class: #synthetic images = #original images * scale.
    """

    directly_update: bool = False
    """
    True: update local model via synthetic loss directly.
    False: recalculate loss on synthetic dataset.
    """

    # ── Affine Dataset ───────────────────────────────────────────────────
    affine_path: str = "quickdrop-affine"
    """Path to save affine dataset results."""

    # ── Optimizer ────────────────────────────────────────────────────────
    momentum: float = 0.9
    """Momentum for the optimizer."""

    weight_decay: float = 0.001
    """Weight decay for the optimizer."""

    # ── Forgetting ───────────────────────────────────────────────────────
    forgetting_epoch: int = 0
    """Number of forgetting epochs for SGA-style forgetting."""

    forgetting_rate: float = 0.0
    """Forgetting rate."""

    # ── DataLoader ───────────────────────────────────────────────────────
    num_workers: int = 0
    """Number of DataLoader workers."""

    pin_memory: bool = False
    """Whether to pin memory for DataLoader."""

    persistent_workers: bool = False
    """Whether to keep DataLoader workers alive between iterations."""

    # ── Post-init ───────────────────────────────────────────────────────
    def __post_init__(self):
        """Set derived attributes after initialization."""
        # QuickDrop's server code uses 'num_selected' instead of 'n_clients'
        # and 'communication_round' instead of 'global_epoch'
        # These aliases are created for backward compatibility.
        object.__setattr__(self, 'num_selected', self.n_clients)
        object.__setattr__(self, 'communication_round', self.global_epoch)
        object.__setattr__(self, 'learning_rate', self.local_lr)
        object.__setattr__(self, 'batch_size', self.local_batch_size)