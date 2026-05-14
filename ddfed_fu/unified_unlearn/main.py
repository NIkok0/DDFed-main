#!/usr/bin/env python3
"""Main entry point for a single federated unlearning experiment.

Usage:
    python -m ddfed_fu.unified_unlearn.main \\
        --dataset FashionMNIST --alpha 0.1 --algo neurotoxin \\
        --pretrain 30 --unlearn 20

Specifying 5 algorithms + 2 datasets + 2 alphas = 20 experiment combos.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_unlearn.config import (
    ALGORITHMS, PRETRAIN_ROUNDS, UNLEARN_ROUNDS,
    GLOBAL_SEED, DEVICE_STR,
)
from unified_unlearn.server import run_server


def main():
    parser = argparse.ArgumentParser(description="Unified Federated Unlearning")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["FashionMNIST", "CIFAR10"],
                        help="Dataset name")
    parser.add_argument("--alpha", type=float, required=True,
                        help="Dirichlet concentration (e.g. 100.0 or 0.1)")
    parser.add_argument("--algo", type=str, required=True,
                        choices=list(ALGORITHMS.keys()) + list(ALGORITHMS.values()),
                        help="Algorithm key (e.g. 'Baseline_FedSGA') or mode string "
                             "(e.g. 'fedsga', 'neurotoxin', 'quickdrop', "
                             "'ddfu_client', 'ddfu_sample').")
    parser.add_argument("--pretrain", type=int, default=PRETRAIN_ROUNDS)
    parser.add_argument("--unlearn", type=int, default=UNLEARN_ROUNDS)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--no-dsa", action="store_true",
                        help="Disable DSA augmentation in FedQuickDrop")
    parser.add_argument("--force-pretrain", action="store_true",
                        help="Re-run pretrain phase even if checkpoint exists")
    args = parser.parse_args()

    # Resolve algorithm name
    if args.algo in ALGORITHMS:
        algo_key = args.algo
        mode = ALGORITHMS[args.algo]
    else:
        # It's a mode string
        mode = args.algo
        # reverse lookup
        algo_key = next((k for k, v in ALGORITHMS.items() if v == mode), mode)

    # Override globals if needed
    import unified_unlearn.config as cfg
    cfg.GLOBAL_SEED = args.seed
    cfg.PRETRAIN_ROUNDS = args.pretrain
    cfg.UNLEARN_ROUNDS = args.unlearn

    print(f"{'='*60}")
    print(f"  Dataset: {args.dataset}  α={args.alpha}")
    print(f"  Algorithm: {algo_key}  (mode={mode})")
    print(f"  Pretrain: {args.pretrain} rounds  |  Unlearn: {args.unlearn} rounds")
    print(f"  Seed: {args.seed}  |  Device: {DEVICE_STR}")
    print(f"{'='*60}")

    t0 = time.time()
    model, forget_uid = run_server(
        dataset=args.dataset,
        alpha=args.alpha,
        algo=mode,
        pretrain_rounds=args.pretrain,
        unlearn_rounds=args.unlearn,
        dsa=not args.no_dsa,
        resume_phase1=not args.force_pretrain,
    )
    elapsed = time.time() - t0

    from unified_unlearn.config import result_csv_path
    csv_f = result_csv_path(args.dataset, args.alpha, mode)
    print(f"✓ Experiment completed in {elapsed:.0f}s")
    print(f"  Forget user: #{forget_uid}")
    print(f"  Results:     {csv_f}")


if __name__ == "__main__":
    main()