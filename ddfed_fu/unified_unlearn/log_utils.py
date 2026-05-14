"""Lightweight CSV logger for federated unlearning experiments."""

import csv
import os
import time


def timestamp() -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S]")


class Logger:
    """Writes one CSV row per round.

    Columns: round, phase, test_acc, test_loss, forget_acc, forget_loss,
             remain_acc, remain_loss
    """

    def __init__(self, csv_path: str):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.f = open(csv_path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "round", "phase",
            "test_acc", "test_loss",
            "forget_acc", "forget_loss",
            "remain_acc", "remain_loss",
        ])
        self.f.flush()

    def log_round(self, rnd: int, phase: str,
                  test_acc: float, test_loss: float,
                  forget_acc: float, forget_loss: float,
                  remain_acc: float, remain_loss: float,
                  cur: int = 0, total: int = 0):
        self.w.writerow([
            rnd, phase,
            round(test_acc, 6), round(test_loss, 6),
            round(forget_acc, 6), round(forget_loss, 6),
            round(remain_acc, 6), round(remain_loss, 6),
        ])
        self.f.flush()

        # ── Terminal output ──
        ts = timestamp()
        if total > 0:
            pct = cur / total * 100
            prog = f"{pct:5.1f}% ({cur}/{total})"
        else:
            prog = ""
        line = (f"{ts} [{prog}] [Round {rnd:3d} | {phase:>14s}] "
                f"test_acc={test_acc:.4f}  test_loss={test_loss:.4f}")
        if phase != "pretrain":
            line += (f"  forget_acc={forget_acc:.4f}  forget_loss={forget_loss:.4f}"
                     f"  remain_acc={remain_acc:.4f}  remain_loss={remain_loss:.4f}")
        print(line, flush=True)

    def close(self):
        self.f.close()