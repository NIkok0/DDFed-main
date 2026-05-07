import re
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "ddfed_crypto" / "config.py"
OUT_DIR = ROOT / "benchmark_outputs"
CLIENT_COUNTS = [10, 20, 30, 40, 50]
ITERATIONS = 500

TMCFE_STAGE_ORDER = [
    "Setup",
    "SKDistribute",
    "DKGenerate",
    "Encrypt",
    "ShareDecrypt",
    "CombineDecrypt",
    "Total pipeline",
]
RODOT_STAGE_ORDER = [
    "Setup",
    "KGen",
    "DKShare",
    "Enc",
    "DKCom",
    "ParDec",
    "ComDec",
    "Total pipeline",
]


def update_config(content: str, values: dict) -> str:
    updated = content
    for key, value in values.items():
        updated = re.sub(rf"^{key}\s*=\s*.+$", f"{key} = {value}", updated, flags=re.MULTILINE)
    return updated


def run_script(script_name: str, env_override: Optional[dict] = None) -> str:
    script_path = ROOT / script_name
    run_env = {**__import__("os").environ, **(env_override or {})}
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=run_env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{script_name} failed: {proc.stderr[:800]}")
    return proc.stdout


def parse_last_ms_values(stdout_text: str, expected_count: int):
    vals = re.findall(r":\s*([0-9]+(?:\.[0-9]+)?)\s*ms", stdout_text, flags=re.MULTILINE)
    if len(vals) < expected_count:
        raise ValueError(f"Expected >= {expected_count} ms values, got {len(vals)}")
    return [float(v) for v in vals[-expected_count:]]


def parse_tmcfe(stdout_text: str):
    # Setup, SKDistribute, DKGenerate, Encrypt, ShareDecrypt, CombineDecrypt, Total
    vals = parse_last_ms_values(stdout_text, 7)
    labels = [
        "Setup",
        "SKDistribute",
        "DKGenerate",
        "Encrypt",
        "ShareDecrypt",
        "CombineDecrypt",
        "Total pipeline",
    ]
    return labels, vals


def parse_rodot(stdout_text: str):
    try:
        vals = [
            extract_value(stdout_text, r"1\. Setup"),
            extract_value(stdout_text, r"2\. KGen"),
            extract_value(stdout_text, r"3\. DKShare"),
            extract_value(stdout_text, r"4\. Enc"),
            extract_value(stdout_text, r"5\. DKCom"),
            extract_value(stdout_text, r"6\. ParDec"),
            extract_value(stdout_text, r"7\. ComDec"),
            extract_value(stdout_text, r"单次全流程流水线耗时总计"),
        ]
        labels = RODOT_STAGE_ORDER
        return labels, vals
    except ValueError:
        vals = parse_last_ms_values(stdout_text, 8)
        return RODOT_STAGE_ORDER, vals


def extract_value(text: str, prefix_regex: str) -> float:
    pattern = re.compile(prefix_regex + r"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Cannot parse metric with pattern: {prefix_regex}")
    return float(m.group(1))


def save_plot(df: pd.DataFrame, title: str, png_path: Path, stage_order: list[str]):
    pivot = df.pivot_table(index="stage", columns="n", values="time_ms", aggfunc="mean")
    pivot = pivot.reindex(stage_order)
    pivot = pivot.reindex(CLIENT_COUNTS, axis=1)

    plt.figure(figsize=(12, 6))
    x = CLIENT_COUNTS
    colors = ["#d7191c", "#2c7bb6", "#fdae61", "#1a9641", "#984ea3", "#00a6ca", "#000000", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    for idx, stage in enumerate(stage_order):
        if stage not in pivot.index:
            continue
        y = pivot.loc[stage].values.astype(float)
        plt.plot(
            x, y,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2.2,
            markersize=6,
            label=stage,
        )

    plt.title(title)
    plt.xlabel("number of clients")
    plt.ylabel("times")
    plt.xticks(CLIENT_COUNTS)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=8, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


def save_single_case_plot(df: pd.DataFrame, title: str, png_path: Path, stage_order: list[str]):
    df_plot = df.copy()
    df_plot["stage"] = pd.Categorical(df_plot["stage"], categories=stage_order, ordered=True)
    df_plot = df_plot.sort_values("stage")

    n = int(df_plot["n"].iloc[0])
    t = int(df_plot["t"].iloc[0])

    plt.figure(figsize=(10, 6))
    colors = ["#d7191c", "#2c7bb6", "#fdae61", "#1a9641", "#984ea3", "#00a6ca", "#000000", "#8c564b", "#7f7f7f"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    # Small x-jitter keeps multiple stage points visible at the same n.
    offsets = [(-0.32 + i * (0.64 / max(1, len(df_plot) - 1))) for i in range(len(df_plot))]

    for idx, (_, row) in enumerate(df_plot.iterrows()):
        x = n + offsets[idx]
        y = float(row["time_ms"])
        plt.scatter(
            [x],
            [y],
            s=70,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            label=str(row["stage"]),
            zorder=3,
        )
        plt.text(x, y, f"{y:.3f}", fontsize=8, ha="center", va="bottom")

    plt.title(title)
    plt.xlabel("number of clients")
    plt.ylabel("times")
    plt.xticks([n], [str(n)])
    plt.xlim(n - 1, n + 1)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=8, ncol=2, frameon=True, title=f"stages (n={n}, t={t})")
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


def run_scheme_matrix(scheme: str, script_name: str, original_config: str):
    rows = []
    for n in CLIENT_COUNTS:
        t = max(2, n // 2)
        config_text = update_config(
            original_config,
            {
                "N_ENCRYPTORS": n,
                "N_DECRYPTORS": n,
                "T_THRESHOLD": t,
            },
        )
        CONFIG_PATH.write_text(config_text, encoding="utf-8")

        env = {"NUM_ITERATIONS_OVERRIDE": str(ITERATIONS)}
        stdout_text = run_script(script_name, env_override=env)
        # log_path = OUT_DIR / f"{scheme}_n{n}_t{t}.log.txt"
        # log_path.write_text(stdout_text, encoding="utf-8")

        labels, values = parse_tmcfe(stdout_text) if scheme == "tmcfe" else parse_rodot(stdout_text)
        for stage, time_ms in zip(labels, values):
            rows.append({"scheme": scheme, "n": n, "t": t, "stage": stage, "time_ms": time_ms})
    return pd.DataFrame(rows)


def run_fixed_decryptor_matrix(
    scheme: str,
    script_name: str,
    n_decryptors: int,
    t_threshold: int,
    original_config: str,
):
    rows = []
    for n_encryptors in CLIENT_COUNTS:
        config_text = update_config(
            original_config,
            {
                "N_ENCRYPTORS": n_encryptors,
                "N_DECRYPTORS": n_decryptors,
                "T_THRESHOLD": t_threshold,
            },
        )
        CONFIG_PATH.write_text(config_text, encoding="utf-8")
        env = {"NUM_ITERATIONS_OVERRIDE": str(ITERATIONS)}
        stdout_text = run_script(script_name, env_override=env)
        # log_path = OUT_DIR / f"{scheme}_enc{n_encryptors}_n{n_decryptors}_t{t_threshold}.log.txt"
        # log_path.write_text(stdout_text, encoding="utf-8")
        labels, values = parse_tmcfe(stdout_text) if scheme == "tmcfe" else parse_rodot(stdout_text)
        for stage, time_ms in zip(labels, values):
            rows.append(
                {
                    "scheme": scheme,
                    "n": n_encryptors,
                    "n_decryptors": n_decryptors,
                    "t": t_threshold,
                    "stage": stage,
                    "time_ms": time_ms,
                }
            )
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original_config = CONFIG_PATH.read_text(encoding="utf-8")

    cases = [
        ("tmcfe", "ex_tmcfe_time.py", 5, 3, TMCFE_STAGE_ORDER),
        ("tmcfe", "ex_tmcfe_time.py", 10, 6, TMCFE_STAGE_ORDER),
        ("rodot_plus", "ex_rodot_plus_time.py", 5, 3, RODOT_STAGE_ORDER),
        ("rodot_plus", "ex_rodot_plus_time.py", 10, 6, RODOT_STAGE_ORDER),
    ]

    outputs = []

    try:
        for scheme, script, n, t, order in cases:
            print(f"[RUN] {scheme} (n={n}, t={t}), clients={CLIENT_COUNTS}")
            df = run_fixed_decryptor_matrix(scheme, script, n, t, original_config)
            csv_path = OUT_DIR / f"{scheme}_n{n}_t{t}_clients_10_50.csv"
            png_path = OUT_DIR / f"{scheme}_n{n}_t{t}_clients_10_50.png"
            df.to_csv(csv_path, index=False)
            save_plot(
                df,
                f"{scheme.upper()} times vs clients (n={n}, t={t})",
                png_path,
                order,
            )
            outputs.append((csv_path, png_path))
    finally:
        CONFIG_PATH.write_text(original_config, encoding="utf-8")
        print("[DONE] config.py restored to original values.")

    print("\n=== Generated files (4 figures) ===")
    for csv_path, png_path in outputs:
        print(f"CSV: {csv_path}")
        print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
