import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "ddfed_crypto" / "config.py"
CLIENT_COUNTS = [10, 20, 30, 40, 50]
ITERATIONS = 500

DMCFE_STAGE_ORDER = [
    "Global Setup",
    "Client Setup",
    "Agree on weight Y",
    "Key Sharing",
    "Encryption",
    "Verify weight Y",
    "Aggregation",
    "Client aggregation total",
    "Server aggregation total",
]
DDMCFE_STAGE_ORDER = [
    "Setup",
    "KeyGen",
    "Encrypt",
    "DKGenShare",
    "DKComb",
    "Decrypt",
    "Total pipeline",
]


def update_config(content: str, values: dict) -> str:
    updated = content
    for key, value in values.items():
        updated = re.sub(rf"^{key}\s*=\s*.+$", f"{key} = {value}", updated, flags=re.MULTILINE)
    return updated


def run_script(script_name: str, env_override: Optional[dict] = None):
    script_path = ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\n===== Running {script_name} =====")
    run_env = {**os.environ, **(env_override or {})}
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=run_env,
    )
    # Avoid terminal encoding issues on Windows consoles (e.g., GBK).
    print(f"{script_name} finished, captured {len(proc.stdout)} chars of output.")
    if proc.returncode != 0:
        print(f"{script_name} stderr length: {len(proc.stderr)}")
    return proc.returncode, proc.stdout, proc.stderr


def extract_value(text: str, prefix_regex: str) -> float:
    pattern = re.compile(prefix_regex + r"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Cannot parse metric with pattern: {prefix_regex}")
    return float(m.group(1))


def parse_dmcfe_ip(stdout_text: str) -> pd.DataFrame:
    agree = extract_value(stdout_text, r"Step 1: Agree on weight Y")
    share = extract_value(stdout_text, r"Step 2: Key Sharing")
    enc = extract_value(stdout_text, r"Step 3: Encryption")
    verify = extract_value(stdout_text, r"Step 1: Verify weight Y")
    agg = extract_value(stdout_text, r"Step 4: Aggregation")
    metrics = {
        "Global Setup": extract_value(stdout_text, r"Global Setup"),
        "Client Setup": extract_value(stdout_text, r"Client Setup"),
        "Agree on weight Y": agree,
        "Key Sharing": share,
        "Encryption": enc,
        "Verify weight Y": verify,
        "Aggregation": agg,
        "Client aggregation total": agree + share + enc,
        "Server aggregation total": verify + agg,
    }
    return pd.DataFrame([{"stage": k, "time_ms": v} for k, v in metrics.items()])


def parse_ddmcfe(stdout_text: str) -> pd.DataFrame:
    # Prefer label-based extraction; fallback to ordered extraction for mojibake output.
    try:
        setup = extract_value(stdout_text, r"1\. Setup")
        keygen = extract_value(stdout_text, r"2\. KeyGen")
        encrypt = extract_value(stdout_text, r"3\. Encrypt")
        dkgenshare = extract_value(stdout_text, r"4\. DKGenShare")
        dkcomb = extract_value(stdout_text, r"5\. DKComb")
        decrypt = extract_value(stdout_text, r"6\. Decrypt")
    except ValueError:
        vals = re.findall(r":\s*([0-9]+(?:\.[0-9]+)?)\s*ms", stdout_text, flags=re.MULTILINE)
        if len(vals) < 6:
            raise ValueError("Cannot parse DDMCFE metrics from output.")
        setup, keygen, encrypt, dkgenshare, dkcomb, decrypt = [float(v) for v in vals[-7:-1]] if len(vals) >= 7 else [float(v) for v in vals[-6:]]
    metrics = {
        "Setup": setup,
        "KeyGen": keygen,
        "Encrypt": encrypt,
        "DKGenShare": dkgenshare,
        "DKComb": dkcomb,
        "Decrypt": decrypt,
        "Total pipeline": setup + keygen + encrypt + dkgenshare + dkcomb + decrypt,
    }
    return pd.DataFrame([{"stage": k, "time_ms": v} for k, v in metrics.items()])


def save_curve_plot(df: pd.DataFrame, title: str, out_png: Path, stage_order: list[str]):
    pivot = df.pivot_table(index="stage", columns="clients", values="time_ms", aggfunc="mean")
    pivot = pivot.reindex(stage_order)
    pivot = pivot.reindex(CLIENT_COUNTS, axis=1)

    plt.figure(figsize=(12, 6))
    colors = ["#d7191c", "#2c7bb6", "#fdae61", "#1a9641", "#984ea3", "#00a6ca", "#000000", "#8c564b", "#7f7f7f"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]
    x = CLIENT_COUNTS
    for idx, stage in enumerate(stage_order):
        if stage not in pivot.index:
            continue
        y = pivot.loc[stage].values.astype(float)
        plt.plot(
            x,
            y,
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
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_failure_plot(message: str, title: str, out_png: Path):
    plt.figure(figsize=(12, 4))
    plt.axis("off")
    plt.title(title)
    plt.text(0.02, 0.6, "Benchmark failed in current environment.", fontsize=12, weight="bold")
    plt.text(0.02, 0.4, message[:500], fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    out_dir = ROOT / "benchmark_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    original_config = CONFIG_PATH.read_text(encoding="utf-8")

    dmcfe_rows = []
    ddmcfe_rows = []

    try:
        for n in CLIENT_COUNTS:
            t = max(2, n // 2)
            cfg = update_config(
                original_config,
                {
                    "N_ENCRYPTORS": n,
                    "N_DECRYPTORS": n,
                    "T_THRESHOLD": t,
                },
            )
            CONFIG_PATH.write_text(cfg, encoding="utf-8")
            print(f"\n[RUN] clients={n}, threshold={t}")

            dmcfe_code, dmcfe_stdout, dmcfe_stderr = run_script(
                "ex_dmcfe_ip_time.py",
                env_override={"NUM_BENCHMARK_ITERATIONS": str(ITERATIONS)},
            )
            ddmcfe_code, ddmcfe_stdout, ddmcfe_stderr = run_script(
                "ex_ddmcfe_time.py",
                env_override={"NUM_ITERATIONS_OVERRIDE": str(ITERATIONS)},
            )

            if dmcfe_code != 0:
                raise RuntimeError(f"ex_dmcfe_ip_time.py failed at n={n}: {dmcfe_stderr[:400]}")
            if ddmcfe_code != 0:
                raise RuntimeError(f"ex_ddmcfe_time.py failed at n={n}: {ddmcfe_stderr[:400]}")

            dmcfe_df_n = parse_dmcfe_ip(dmcfe_stdout)
            dmcfe_df_n.insert(0, "clients", n)
            dmcfe_df_n.insert(1, "threshold", t)
            dmcfe_rows.append(dmcfe_df_n)

            ddmcfe_df_n = parse_ddmcfe(ddmcfe_stdout)
            ddmcfe_df_n.insert(0, "clients", n)
            ddmcfe_df_n.insert(1, "threshold", t)
            ddmcfe_rows.append(ddmcfe_df_n)
    finally:
        CONFIG_PATH.write_text(original_config, encoding="utf-8")
        print("[DONE] config.py restored.")

    dmcfe_df = pd.concat(dmcfe_rows, ignore_index=True)
    ddmcfe_df = pd.concat(ddmcfe_rows, ignore_index=True)

    dmcfe_csv = out_dir / "ex_dmcfe_ip_time_clients_10_50.csv"
    ddmcfe_csv = out_dir / "ex_ddmcfe_time_clients_10_50.csv"
    dmcfe_png = out_dir / "ex_dmcfe_ip_time_clients_10_50.png"
    ddmcfe_png = out_dir / "ex_ddmcfe_time_clients_10_50.png"
    merged_csv = out_dir / "ex_dmcfe_ddmcfe_clients_10_50_all.csv"

    dmcfe_df.to_csv(dmcfe_csv, index=False)
    ddmcfe_df.to_csv(ddmcfe_csv, index=False)
    pd.concat([dmcfe_df.assign(scheme="dmcfe_ip"), ddmcfe_df.assign(scheme="ddmcfe")], ignore_index=True).to_csv(
        merged_csv, index=False
    )

    save_curve_plot(dmcfe_df, "DMCFE-IP times vs number of clients", dmcfe_png, DMCFE_STAGE_ORDER)
    save_curve_plot(ddmcfe_df, "DDMCFE times vs number of clients", ddmcfe_png, DDMCFE_STAGE_ORDER)

    print("\n===== Export Completed =====")
    print(f"DMCFE-IP CSV: {dmcfe_csv}")
    print(f"DMCFE-IP PNG: {dmcfe_png}")
    print(f"DDMCFE CSV: {ddmcfe_csv}")
    print(f"DDMCFE PNG: {ddmcfe_png}")
    print(f"ALL CSV: {merged_csv}")


if __name__ == "__main__":
    main()
