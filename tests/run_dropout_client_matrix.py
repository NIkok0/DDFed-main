import re
import sys
import subprocess
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "ddfed_crypto" / "config.py"
OUT_DIR = ROOT / "benchmark_outputs" / "dropout_client_matrix"

CLIENT_COUNTS = [10, 20, 30, 40, 50]
DROPOUT_RATES = [0.10, 0.20, 0.30]
DMCFE_ITER = 1
RODOT_ITER = 500

DMCFE_STAGE_ORDER = [
    "Global Setup",
    "Client Setup",
    "Client: Step 1 Agree on weight Y",
    "Client: Step 2 Key Sharing",
    "Client: Step 3 Encryption",
    "Server: Step 1 Verify weight Y",
    "Server: Step 4 Aggregation",
    "Client Aggregation Phase Total",
    "Server Aggregation Phase Total",
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


def normalize_stage_name(stage: str) -> str:
    mapping = {
        "客户端 (Client) Aggregation Phase 纯计算负担": "Client Aggregation Phase Total",
        "服务器 (Server) Aggregation Phase 纯计算负担": "Server Aggregation Phase Total",
    }
    return mapping.get(stage, stage)


def ordered_pivot(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    df2 = df.copy()
    df2["stage"] = df2["stage"].map(normalize_stage_name)
    pivot = df2.pivot_table(index="stage", columns="clients", values="time_ms", aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    order = DMCFE_STAGE_ORDER if scheme == "dmcfe_ip" else RODOT_STAGE_ORDER
    present_order = [s for s in order if s in pivot.index]
    return pivot.reindex(present_order)


def set_config_values(content: str, kv: dict) -> str:
    updated = content
    for key, value in kv.items():
        updated = re.sub(rf"^{key}\s*=\s*.+$", f"{key} = {value}", updated, flags=re.MULTILINE)
    return updated


def run_python(script_rel: str, extra_env: dict) -> str:
    script_path = ROOT / script_rel
    env = dict(**extra_env)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**__import__("os").environ, **env},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{script_rel} failed.\nSTDERR:\n{proc.stderr[:1000]}")
    return proc.stdout


def pick_ms(text: str, label_regex: str) -> float:
    m = re.search(label_regex + r"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Cannot parse label: {label_regex}")
    return float(m.group(1))


def parse_dmcfe(text: str) -> dict:
    agree = pick_ms(text, r"Step 1: Agree on weight Y")
    share = pick_ms(text, r"Step 2: Key Sharing")
    enc = pick_ms(text, r"Step 3: Encryption")
    verify = pick_ms(text, r"Step 1: Verify weight Y")
    agg = pick_ms(text, r"Step 4: Aggregation")
    return {
        "Global Setup": pick_ms(text, r"Global Setup"),
        "Client Setup": pick_ms(text, r"Client Setup"),
        "Client: Step 1 Agree on weight Y": agree,
        "Client: Step 2 Key Sharing": share,
        "Client: Step 3 Encryption": enc,
        "Server: Step 1 Verify weight Y": verify,
        "Server: Step 4 Aggregation": agg,
        "Client Aggregation Phase Total": agree + share + enc,
        "Server Aggregation Phase Total": verify + agg,
    }


def parse_rodot(text: str) -> dict:
    try:
        return {
            "Setup": pick_ms(text, r"1\. Setup"),
            "KGen": pick_ms(text, r"2\. KGen"),
            "DKShare": pick_ms(text, r"3\. DKShare"),
            "Enc": pick_ms(text, r"4\. Enc"),
            "DKCom": pick_ms(text, r"5\. DKCom"),
            "ParDec": pick_ms(text, r"6\. ParDec"),
            "ComDec": pick_ms(text, r"7\. ComDec"),
            "Total pipeline": pick_ms(text, r"单次全流程流水线耗时总计"),
        }
    except ValueError:
        vals = re.findall(r":\s*([0-9]+(?:\.[0-9]+)?)\s*ms", text, flags=re.MULTILINE)
        if len(vals) < 8:
            raise ValueError("Cannot parse Rodot+ metrics from output.")
        setup, kgen, dkshare, enc, dkcom, pardec, comdec, total = [float(v) for v in vals[-8:]]
        return {
            "Setup": setup,
            "KGen": kgen,
            "DKShare": dkshare,
            "Enc": enc,
            "DKCom": dkcom,
            "ParDec": pardec,
            "ComDec": comdec,
            "Total pipeline": total,
        }


def save_curve_png(df_pivot: pd.DataFrame, title: str, out_png: Path):
    plt.figure(figsize=(12, 6))
    x = [int(c) for c in df_pivot.columns]
    for idx, row in enumerate(df_pivot.index):
        y = df_pivot.loc[row].values.astype(float)
        plt.plot(x, y, marker="o", linewidth=2.0, label=str(row))
        plt.text(x[-1] + 0.4, y[-1], str(idx + 1), fontsize=8)
    plt.title(title)
    plt.xlabel("Number of clients")
    plt.ylabel("times")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xlim(min(x) - 1, max(x) + 4)
    plt.legend(fontsize=8, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def refresh_existing_pngs():
    for scheme in ["dmcfe_ip", "rodot_plus"]:
        for rate in DROPOUT_RATES:
            base = f"{scheme}_drop{int(rate*100)}"
            table_csv = OUT_DIR / f"{base}_table.csv"
            png = OUT_DIR / f"{base}.png"
            if not table_csv.exists():
                continue
            table_df = pd.read_csv(table_csv)
            stage_col = table_df.columns[0]
            long_df = table_df.melt(id_vars=[stage_col], var_name="clients", value_name="time_ms")
            long_df = long_df.rename(columns={stage_col: "stage"})
            long_df["clients"] = long_df["clients"].astype(int)
            long_df["scheme"] = scheme
            pivot = ordered_pivot(long_df, scheme)
            pivot.to_csv(table_csv, encoding="utf-8-sig")
            save_curve_png(pivot, f"{scheme.upper()} ({int(rate*100)}% dropout)", png)
            print(f"[REFRESH] {png.name}")


def run_scheme_matrix(scheme: str, drop_rate: float, original_config: str):
    rows = []
    for n in CLIENT_COUNTS:
        t = max(2, n // 2)
        drop_total = int(round(n * drop_rate))
        drop_total = min(drop_total, n - 1)

        if scheme == "rodot_plus":
            drop_k = drop_total // 2
            drop_m = drop_total - drop_k
            kv = {
                "N_ENCRYPTORS": n,
                "N_DECRYPTORS": 5,
                "T_THRESHOLD": 3,
                "N_DROPPED_K": drop_k,
                "N_DROPPED_M": drop_m,
                "N_DROPPED_D": 0,
            }
            script = "ex_rodot_plus_time.py"
            env = {"NUM_ITERATIONS_OVERRIDE": str(RODOT_ITER)}
        else:
            kv = {
                "N_ENCRYPTORS": n,
                "N_DECRYPTORS": n,
                "T_THRESHOLD": t,
            }
            script = "ex_dmcfe_ip_time.py"
            env = {
                "NUM_BENCHMARK_ITERATIONS": str(DMCFE_ITER),
                "DROP_COUNT_OVERRIDE": str(drop_total),
            }

        new_config = set_config_values(original_config, kv)
        CONFIG_PATH.write_text(new_config, encoding="utf-8")
        print(f"[RUN] {scheme} drop={int(drop_rate*100)}% clients number={n} threshold={kv['T_THRESHOLD']} drop={drop_total}")
        out = run_python(script, env)

        metrics = parse_rodot(out) if scheme == "rodot_plus" else parse_dmcfe(out)
        for stage, ms in metrics.items():
            rows.append(
                {
                    "scheme": scheme,
                    "drop_rate": int(drop_rate * 100),
                    "clients": n,
                    "stage": stage,
                    "time_ms": ms,
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-only", action="store_true", help="Only refresh existing PNG/table styles.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh_only:
        refresh_existing_pngs()
        return

    original_config = CONFIG_PATH.read_text(encoding="utf-8")

    all_frames = []
    try:
        for scheme in ["dmcfe_ip", "rodot_plus"]:
            for rate in DROPOUT_RATES:
                base = f"{scheme}_drop{int(rate*100)}"
                long_csv = OUT_DIR / f"{base}_long.csv"
                table_csv = OUT_DIR / f"{base}_table.csv"
                png = OUT_DIR / f"{base}.png"
                if long_csv.exists() and table_csv.exists() and png.exists():
                    print(f"[SKIP] {base} already exists.")
                    continue
                df = run_scheme_matrix(scheme, rate, original_config)
                all_frames.append(df)
                pivot = ordered_pivot(df, scheme)
                df.to_csv(long_csv, index=False)
                pivot.to_csv(table_csv, encoding="utf-8-sig")
                save_curve_png(pivot, f"{scheme.upper()} ({int(rate*100)}% dropout)", png)
                print(f"[OK] {base}: {table_csv.name}, {png.name}")
    finally:
        CONFIG_PATH.write_text(original_config, encoding="utf-8")
        print("[DONE] config.py restored.")

    merged = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if not merged.empty:
        merged.to_csv(OUT_DIR / "all_results_long.csv", index=False)
        print("[DONE] all_results_long.csv generated.")


if __name__ == "__main__":
    main()
