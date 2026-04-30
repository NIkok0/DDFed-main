import csv
import os
import subprocess
import sys
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
ALLOCATOR_SCRIPT = PROJECT_ROOT / "env_generator" / "dilichlet_allocator" / "dilichlet_allocator.py"
FEDQUICKDROP_SCRIPT = PROJECT_ROOT / "FedQuickDrop" / "server" / "server_fedquickdrop.py"
RESULT_FIGURE = PROJECT_ROOT / "result" / "accuracy_vs_clients.png"

STRICT_MODE = True
CLIENTS_LIST = [10, 20]
ALPHA = 0.1
SEED = 42
COMMUNICATION_ROUND = 1
LOCAL_EPOCH = 1
MODEL = "ConvNet"
PREFERRED_DEVICE = "cuda:0"

EXPERIMENT_DATASETS = [
    {"label": "fashionmnist", "dataset_name": "FashionMNIST"},
    {"label": "cifar10", "dataset_name": "CIFAR10"},
]


def resolve_device(preferred_device: str) -> str:
    if not preferred_device.startswith("cuda"):
        return preferred_device

    try:
        import torch  # pylint: disable=import-outside-toplevel,import-error
        if torch.cuda.is_available():
            return preferred_device
    except Exception:
        pass

    print(f"[INFO] CUDA unavailable, fallback device: cpu (preferred was {preferred_device})")
    return "cpu"


def env_name(dataset_name: str, n_clients: int) -> str:
    return f"{dataset_name.lower()}-seed{SEED}-u{n_clients}-alpha{ALPHA}"


def env_train_path_for_training(dataset_name: str, n_clients: int) -> Path:
    return PROJECT_ROOT / "env" / "dilichlet" / env_name(dataset_name, n_clients) / "train" / "train.pt"


def env_train_path_from_allocator(dataset_name: str, n_clients: int) -> Path:
    return (
        PROJECT_ROOT
        / "env"
        / "dilichlet"
        / dataset_name
        / env_name(dataset_name, n_clients)
        / "train"
        / "train.pt"
    )


def get_accuracy_csv_path(dataset_name: str, n_clients: int) -> Path:
    e_name = env_name(dataset_name, n_clients)
    folder = PROJECT_ROOT / "FedQuickDrop" / "save" / f"tseed{SEED}-{e_name}-global-avg"
    file_name = f"tseed{SEED}-avg-{dataset_name}-cr{COMMUNICATION_ROUND}-ep{LOCAL_EPOCH}-{MODEL}.csv"
    return folder / file_name


def read_final_accuracy(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)

    if len(rows) <= 1:
        return None

    last = rows[-1]
    if not last:
        return None

    try:
        return float(last[0])
    except (ValueError, IndexError):
        return None


def plot_results(result_points: dict[str, list[tuple[int, float]]]) -> None:
    RESULT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for dataset_label, points in result_points.items():
        if not points:
            continue
        points = sorted(points, key=lambda x: x[0])
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        plt.plot(xs, ys, marker="o", label=dataset_label)

    plt.xlabel("Number of Clients")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Clients")
    plt.grid(True, alpha=0.3)
    has_points = any(points for points in result_points.values())
    if has_points:
        plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_FIGURE)
    print(f"[INFO] Saved figure to: {RESULT_FIGURE}")


def run_command(command: list[str], cwd: Path, env_vars: dict) -> int:
    print(" ".join(command))
    return subprocess.run(command, cwd=cwd, env=env_vars).returncode


def ensure_env(dataset_name: str, n_clients: int, python_exe: str, env_vars: dict) -> bool:
    output_train = env_train_path_for_training(dataset_name, n_clients)
    if output_train.exists():
        return True

    allocator_output = env_train_path_from_allocator(dataset_name, n_clients)
    if allocator_output.exists():
        target_root = output_train.parents[1]
        source_root = allocator_output.parents[1]
        target_root.parent.mkdir(parents=True, exist_ok=True)
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.copytree(source_root, target_root)
        return output_train.exists()

    command = [
        python_exe,
        str(ALLOCATOR_SCRIPT),
        "--dataset_name",
        dataset_name,
        "--num_clients",
        str(n_clients),
        "--alpha",
        str(ALPHA),
        "--seed",
        str(SEED),
    ]
    print(f"[GEN ] dataset={dataset_name}, n_clients={n_clients} -> generating FL env")
    code = run_command(command, PROJECT_ROOT / "env_generator" / "dilichlet_allocator", env_vars)
    if code != 0:
        return False

    allocator_output = env_train_path_from_allocator(dataset_name, n_clients)
    if not allocator_output.exists():
        return False

    target_root = output_train.parents[1]
    source_root = allocator_output.parents[1]
    target_root.parent.mkdir(parents=True, exist_ok=True)
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root)
    return output_train.exists()


def run_fedquickdrop(dataset_name: str, n_clients: int, device: str, python_exe: str, env_vars: dict) -> int:
    command = [
        python_exe,
        str(FEDQUICKDROP_SCRIPT),
        "--device",
        device,
        "--dataset",
        dataset_name,
        "--model",
        MODEL,
        "--env_path",
        "../../env",
        "--strategy",
        "dilichlet",
        "--env",
        env_name(dataset_name, n_clients),
        "--communication_round",
        str(COMMUNICATION_ROUND),
        "--local_epoch",
        str(LOCAL_EPOCH),
        "--seed",
        str(SEED),
    ]
    return run_command(command, PROJECT_ROOT / "FedQuickDrop" / "server", env_vars)


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    python_exe = sys.executable

    if not ALLOCATOR_SCRIPT.exists() or not FEDQUICKDROP_SCRIPT.exists():
        print("[ERROR] Required scripts not found. Please run under quickdrop-main.")
        sys.exit(1)

    device = resolve_device(PREFERRED_DEVICE)
    print(f"Using FedQuickDrop script: {FEDQUICKDROP_SCRIPT}")
    print(f"Client settings: {CLIENTS_LIST}")
    print(f"Datasets: {[d['dataset_name'] for d in EXPERIMENT_DATASETS]}")
    print(f"Device: {device}")
    print("-" * 80)

    env_vars = os.environ.copy()
    env_vars.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    current_pythonpath = env_vars.get("PYTHONPATH", "")
    env_vars["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + current_pythonpath if current_pythonpath else "")

    failed_runs = []
    result_points: dict[str, list[tuple[int, float]]] = {d["label"]: [] for d in EXPERIMENT_DATASETS}

    for dataset_cfg in EXPERIMENT_DATASETS:
        label = dataset_cfg["label"]
        dataset_name = dataset_cfg["dataset_name"]
        for n_clients in CLIENTS_LIST:
            print(f"[STEP] dataset={label}, n_clients={n_clients}")
            if not ensure_env(dataset_name, n_clients, python_exe, env_vars):
                failed_runs.append((label, n_clients, "env_generation_failed"))
                print(f"[FAIL] dataset={label}, n_clients={n_clients}, reason=env_generation_failed")
                if STRICT_MODE:
                    sys.exit(1)
                print("-" * 80)
                continue

            code = run_fedquickdrop(dataset_name, n_clients, device, python_exe, env_vars)
            if code != 0:
                failed_runs.append((label, n_clients, f"train_failed:{code}"))
                print(f"[FAIL] dataset={label}, n_clients={n_clients}, reason=train_failed:{code}")
                if STRICT_MODE:
                    sys.exit(1)
                print("-" * 80)
                continue

            csv_path = get_accuracy_csv_path(dataset_name, n_clients)
            acc = read_final_accuracy(csv_path)
            if acc is None:
                failed_runs.append((label, n_clients, "missing_accuracy_csv"))
                print(f"[FAIL] dataset={label}, n_clients={n_clients}, reason=missing_accuracy_csv")
                if STRICT_MODE:
                    sys.exit(1)
            else:
                result_points[label].append((n_clients, acc))
                print(f"[ OK ] dataset={label}, n_clients={n_clients}, final_acc={acc:.4f}")
            print("-" * 80)

    plot_results(result_points)

    if STRICT_MODE:
        for dataset_label, points in result_points.items():
            if not points:
                print(f"[ERROR] STRICT_MODE enabled: dataset={dataset_label} has no successful result points.")
                sys.exit(1)

    if failed_runs:
        print("Some runs failed:")
        for dataset_label, n_clients, reason in failed_runs:
            print(f"  - dataset={dataset_label}, n_clients={n_clients}, reason={reason}")

    if failed_runs:
        sys.exit(1)

    print("All runs finished successfully.")


if __name__ == "__main__":
    main()
