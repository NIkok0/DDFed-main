# run_experiments.py
# 调度脚本：并行运行多个联邦学习实验
# 控制变量：参与客户端数量（10,20,30,40,50）
# 数据集：fashion-mnist, imagenet

import multiprocessing as mp
import subprocess
import os
import time
from datetime import datetime

# 实验配置
DATASETS = ["fashion-mnist", "imagenet"]
CLIENT_COUNTS = [10, 20, 30, 40, 50]

# 其他实验参数（可根据需要调整）
OTHER_PARAMS = {
    "global_epoch": 20,
    "local_epoch": 10,
    "local_lr": 0.005,
    "local_batch_size": 64,
    "seed": 1,
}

def run_experiment(dataset, n_clients, log_dir="logs"):
    """运行单个实验"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建实验标识符
    exp_id = f"{dataset}_clients{n_clients}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{exp_id}_{timestamp}.log")
    
    # 构建命令 - 传递客户端数量参数
    cmd = f'python "Federated unlearning/FedEraser-Code/Fed_Unlearn_main.py" --data_name {dataset} --n_clients {n_clients}'
    
    print(f"[{timestamp}] Starting: {exp_id}")
    print(f"  Command: {cmd}")
    print(f"  Log file: {log_file}")
    
    # 执行实验
    start_time = time.time()
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"="*60 + "\n")
        f.write(f"Experiment: {exp_id}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Client Count: {n_clients}\n")
        f.write(f"Start Time: {timestamp}\n")
        f.write(f"Parameters: {OTHER_PARAMS}\n")
        f.write(f"="*60 + "\n\n")
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 实时写入日志
        for line in process.stdout:
            f.write(line)
            print(f"  [{exp_id}] {line.rstrip()}")
        
        process.wait()
    
    elapsed = time.time() - start_time
    status = "SUCCESS" if process.returncode == 0 else "FAILED"
    
    print(f"[{exp_id}] Completed: {status} ({elapsed:.1f}s)")
    return {
        "dataset": dataset,
        "n_clients": n_clients,
        "status": status,
        "elapsed": elapsed,
        "log_file": log_file
    }

def main():
    """主函数：并行调度所有实验"""
    print("="*60)
    print("Federated Unlearning Experiment Scheduler")
    print("="*60)
    print(f"Datasets: {DATASETS}")
    print(f"Client Counts: {CLIENT_COUNTS}")
    print(f"Total Experiments: {len(DATASETS) * len(CLIENT_COUNTS)}")
    print("="*60)
    
    # 生成实验任务列表
    experiments = []
    for dataset in DATASETS:
        for n_clients in CLIENT_COUNTS:
            experiments.append((dataset, n_clients))
    
    # 并行执行实验（限制并发数）
    max_workers = min(mp.cpu_count(), len(experiments))
    print(f"\nStarting {len(experiments)} experiments with {max_workers} workers...\n")
    
    results = []
    with mp.Pool(processes=max_workers) as pool:
        # 使用 starmap 并行执行
        results = pool.starmap(run_experiment, experiments)
    
    # 汇总结果
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    
    success_count = 0
    for result in results:
        status_icon = "✓" if result["status"] == "SUCCESS" else "✗"
        print(f"{status_icon} {result['dataset']:15} | Clients: {result['n_clients']:2} | "
              f"Time: {result['elapsed']:7.1f}s | {result['status']}")
        if result["status"] == "SUCCESS":
            success_count += 1
    
    print("="*60)
    print(f"Total: {len(results)} | Success: {success_count} | Failed: {len(results) - success_count}")
    print("="*60)
    
    # 保存结果到文件
    result_file = f"logs/experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("logs", exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Experiment Results\n")
        f.write("="*60 + "\n")
        for result in results:
            f.write(f"{result['dataset']},{result['n_clients']},{result['status']},{result['elapsed']:.1f}\n")
    
    print(f"\nResults saved to: {result_file}")

if __name__ == "__main__":
    main()