import os

# 批量运行 fashion-mnist 和 imagenet 实验，支持自定义客户端数量
clients_list = [10, 20, 30, 40, 50]
datasets = ["fashion-mnist", "imagenet"]

for data_name in datasets:
    for n_clients in clients_list:
        print(f"Running {data_name} with {n_clients} clients...")
        cmd = f"python main.py --data_name {data_name} --n_clients {n_clients}"
        os.system(cmd)
