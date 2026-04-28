# FedAvg / DDFed / TMCFE 实验说明

本目录提供了一个统一实验入口，用于运行：

- `FedAvg` 基线（明文聚合）
- `FedAvg + DDFed`（安全聚合，Rodot+）
- `FedAvg + TMCFE`（安全聚合）

---

## 0）首次 Pull 后的环境与数据准备

首次拉取项目后，建议先完成下面 4 步，再运行后续实验命令。

### 0.1 创建并激活 Python 环境（Windows PowerShell）

### 0.2 安装依赖

推荐一键安装：

```bash
pip install -r requirements.txt
```

### 0.3 验证 GPU 可用

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

若输出 `True` 且显示显卡名称，说明 GPU 环境可用。若不可用，检查是否有GPU或安装驱动。

### 0.4 数据与联邦环境说明

- 对于 `MNIST/FashionMNIST/SVHN/CIFAR10/CIFAR100`：首次运行会自动下载数据集到 `--data_path`（默认 `./data`）。
- 对于 `TinyImageNet`：首次运行会自动下载 `tiny-imagenet-200.zip` 并自动整理验证集目录（首次较慢，需联网和磁盘空间）。

具体环境安装见本 README 的“联邦数据划分与环境生成”

---

## 1）联邦数据划分与环境生成

在  ```ddfed_fl``` 目录下执行下列命令

### 1.1 生成联邦划分环境（划分数据集）

生成 **IID** 环境：

```bash
PYTHONPATH=. python ./env_generator/dilichlet_allocator/dilichlet_allocator.py --dataset_name MNIST --num_clients 20 --alpha 100.0 --seed 42
```

生成 **Non-IID** 环境（$\alpha=0.1$）：

```bash
PYTHONPATH=. python ./env_generator/dilichlet_allocator/dilichlet_allocator.py --dataset_name MNIST --num_clients 20 --alpha 0.1 --seed 42
```
注：生成的环境将保存在 DDFed-main/env 目录下。

准备完成后，直接从本 README 的“实验执行命令”开始运行即可。

---

## 2）实验执行命令

请在 `DDFed-main/ddfed_fl` 目录下执行。

统一对照参数（除方法与入口外保持一致）：

- `--device cuda:0`
- `--env_path "./env"`
- `--env affine-mnist-seed42-u20-alpha0.1-scale0.01`
- `--num_rounds 5`
- `--local_epochs 1`
- `--batch_size 64`
- `--participation_rate 0.2`
- `--threshold 5`
- `--num_decryptors 10`
- `--ddfed_project_root "../../DDFed-main"`
- `--quantization_scale 100000`
- `--secure_pack_size 8`
- `--packing_value_bits 32`
- `--seed 0`

### 2.1 运行 FedAvg baseline

IID:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m FedAvg.server.train_fedavg_tmcfe \\
  --method fedavg \
  --device cuda:0 \
  --env_path "../env" \
  --strategy "dilichlet/MNIST" \
  --env "mnist-seed42-u20-alpha100.0" \
  --num_rounds 20 \
  --local_epochs 1 \
  --batch_size 64 \
  --participation_rate 0.2 \
  --seed 0
```

Non-IID:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m FedAvg.server.train_fedavg_tmcfe \\
  --method fedavg \
  --device cuda:0 \
  --env_path "../env" \
  --strategy "dilichlet/MNIST" \
  --env "mnist-seed42-u20-alpha0.1" \
  --num_rounds 20 \
  --local_epochs 1 \
  --batch_size 64 \
  --participation_rate 0.2 \
  --seed 0
```


