# DDFed-FL (Federated Learning with Secure Aggregation)

联邦学习安全聚合实验框架。在一个统一的 FedAvg 流程中，接入 3 种安全聚合方案进行对比：

| 方案 | 方法 | 聚合类型 | 密码原语 |
|------|------|:---:|------|
| **FedAvg Baseline** | 明文聚合 | 明文 | — |
| **DDFed** | 安全聚合 | 密文 | Rodot+ |
| **TMCFE** | 安全聚合 | 密文 | Threshold MCFE |
| **Lepcat** | 安全聚合 | 密文 | DMCFE-IP |

---

## 目录结构

```
ddfed_fl/
├── FedAvg/                            # FedAvg + 安全聚合服务器与客户端
│   ├── __init__.py
│   ├── utils.py                       # 随机种子固定
│   ├── client/
│   │   ├── __init__.py
│   │   └── client_fedavg.py           # FedAvg 客户端逻辑
│   └── server/
│       ├── __init__.py
│       ├── secure_packing.py          # 明文打包算法 (pack/unpack)
│       ├── server_fedavg.py           # FedAvg Server 基类
│       ├── train_fedavg_baseline.py   # [Baseline] 明文 FedAvg 训练入口
│       ├── train_fedavg_ddfed.py      # [DDFed] Rodot+ 安全聚合入口
│       ├── train_fedavg_tmcfe.py      # [TMCFE] 安全聚合入口
│       └── train_fedavg_lepcat.py     # [Lepcat] DMCFE-IP 安全聚合入口
├── utils/                             # 通用工具
│   ├── __init__.py
│   ├── fed_utils.py                   # 数据集加载、训练/评估工具函数
│   ├── networks.py                    # 模型架构定义 (MLP/CNN/ResNet)
│   ├── optimizers.py                  # 优化器工厂
│   └── device_test.py                 # GPU 可用性检测
├── env_generator/                     # FL 联邦环境生成器
│   ├── __init__.py
│   ├── utils.py                       # 环境工具函数
│   ├── readMe.md                      # 分配器使用说明
│   ├── generate_fl_env.sh             # 一键环境生成脚本
│   ├── dilichlet_allocator/
│   │   ├── __init__.py
│   │   └── dilichlet_allocator.py     # Dirichlet 分布分配器主入口
│   └── preprocessing/
│       ├── __init__.py
│       └── baselines_dataloader.py    # Client 端多数据集加载器
├── figures/                           # 实验结果图
├── results/                           # 实验结果 CSV
├── requirements.txt                   # Python 依赖
└── README.md                          # 本文件
```

---

## 方案对比总览

| 特性 | FedAvg Baseline | DDFed (Rodot+) | TMCFE | Lepcat (DMCFE-IP) |
|------|:---:|:---:|:---:|:---:|
| 加密聚合 | ✗ | ✓ | ✓ | ✓ |
| 阈值解密 | — | ✓ | ✓ | ✓ |
| 明文打包支持 | — | ✓ | ✓ | ✓ |
| 客户端掉线容忍 | — | ✓ | ✓ | ✓ |
| 解密节点掉线容忍 | — | ✓ | ✓ | ✓ |
| 重放攻击模拟 | — | ✓ | ✓ | ✓ |
| Sanity Check | ✗ | ✓ | ✓ | ✓ |

---

## 环境要求

- **Python**: 3.8+
- **PyTorch**: 1.6+（推荐 2.0+）
- **CUDA**: 可选（推荐 12.1+）

### 依赖安装

```bash
pip install -r requirements.txt
```

### 验证 GPU 可用

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

若输出 `True` 且显示显卡名称，说明 GPU 环境可用。

### 数据与环境准备

- 训练数据默认路径：`--data_path ../../data`
- 联邦环境默认路径：`--env_path ./env`
- 支持数据集：MNIST / FashionMNIST / CIFAR10 / CIFAR100 / SVHN / TinyImageNet
- 首次运行会自动下载数据集到 `--data_path` 指定目录
- 联邦环境由 `env_generator/dilichlet_allocator/dilichlet_allocator.py` 生成

生成联邦环境示例：

```bash
cd /home/hyr/miaoli/DDFed-main/ddfed_fl
python env_generator/dilichlet_allocator/dilichlet_allocator.py \
    --dataset_name FashionMNIST \
    --num_clients 30 \
    --alpha 0.1 \
    --seed 42
```

---

## 统一入口

所有实验通过以下 4 个训练入口运行：

| 入口脚本 | 方法 | 描述 |
|----------|------|------|
| `FedAvg/server/train_fedavg_baseline.py` | `fedavg` | 明文 FedAvg 基线 |
| `FedAvg/server/train_fedavg_ddfed.py` | `fedavg_ddfed` | DDFed / Rodot+ 安全聚合 |
| `FedAvg/server/train_fedavg_tmcfe.py` | `fedavg_tmcfe` | TMCFE 安全聚合 |
| `FedAvg/server/train_fedavg_lepcat.py` | `fedavg_lepcat` | Lepcat / DMCFE-IP 安全聚合 |

---

## 核心 CLI 参数

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--method` | str | — | 运行方法：`fedavg` / `fedavg_ddfed` / `fedavg_tmcfe` / `fedavg_lepcat` |
| `--env` | str | — | 联邦环境名（如 `affine-fashionmnist-seed42-u20-alpha0.1-0.01`） |
| `--env_path` | str | `./env` | 联邦环境目录 |
| `--data_path` | str | `../../data` | 数据集根目录 |
| `--dataset` | str | `MNIST` | 数据集名称 |
| `--device` | str | `cuda:0` | 计算设备 |
| `--num_rounds` | int | — | 全局通信轮数 |
| `--local_epochs` | int | — | 客户端本地训练轮数 |
| `--batch_size` | int | — | 批次大小 |
| `--participation_rate` | float | — | 每轮客户端参与比例 |
| `--seed` | int | `0` | 随机种子 |

### 安全聚合公共参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--threshold` | int | — | 阈值解密门限 |
| `--num_decryptors` | int | — | 解密节点总数 |
| `--quantization_scale` | int | `100000` | 量化缩放因子 |
| `--secure_pack_size` | int | `8` | 安全打包块大小 |
| `--packing_value_bits` | int | `32` | 每元素编码位数 |
| `--skip_zero_blocks` | bool | `true` | 跳过全零块加速 |
| `--lambda_sec` | int | — | 安全参数（λ） |

### DDFed / TMCFE 专用参数

| 参数 | 入口 | 说明 |
|------|------|------|
| `--ddfed_project_root` | DDFed / TMCFE | DDFed 项目根目录（如 `../../DDFed-main`） |
| `--setup_once` | TMCFE | 是否复用一次 Setup |
| `--max_dlog` | TMCFE | 离散对数搜索上限 |

### Lepcat 专用参数

| 参数 | 说明 |
|------|------|
| `--lepcat_project_root` | Lepcat 项目根目录（如 `../../DDFed-main`） |
| `--active_client_count` | 活跃客户端数（优先于 `participation_rate`） |
| `--participant_counts` | 参与者数量列表（如 `5,10,20`），用于 scalability 实验 |
| `--pack_sizes` | 打包大小列表（如 `1,4,8,16`） |
| `--experiment` | 实验类型（如 `participant_scalability`） |
| `--save_results` | 是否保存结果 CSV |
| `--save_figures` | 是否保存图片 |
| `--results_dir` | 结果保存目录 |
| `--figures_dir` | 图片保存目录 |

### 掉线模拟参数

| 参数 | 说明 |
|------|------|
| `--simulate_client_dropout` | 启用客户端掉线模拟 |
| `--client_dropout_rate` | 客户端掉线率（如 `0.2`） |
| `--simulate_decryptor_dropout` | 启用解密节点掉线模拟 |
| `--decryptor_dropout_rate` | 解密节点掉线率（如 `0.3`） |

### 重放攻击参数

| 参数 | 说明 |
|------|------|
| `--simulate_replay_attack` | 启用重放攻击模拟 |
| `--replay_attack_type` | 攻击类型：`client_ciphertext` / `partial_decryption` / `cross_set` |
| `--replay_ratio` | 重放比例 |
| `--replay_source_round` | 重放来源轮次 |
| `--replay_target_round` | 重放目标轮次 |

---

## 快速开始

```bash
# 进入 ddfed_fl 目录
cd /home/hyr/miaoli/DDFed-main/ddfed_fl

# 运行 FedAvg 明文基线
python -m FedAvg.server.train_fedavg_baseline \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 \
    --local_epochs 1 \
    --batch_size 64 \
    --participation_rate 0.2 \
    --seed 0

# 运行 FedAvg + DDFed (Rodot+)
python -m FedAvg.server.train_fedavg_ddfed \
    --method fedavg_ddfed \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 \
    --local_epochs 1 \
    --batch_size 64 \
    --participation_rate 0.2 \
    --threshold 5 \
    --num_decryptors 10 \
    --ddfed_project_root "../../DDFed-main" \
    --quantization_scale 100000 \
    --secure_pack_size 8 \
    --packing_value_bits 32 \
    --skip_zero_blocks true \
    --seed 0

# 运行 FedAvg + TMCFE
python -m FedAvg.server.train_fedavg_tmcfe \
    --method fedavg_tmcfe \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 \
    --local_epochs 1 \
    --batch_size 64 \
    --participation_rate 0.2 \
    --threshold 5 \
    --num_decryptors 10 \
    --ddfed_project_root "../../DDFed-main" \
    --quantization_scale 100000 \
    --secure_pack_size 8 \
    --packing_value_bits 32 \
    --skip_zero_blocks true \
    --setup_once true \
    --seed 0

# 运行 FedAvg + Lepcat (DMCFE-IP)
python -m FedAvg.server.train_fedavg_lepcat \
    --method fedavg_lepcat \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 \
    --local_epochs 1 \
    --batch_size 64 \
    --active_client_count 10 \
    --threshold 5 \
    --seed 0 \
    --lepcat_project_root "../../DDFed-main" \
    --quantization_scale 100000 \
    --secure_pack_size 1 \
    --packing_value_bits 32 \
    --save_results true \
    --save_figures true
```

---

## 实验矩阵

### 客户端掉线实验

```bash
# DDFed + 客户端掉线
python -m FedAvg.server.train_fedavg_ddfed \
    --method fedavg_ddfed \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 --local_epochs 1 --batch_size 64 \
    --participation_rate 0.2 --threshold 5 --num_decryptors 10 \
    --ddfed_project_root "../../DDFed-main" \
    --quantization_scale 100000 --secure_pack_size 8 --packing_value_bits 32 \
    --skip_zero_blocks true --seed 0 \
    --simulate_client_dropout true --client_dropout_rate 0.2

# TMCFE + 客户端掉线
python -m FedAvg.server.train_fedavg_tmcfe \
    --method fedavg_tmcfe \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 --local_epochs 1 --batch_size 64 \
    --participation_rate 0.2 --threshold 5 --num_decryptors 10 \
    --ddfed_project_root "../../DDFed-main" \
    --quantization_scale 100000 --secure_pack_size 8 --packing_value_bits 32 \
    --skip_zero_blocks true --setup_once true --seed 0 \
    --simulate_client_dropout true --client_dropout_rate 0.2
```

### 解密节点掉线实验

```bash
# DDFed + 解密节点掉线
python -m FedAvg.server.train_fedavg_ddfed \
    --method fedavg_ddfed \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 --local_epochs 1 --batch_size 64 \
    --participation_rate 0.2 --threshold 5 --num_decryptors 10 \
    --ddfed_project_root "../../DDFed-main" \
    --quantization_scale 100000 --secure_pack_size 8 --packing_value_bits 32 \
    --skip_zero_blocks true --seed 0 \
    --simulate_decryptor_dropout true --decryptor_dropout_rate 0.3

# TMCFE + 解密节点掉线
python -m FedAvg.server.train_fedavg_tmcfe \
    --method fedavg_tmcfe \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 --local_epochs 1 --batch_size 64 \
    --participation_rate 0.2 --threshold 5 --num_decryptors 10 \
    --ddfed_project_root "../../DDFed-main" \
    --quantization_scale 100000 --secure_pack_size 8 --packing_value_bits 32 \
    --skip_zero_blocks true --setup_once true --seed 0 \
    --simulate_decryptor_dropout true --decryptor_dropout_rate 0.3
```

### 重放攻击实验

```bash
# DDFed + 重放攻击 (client_ciphertext)
python -m FedAvg.server.train_fedavg_ddfed \
    --method fedavg_ddfed \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 --local_epochs 1 --batch_size 64 \
    --participation_rate 0.2 --threshold 5 --num_decryptors 10 \
    --ddfed_project_root "../../DDFed-main" \
    --quantization_scale 100000 --secure_pack_size 8 --packing_value_bits 32 \
    --skip_zero_blocks true --seed 0 \
    --simulate_replay_attack true --replay_attack_type client_ciphertext \
    --replay_ratio 0.2 --replay_source_round 1 --replay_target_round 2

# TMCFE + 重放攻击
python -m FedAvg.server.train_fedavg_tmcfe \
    --method fedavg_tmcfe \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 --local_epochs 1 --batch_size 64 \
    --participation_rate 0.2 --threshold 5 --num_decryptors 10 \
    --ddfed_project_root "../../DDFed-main" \
    --quantization_scale 100000 --secure_pack_size 8 --packing_value_bits 32 \
    --skip_zero_blocks true --setup_once true --seed 0 \
    --simulate_replay_attack true --replay_attack_type client_ciphertext \
    --replay_ratio 0.2 --replay_source_round 1 --replay_target_round 2
```

### 参与者数量扩展实验 (Lepcat)

```bash
# 单 pack size 扩展
python -m FedAvg.server.train_fedavg_lepcat \
    --method fedavg_lepcat --experiment participant_scalability \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 --local_epochs 1 --batch_size 64 \
    --participant_counts 5,10,20 --threshold 5 --seed 0 \
    --lepcat_project_root "../../DDFed-main" \
    --quantization_scale 100000 --secure_pack_size 8 --packing_value_bits 32 \
    --save_results true --save_figures true

# 多 pack size 扩展
python -m FedAvg.server.train_fedavg_lepcat \
    --method fedavg_lepcat --experiment participant_scalability \
    --device cuda:0 \
    --env_path "./env" \
    --env affine-fashionmnist-seed42-u20-alpha0.1-0.01 \
    --num_rounds 5 --local_epochs 1 --batch_size 64 \
    --participant_counts 5,10,20 --pack_sizes 1,4,8,16 \
    --threshold 5 --seed 0 \
    --lepcat_project_root "../../DDFed-main" \
    --quantization_scale 100000 --packing_value_bits 32 \
    --save_results true --save_figures true
```

---

## Sanity Check（正确性验证）

对比明文聚合与安全聚合的数值误差：

```bash
# DDFed Sanity Check
python -m FedAvg.server.train_fedavg_ddfed \
    --method sanity_packing --device cpu --seed 0 \
    --num_decryptors 3 --threshold 2 \
    --quantization_scale 100000 --secure_pack_size 8 --packing_value_bits 32 \
    --ddfed_project_root "../../DDFed-main"

# TMCFE Sanity Check
python -m FedAvg.server.train_fedavg_tmcfe \
    --method fedavg_tmcfe --device cpu \
    --run_sanity_check --sanity_check_only \
    --quantization_scale 100000 --secure_pack_size 8 --packing_value_bits 32 \
    --skip_zero_blocks true --num_decryptors 3 --threshold 2 --setup_once true \
    --seed 0

# Lepcat Sanity Check
python -m FedAvg.server.train_fedavg_lepcat \
    --method sanity_lepcat --threshold 2 --seed 0 \
    --lepcat_project_root "../../DDFed-main" \
    --quantization_scale 100000 --secure_pack_size 8 --packing_value_bits 32 \
    --active_client_count 5 --participant_counts 5,10,20 \
    --pack_sizes 1,8 --save_results true
```

---

## 结果输出

实验结果自动保存到 `results/` 目录：

| 方法 | 输出文件 |
|------|----------|
| FedAvg Baseline | `results/fedavg_baseline.csv` |
| DDFed (Rodot+) | `results/fedavg_ddfed.csv` |
| DDFed + Dropout | `results/fedavg_ddfed_dropout.csv` |
| DDFed + Replay | `results/fedavg_ddfed_replay.csv` |
| TMCFE | `results/fedavg_tmcfe.csv` |
| TMCFE + Dropout | `results/fedavg_tmcfe_dropout.csv` |
| TMCFE + Replay | `results/fedavg_tmcfe_replay.csv` |
| Lepcat | `results/fedavg_lepcat.csv` |
| Lepcat + Dropout | `results/fedavg_lepcat_dropout.csv` |
| Lepcat + Replay | `results/fedavg_lepcat_replay.csv` |

### CSV 统一字段

每轮记录包含以下字段，三种安全聚合方法输出格式统一，可直接拼接对比：

| 字段 | 说明 |
|------|------|
| `round` | 通信轮次 |
| `method` | 聚合方法 |
| `num_clients` | 客户端总数 |
| `participation_rate` | 参与率 |
| `train_loss` | 训练损失 |
| `test_accuracy` | 测试准确率 |
| `test_loss` | 测试损失 |
| `communication_time` | 通信时间 |
| `encryption_time` / `enc_time` | 加密时间 |
| `partial_decryption_time` / `par_dec_time` | 部分解密时间 |
| `combine_decryption_time` / `com_dec_time` | 组合解密时间 |
| `total_crypto_time` | 总加密时间 |
| `secure_pack_size` / `slot_bits` / `packing_value_bits` | 打包参数 |
| `num_model_params` / `num_ciphertexts` / `compression_ratio` | 密文数量统计 |
| `max_abs_error` | 最大绝对误差 |
| `pack_block_count` / `pack_avg_size` / `pack_max_base` / `pack_max_qmax` | 打包统计 |
| `round_total_time` | 单轮总时间 |

---

## 注意事项

- 推荐使用 `secure_pack_size=1` 以获得最稳定的数值精度
- DDFed 支持明文打包（`secure_pack_size > 1`）；TMCFE 受 `max_dlog` 限制可能自动收缩有效打包长度
- 若 CUDA 不可用，脚本自动回退到 CPU
- 对于 FashionMNIST / CIFAR10 / CIFAR100，首次运行自动下载数据集
- 联邦环境需首先生成，默认读取 `--env_path` 下指定的环境目录

---

## 引用

- **DDFed (Rodot+)**: 见 `../../ddfed_crypto/README.md`
- **TMCFE**: Threshold Multi-Client Functional Encryption 安全聚合方案
- **Lepcat (DMCFE-IP)**: Decentralized MCFE for Inner-Product 安全聚合方案