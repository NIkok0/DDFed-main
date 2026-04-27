# FedAvg / DDFed / TMCFE 实验说明

本目录提供了一个统一实验入口，用于运行：

- `FedAvg` 基线（明文聚合）
- `FedAvg + DDFed`（安全聚合，Rodot+）
- `FedAvg + TMCFE`（安全聚合）

---

## 0）首次 Pull 后的环境与数据准备

首次拉取项目后，建议先完成下面 4 步，再运行后续实验命令。

### 0.1 创建并激活 Python 环境（Windows PowerShell）

```powershell
conda create -n myenv python=3.10 -y
conda activate myenv
```

### 0.2 安装依赖

推荐一键安装：

```powershell
pip install -r requirements.txt
```

如果你需要固定 GPU 版 PyTorch（而不是默认源解析），可改为：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 0.3 验证 GPU 可用

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

若输出 `True` 且显示显卡名称，说明 GPU 环境可用。

### 0.4 数据与联邦环境说明

- 联邦划分环境（`--env_path ./env`）已在仓库中提供，可直接使用。
- 对于 `MNIST/FashionMNIST/SVHN/CIFAR10/CIFAR100`：首次运行会自动下载数据集到 `--data_path`（默认 `./data`）。
- 对于 `TinyImageNet`：首次运行会自动下载 `tiny-imagenet-200.zip` 并自动整理验证集目录（首次较慢，需联网和磁盘空间）。

准备完成后，直接从本 README 的“实验执行命令”开始运行即可。

---

## 1）主入口

- 脚本（TMCFE）：`ddfed_fl/FedAvg/server/train_fedavg_tmcfe.py`
- 脚本（DDFed/Rodot+）：`ddfed_fl/FedAvg/server/train_fedavg_ddfed.py`

两个脚本分别将 TMCFE / DDFed 安全聚合接入到 FedAvg 训练流程，并支持与 baseline 对照。

---

## 2）关键函数（train_fedavg_tmcfe.py）

已实现：

- `compute_model_delta(global_model, local_model)`
- `apply_model_delta(global_model, delta_global)`
- `flatten_model_update(delta)`
- `unflatten_model_update(vector, model_template)`
- `quantize_update(vector, quantization_scale)`
- `dequantize_update(vector, quantization_scale)`
- `plaintext_fedavg_aggregate(client_deltas, client_weights)`
- `secure_aggregate_tmcfe(client_deltas, client_weights, active_clients, active_decryptors, label, tmcfe_ctx)`
- `train_fedavg_tmcfe(config)`
- `sanity_check_secure_aggregation(config)`

---

## 3）目录与数据说明（已内置）

为保证只依赖 `DDFed-main` 即可运行，本目录已完整包含 FedAvg 所需代码：

- `ddfed_fl/FedAvg/`（FedAvg 完整代码）
- `ddfed_fl/utils/`（训练与模型工具）
- `ddfed_fl/env_generator/`（环境生成相关代码）
- `ddfed_fl/env/quickdrop-affine/`（默认联邦划分环境）

默认数据路径：

- `--data_path ./data`
- 数据下载规则同“0.4 数据与联邦环境说明”。

默认环境路径：

- `--env_path ./env`

---

## 4）主要命令行参数

核心参数：

- `--method {fedavg,fedavg_ddfed,fedavg_tmcfe}`
- `--num_clients`
- `--participation_rate`
- `--num_rounds`
- `--local_epochs`
- `--threshold`
- `--num_decryptors`
- `--seed`
- `--device`（建议 `cuda:0`）

安全聚合公共参数（DDFed / TMCFE）：

- `--lambda_sec`
- `--quantization_scale`
- `--secure_pack_size`
- `--packing_value_bits`（默认 32）
- `--skip_zero_blocks`（默认 true）
- `--ddfed_project_root`

TMCFE 额外参数：

- `--setup_once`
- `--max_dlog`

DDFed 额外参数：

- `--max_plaintext_bits`

掉线参数：

- `--simulate_client_dropout`
- `--client_dropout_rate`
- `--simulate_decryptor_dropout`
- `--decryptor_dropout_rate`

重放攻击参数：

- `--simulate_replay_attack`
- `--replay_attack_type {client_ciphertext,partial_decryption,cross_set}`
- `--replay_ratio`
- `--replay_source_round`
- `--replay_target_round`

---

## 5）实验执行命令

请在 `E:\code\my-test\DDFed-main` 目录下执行。

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

### 5.1 运行 FedAvg baseline

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_tmcfe `
  --method fedavg `
  --device cuda:0 `
  --env_path "./env" `
  --env affine-mnist-seed42-u20-alpha0.1-scale0.01 `
  --num_rounds 5 `
  --local_epochs 1 `
  --batch_size 64 `
  --participation_rate 0.2 `
  --seed 0
```

### 5.2 运行 FedAvg + TMCFE

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_tmcfe `
  --method fedavg_tmcfe `
  --device cuda:0 `
  --env_path "./env" `
  --env affine-mnist-seed42-u20-alpha0.1-scale0.01 `
  --num_rounds 5 `
  --local_epochs 1 `
  --batch_size 64 `
  --participation_rate 0.2 `
  --threshold 5 `
  --num_decryptors 10 `
  --ddfed_project_root "../../DDFed-main" `
  --quantization_scale 100000 `
  --secure_pack_size 8 `
  --packing_value_bits 32 `
  --skip_zero_blocks true `
  --setup_once true `
  --seed 0
```

### 5.2b 运行 FedAvg + DDFed（Rodot+）

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_ddfed `
  --method fedavg_ddfed `
  --device cuda:0 `
  --env_path "./env" `
  --env affine-mnist-seed42-u20-alpha0.1-scale0.01 `
  --num_rounds 5 `
  --local_epochs 1 `
  --batch_size 64 `
  --participation_rate 0.2 `
  --threshold 5 `
  --num_decryptors 10 `
  --ddfed_project_root "../../DDFed-main" `
  --quantization_scale 100000 `
  --secure_pack_size 8 `
  --packing_value_bits 32 `
  --skip_zero_blocks true `
  --seed 0
```

### 5.3 客户端掉线实验

FedAvg + TMCFE：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_tmcfe `
  --method fedavg_tmcfe `
  --device cuda:0 `
  --env_path "./env" `
  --env affine-mnist-seed42-u20-alpha0.1-scale0.01 `
  --num_rounds 5 `
  --local_epochs 1 `
  --batch_size 64 `
  --participation_rate 0.2 `
  --threshold 5 `
  --num_decryptors 10 `
  --ddfed_project_root "../../DDFed-main" `
  --quantization_scale 100000 `
  --secure_pack_size 8 `
  --packing_value_bits 32 `
  --skip_zero_blocks true `
  --setup_once true `
  --seed 0 `
  --simulate_client_dropout true `
  --client_dropout_rate 0.2
```

FedAvg + DDFed（Rodot+）：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_ddfed `
  --method fedavg_ddfed `
  --device cuda:0 `
  --env_path "./env" `
  --env affine-mnist-seed42-u20-alpha0.1-scale0.01 `
  --num_rounds 5 `
  --local_epochs 1 `
  --batch_size 64 `
  --participation_rate 0.2 `
  --threshold 5 `
  --num_decryptors 10 `
  --ddfed_project_root "../../DDFed-main" `
  --quantization_scale 100000 `
  --secure_pack_size 8 `
  --packing_value_bits 32 `
  --skip_zero_blocks true `
  --simulate_client_dropout true `
  --client_dropout_rate 0.2 `
  --seed 0
```

### 5.4 解密节点掉线实验

FedAvg + TMCFE：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_tmcfe `
  --method fedavg_tmcfe `
  --device cuda:0 `
  --env_path "./env" `
  --env affine-mnist-seed42-u20-alpha0.1-scale0.01 `
  --num_rounds 5 `
  --local_epochs 1 `
  --batch_size 64 `
  --participation_rate 0.2 `
  --threshold 5 `
  --num_decryptors 10 `
  --ddfed_project_root "../../DDFed-main" `
  --quantization_scale 100000 `
  --secure_pack_size 8 `
  --packing_value_bits 32 `
  --skip_zero_blocks true `
  --setup_once true `
  --seed 0 `
  --simulate_decryptor_dropout true `
  --decryptor_dropout_rate 0.3
```

FedAvg + DDFed（Rodot+）：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_ddfed `
  --method fedavg_ddfed `
  --device cuda:0 `
  --env_path "./env" `
  --env affine-mnist-seed42-u20-alpha0.1-scale0.01 `
  --num_rounds 5 `
  --local_epochs 1 `
  --batch_size 64 `
  --participation_rate 0.2 `
  --threshold 5 `
  --num_decryptors 10 `
  --ddfed_project_root "../../DDFed-main" `
  --quantization_scale 100000 `
  --secure_pack_size 8 `
  --packing_value_bits 32 `
  --skip_zero_blocks true `
  --simulate_decryptor_dropout true `
  --decryptor_dropout_rate 0.3 `
  --seed 0
```

### 5.5 重放攻击实验

FedAvg + TMCFE：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_tmcfe `
  --method fedavg_tmcfe `
  --device cuda:0 `
  --env_path "./env" `
  --env affine-mnist-seed42-u20-alpha0.1-scale0.01 `
  --num_rounds 5 `
  --local_epochs 1 `
  --batch_size 64 `
  --participation_rate 0.2 `
  --threshold 5 `
  --num_decryptors 10 `
  --ddfed_project_root "../../DDFed-main" `
  --quantization_scale 100000 `
  --secure_pack_size 8 `
  --packing_value_bits 32 `
  --skip_zero_blocks true `
  --setup_once true `
  --seed 0 `
  --simulate_replay_attack true `
  --replay_attack_type client_ciphertext `
  --replay_ratio 0.2 `
  --replay_source_round 1 `
  --replay_target_round 2
```

FedAvg + DDFed（Rodot+）：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_ddfed `
  --method fedavg_ddfed `
  --device cuda:0 `
  --env_path "./env" `
  --env affine-mnist-seed42-u20-alpha0.1-scale0.01 `
  --num_rounds 5 `
  --local_epochs 1 `
  --batch_size 64 `
  --participation_rate 0.2 `
  --threshold 5 `
  --num_decryptors 10 `
  --ddfed_project_root "../../DDFed-main" `
  --quantization_scale 100000 `
  --secure_pack_size 8 `
  --packing_value_bits 32 `
  --skip_zero_blocks true `
  --simulate_replay_attack true `
  --replay_attack_type client_ciphertext `
  --replay_ratio 0.2 `
  --replay_source_round 1 `
  --replay_target_round 2 `
  --seed 0
```

---

## 6）Sanity Check（正确性检查）

用于比较明文聚合与安全聚合的数值误差：

FedAvg + TMCFE：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_tmcfe `
  --method fedavg_tmcfe `
  --device cpu `
  --run_sanity_check `
  --sanity_check_only `
  --quantization_scale 100000 `
  --secure_pack_size 8 `
  --packing_value_bits 32 `
  --skip_zero_blocks true `
  --num_decryptors 3 `
  --threshold 2 `
  --setup_once true `
  --seed 0
```

FedAvg + DDFed（Rodot+）：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_ddfed `
  --method sanity_packing `
  --device cpu `
  --seed 0 `
  --num_decryptors 3 `
  --threshold 2 `
  --quantization_scale 100000 `
  --secure_pack_size 8 `
  --packing_value_bits 32 `
  --ddfed_project_root "../../DDFed-main"
```

输出示例：

- `sanity_check max_abs_error=...`

---

## 7）结果文件与输出流程（横轴=客户端数量）

实验结果默认保存到：

- `results/fedavg_baseline.csv`
- `results/fedavg_ddfed.csv`
- `results/fedavg_tmcfe.csv`
- `results/fedavg_tmcfe_dropout.csv`（当启用 dropout 参数）
- `results/fedavg_tmcfe_replay.csv`（当启用 replay 参数）

每轮记录字段（核心）包含：

- `train_loss`
- `test_accuracy`
- `test_loss`
- `active_clients / dropped_clients`
- `active_decryptors / dropped_decryptors`
- `aggregation_success / failure_reason`
- `enc_time / dk_generate_time / par_dec_time / com_dec_time / total_crypto_time`
- `max_abs_error`
- `secure_pack_size / slot_bits / packing_value_bits / padding_bits`
- `num_model_params / num_ciphertexts / compression_ratio / packing_overflow`

### 7.1 客户端数量 sweep（横轴自变量）

说明：默认输出流程采用**客户端数量**作为横轴（`number of clients`），通过切换不同联邦环境（例如 `u10`、`u20`）并记录每组结果实现对比。`secure_pack_size` 仅作为一个可调参数使用。

示例（DDFed，pack=8）：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m FedAvg.server.train_fedavg_ddfed `
  --method fedavg_ddfed `
  --device cuda:0 `
  --env_path "./env" `
  --env affine-cifar10-seed42-u10-alpha0.1-0.01 `
  --num_rounds 1 `
  --local_epochs 1 `
  --batch_size 64 `
  --participation_rate 0.2 `
  --threshold 2 `
  --num_decryptors 3 `
  --ddfed_project_root "../../DDFed-main" `
  --quantization_scale 100000 `
  --secure_pack_size 8 `
  --packing_value_bits 32 `
  --skip_zero_blocks true `
  --seed 0
```

把 `--env` 改为 `u20`（或更多用户环境）重复运行，即可得到以客户端数量为横轴的对照数据。

### 7.2 图像输出命名

为保证横轴是客户端数量，建议最终图采用以下命名：

- `figures/crypto_time_vs_num_clients.png`
- `figures/round_time_vs_num_clients.png`
- `figures/accuracy_vs_round.png`
- `figures/max_abs_error_vs_num_clients.png`
- `figures/num_ciphertexts_vs_num_clients.png`

---

## 8）注意事项

- 默认推荐 `secure_pack_size=1`，数值更稳定。
- DDFed 已支持明文打包（`secure_pack_size>1`）；TMCFE 受 `max_dlog` 限制可能自动收缩有效打包长度。
- 若 CUDA 不可用，脚本会自动回退到 CPU。
