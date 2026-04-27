# FedAvg + DDFed 实验说明

本目录提供了一个统一实验入口，用于运行：

- `FedAvg` 基线（明文聚合）
- `FedAvg + DDFed`（安全聚合）

实现目标是：**不重写 FedAvg 与 DDFed 的核心逻辑**，只做必要封装与接口适配。

---

## 1）主入口

- 脚本：`ddfed_fl/fedavg_ddfed_experiment.py`

该脚本将 DDFed 安全聚合接入到 FedAvg 训练流程，并通过命令行参数切换基线/安全模式。

---

## 2）关键函数

在 `fedavg_ddfed_experiment.py` 中实现了：

- `compute_model_delta(global_model, local_model)`
- `apply_model_delta(global_model, delta_global)`
- `flatten_model_update(delta)`
- `unflatten_model_update(vector, model_template)`
- `secure_aggregate_ddfed(client_deltas, client_weights, ddfed_config)`
- `train_fedavg_ddfed(config)`
- `sanity_check_secure_aggregation(config)`

---

## 3）目录与数据说明（已内置）

为保证只依赖 `DDFed-main` 即可运行，本目录已完整包含 FedAvg 所需代码：

- `ddfed_fl/FedAvg/`（FedAvg 完整代码）
- `ddfed_fl/utils/`（训练与模型工具）
- `ddfed_fl/env_generator/`（环境生成相关代码）
- `ddfed_fl/env/quickdrop-affine/`（默认联邦划分环境）

默认数据路径：

- `--data_path ./data`（MNIST/FashionMNIST 等会自动下载到该目录）

默认环境路径：

- `--env_path ./env`

---

## 4）主要命令行参数

核心参数：

- `--method {fedavg,fedavg_ddfed}`
- `--num_clients`
- `--participation_rate`
- `--num_rounds`
- `--local_epochs`
- `--threshold`
- `--num_decryptors`
- `--seed`
- `--device`（建议 `cuda:0`）

DDFed 相关参数：

- `--lambda_sec`
- `--quantization_scale`
- `--secure_pack_size`
- `--max_plaintext_bits`

---

## 5）GPU 运行命令

请在 `E:\code\my-test\DDFed-main` 目录下执行。

### 5.1 运行 FedAvg 基线

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m ddfed_fl.fedavg_ddfed_experiment `
  --method fedavg `
  --device cuda:0 `
  --num_rounds 50 `
  --local_epochs 5 `
  --participation_rate 0.4 `
  --seed 0
```

### 5.2 运行 FedAvg + DDFed

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m ddfed_fl.fedavg_ddfed_experiment `
  --method fedavg_ddfed `
  --device cuda:0 `
  --num_rounds 5 `
  --local_epochs 1 `
  --participation_rate 0.2 `
  --threshold 5 `
  --num_decryptors 10 `
  --quantization_scale 100000 `
  --secure_pack_size 1 `
  --seed 0
```

---

## 6）Sanity Check（正确性检查）

用于比较明文聚合与安全聚合的数值误差：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& "E:\Conda\envs\myenv\python.exe" -m ddfed_fl.fedavg_ddfed_experiment `
  --method fedavg `
  --device cpu `
  --num_rounds 1 `
  --run_sanity_check `
  --num_decryptors 3 `
  --threshold 2 `
  --lambda_sec 32
```

输出示例：

- `sanity_check max_abs_error=...`

---

## 7）结果文件

实验结果默认保存到：

- `ddfed_fl/results/fedavg_baseline.csv`
- `ddfed_fl/results/fedavg_ddfed.csv`

每轮记录字段包含：

- `train_loss`
- `test_accuracy`
- `test_loss`
- `communication_time`
- `encryption_time`
- `partial_decryption_time`
- `combine_decryption_time`
- `round_total_time`
- `total_training_time`

---

## 8）注意事项

- 默认推荐 `secure_pack_size=1`，数值更稳定。
- 调大 `secure_pack_size` 可能提升速度，但可能增大误差。
- 若 CUDA 不可用，脚本会自动回退到 CPU。
