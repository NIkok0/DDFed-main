# DDFed-FU (Federated Unlearning)

联邦学习遗忘（Federated Unlearning）框架。整合了三种联邦遗忘方法：

| 子项目 | 方法 | 描述 |
|--------|------|------|
| **FedEraser** | 参数校准 | 通过移除被遗忘客户端后重新校准全局模型，实现精准数据遗忘 |
| **Backdoor Unlearning** | 后门擦除 | 在联邦学习中投毒并擦除后门，实现"Get Rid Of Your Trail" |
| **QuickDrop** | 数据蒸馏 | 利用合成数据快速执行目标客户端的遗忘与恢复 |

---

## 目录结构

```
ddfed_fu/
├── common/                          # 公共配置与 CLI 解析
│   ├── __init__.py
│   ├── cli_parser.py
│   └── base_config.py
├── baselines/                       # 基线方法
├── check_point/                     # 预训练模型检查点
├── save/                            # 实验结果保存
├── result/                          # 可视化结果
│
├── FedEraser-Code/                  # [FedEraser] Federated Unlearning
│   ├── Fed_Unlearn_main.py          # 主入口
│   ├── Fed_Unlearn_base.py          # 遗忘核心算法
│   ├── FL_base.py                   # FedAvg 基础
│   ├── data_preprocess.py           # 数据预处理
│   ├── model_initiation.py          # 模型定义
│   ├── membership_inference.py      # 成员推断攻击
│   ├── run_experiments.py           # 批量实验调度
│   └── README.md                    # 详细文档
│
├── federated_backdoor_unlearning-main/  # [Backdoor] 后门遗忘
│   ├── main.py                      # 主入口
│   ├── unlearning.py                # 遗忘算法
│   ├── neurotoxin.py                # 毒素注入
│   ├── vgg_model.py                 # VGG 模型
│   ├── config.py                    # 配置参数
│   ├── data_loader.py               # 数据加载
│   ├── argument_parser.py           # 参数解析
│   ├── plot.py                      # 结果可视化
│   ├── utils.py                     # 工具函数
│   ├── run_fashion_and_imagenet.py  # 扩展数据集实验
│   ├── examples/                    # 示例日志
│   └── README.md                    # 详细文档
│
├── quickdrop-main/                  # [QuickDrop] 快速丢弃
│   ├── FedQuickDrop/                # QuickDrop 服务器与客户端
│   ├── FedAvg/                      # 标准 FedAvg
│   ├── FedNaiveRetrain/             # 朴素重训练
│   ├── FedRecover/                  # 恢复
│   ├── FedSGA/                      # 随机梯度上升
│   ├── env_generator/               # FL 环境生成（DiLICHET等）
│   ├── reproduce/                   # 可复现实验
│   ├── merit/                       # 验证方法
│   ├── utils/                       # 数据集、模型等
│   └── readMe.md                    # 详细文档
│
├── check_env.sh                     # 环境完整性检查
├── smoke_test_federaser.sh          # FedEraser 冒烟测试
├── smoke_test_backdoor.sh           # Backdoor 冒烟测试
├── smoke_test_quickdrop.sh          # QuickDrop 冒烟测试
├── smoke_test_all.sh                # 全集冒烟测试
└── README.md                        # 本文件
```

---

## 统一遗忘实验 (Unified Unlearn)

`unified_unlearn/` 是一个统一框架，在**相同的数据划分、模型架构、预训练 checkpoint** 下对比 5 种联邦遗忘算法。

### 5 个算法方案

| # | 算法名 | 类型 | 遗忘用户本地逻辑 |
|---|--------|------|------------------|
| 1 | `Baseline_FedSGA` | Baseline | 在 unlearnset（单类数据）上做 Gradient Ascent |
| 2 | `Baseline_Neurotoxin` | Baseline | `total_loss = clean_loss + gamma * backdoor_loss + beta * penalty`（backdoor_loss 对 target class 取负 CE） |
| 3 | `Baseline_FedQuickDrop` | Baseline | 合成数据蒸馏（DC/DSA）+ affine 操作消除影响 |
| 4 | `Proposed_ddfu_client` | Proposed | 与 Neurotoxin 完全相同的遗忘公式，应用于全部本地数据 |
| 5 | `Proposed_ddfu_sample` | Proposed | 前半数据 Neurotoxin 遗忘 → g₁；后半数据 FedAvg 训练 → g₂；合并 g = g₁/2 + g₂/2 上传 |

### 统一组件

| 组件 | 选择 |
|------|------|
| **模型架构** | ConvNet（net_width=64, net_depth=3, BatchNorm, MaxPool） |
| **数据集** | FashionMNIST + CIFAR-10 |
| **Non-IID 划分** | Dirichlet 分布，α=100（mild）/ α=0.1（strong） |
| **总用户 / 每轮参与** | 20 / 4 |
| **学习率 / 动量** | 0.01 / 0.5 |
| **Phase 1 预训练轮数** | 30 轮（T_unlearn） |
| **Phase 2 遗忘轮数** | 可配置（默认 20 轮） |

### 随机种子 & 公平性保证

- 同一 env（如 `fashionmnist-seed42-u20-alpha0.1`）下，5 个方案的**数据划分完全一致**
- Phase 1 预训练到第 30 轮时，**所有方案保存 checkpoint**，Phase 2 从此 checkpoint 出发
- 遗忘用户 index 由固定 seed 产生，同一 env 下所有方案选中同一用户
- Phase 2 每轮选中的 3 个正常用户轨迹也由可复现的随机序列保证一致

### 文件结构

```
unified_unlearn/
├── __init__.py
├── config.py                  # 统一配置（场景、超参数、方案选择）
├── model.py                   # ConvNet 模型定义
├── dataset.py                 # 数据集加载 + Dirichlet 划分 + DataLoader
├── server.py                  # Phase 1 FedAvg Server + Phase 2 遗忘 Server
├── client.py                  # 所有 Client：FedAvg, FedSGA, Neurotoxin, QuickDrop, ddfu
├── log_utils.py               # 日志记录（Round, Loss, Accuracy）
├── main.py                    # 主入口（CLI 单次实验）
└── run_all.sh                 # Shell 脚本：一键运行全部 20 组实验
```

### 环境要求

与 `ddfed_fu` 一致：

- **Python**: 3.8+
- **PyTorch**: 1.6+（推荐 2.0+）
- **torchvision**
- **CUDA**: 可选

### 快速开始

#### 单次实验

```bash
cd /home/hyr/miaoli/DDFed-main
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main \
    --dataset FashionMNIST \
    --alpha 0.1 \
    --algo Baseline_FedSGA \
    --pretrain 30 \
    --unlearn 20
```

#### 一键运行全部 20 组实验

```bash
cd /home/hyr/miaoli/DDFed-main
bash ddfed_fu/unified_unlearn/run_all.sh
```

该脚本遍历 5 个算法 × 2 个数据集 × 2 个 α 值 = 20 组实验，汇总通过/失败状态。

### 完整实验矩阵（20 组命令）

以下 20 条命令覆盖全部实验组合。每条命令可单独执行，也可由 `run_all.sh` 自动串行。

> **重要**：第一次运行某个 `(dataset, alpha)` 组合时，需要使用 `--force-pretrain` 来生成 Phase 1 预训练 checkpoint。后续相同 `(dataset, alpha)` 的算法会**自动复用**该 checkpoint，无需再加 `--force-pretrain`。

#### FashionMNIST, α=100.0（mild Non-IID）

```bash
cd /home/hyr/miaoli/DDFed-main
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 100.0 --algo Baseline_FedSGA      --pretrain 30 --unlearn 20 --force-pretrain
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 100.0 --algo Baseline_Neurotoxin  --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 100.0 --algo Baseline_FedQuickDrop   --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 100.0 --algo Proposed_ddfu_client --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 100.0 --algo Proposed_ddfu_sample --pretrain 30 --unlearn 20
```

#### FashionMNIST, α=0.1（strong Non-IID）

```bash
cd /home/hyr/miaoli/DDFed-main
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 0.1 --algo Baseline_FedSGA      --pretrain 30 --unlearn 20 --force-pretrain
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 0.1 --algo Baseline_Neurotoxin  --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 0.1 --algo Baseline_FedQuickDrop   --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 0.1 --algo Proposed_ddfu_client --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 0.1 --algo Proposed_ddfu_sample --pretrain 30 --unlearn 20
```

#### CIFAR-10, α=100.0（mild Non-IID）

```bash
cd /home/hyr/miaoli/DDFed-main
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 100.0 --algo Baseline_FedSGA      --pretrain 30 --unlearn 20 --force-pretrain
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 100.0 --algo Baseline_Neurotoxin  --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 100.0 --algo Baseline_FedQuickDrop   --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 100.0 --algo Proposed_ddfu_client --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 100.0 --algo Proposed_ddfu_sample --pretrain 30 --unlearn 20
```

#### CIFAR-10, α=0.1（strong Non-IID）

```bash
cd /home/hyr/miaoli/DDFed-main
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 0.1 --algo Baseline_FedSGA      --pretrain 30 --unlearn 20 --force-pretrain
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 0.1 --algo Baseline_Neurotoxin  --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 0.1 --algo Baseline_FedQuickDrop   --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 0.1 --algo Proposed_ddfu_client --pretrain 30 --unlearn 20
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 0.1 --algo Proposed_ddfu_sample --pretrain 30 --unlearn 20
```

#### 实验矩阵汇总

| 数据集 | α | Baseline_FedSGA | Baseline_Neurotoxin | Baseline_FedQuickDrop | Proposed_ddfu_client | Proposed_ddfu_sample |
|---|---|---|---|---|---|---|
| FashionMNIST | 100.0 | `--force-pretrain` | 复用 ckpt | 复用 ckpt | 复用 ckpt | 复用 ckpt |
| FashionMNIST | 0.1 | `--force-pretrain` | 复用 ckpt | 复用 ckpt | 复用 ckpt | 复用 ckpt |
| CIFAR10 | 100.0 | `--force-pretrain` | 复用 ckpt | 复用 ckpt | 复用 ckpt | 复用 ckpt |
| CIFAR10 | 0.1 | `--force-pretrain` | 复用 ckpt | 复用 ckpt | 复用 ckpt | 复用 ckpt |

#### run_all.sh 脚本流程

`run_all.sh` 自动遍历上述 20 组实验，核心逻辑：
1. 遍历 `DATASETS` → `ALPHAS` → `ALGOS` 三重循环
2. 每组调用 `python -m ddfed_fu.unified_unlearn.main --dataset ... --alpha ... --algo ... --pretrain 30 --unlearn 20`
3. 捕获退出码，统计通过/失败数量
4. 打印汇总 `Total: 20, Pass: N, Fail: M`

> **注意**：`run_all.sh` 不包含 `--force-pretrain`，因此需要**先手动运行 4 组带 `--force-pretrain` 的 Baseline_FedSGA**（每个 `(dataset, alpha)` 组合一次），生成 Phase 1 checkpoint。之后脚本可自动复用。

### 分步实验指南（推荐顺序）

**Step 1 — 确保环境正确**

```bash
cd /home/hyr/miaoli/DDFed-main
PYTHONPATH=. python -c "from ddfed_fu.unified_unlearn import main; print('OK')"
```

**Step 2 — 生成 4 个 Phase 1 checkpoint（每个 scenario 一次）**

```bash
cd /home/hyr/miaoli/DDFed-main

# FashionMNIST α=100
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 100.0 --algo Baseline_FedSGA --pretrain 30 --unlearn 20 --force-pretrain

# FashionMNIST α=0.1
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset FashionMNIST --alpha 0.1   --algo Baseline_FedSGA --pretrain 30 --unlearn 20 --force-pretrain

# CIFAR10 α=100
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 100.0 --algo Baseline_FedSGA --pretrain 30 --unlearn 20 --force-pretrain

# CIFAR10 α=0.1
PYTHONPATH=. python -m ddfed_fu.unified_unlearn.main --dataset CIFAR10 --alpha 0.1   --algo Baseline_FedSGA --pretrain 30 --unlearn 20 --force-pretrain
```

**Step 3 — 运行全部 20 组实验**

```bash
bash ddfed_fu/unified_unlearn/run_all.sh
```

**Step 4 — 查看结果**

结果 CSV 保存在 `result/unified_unlearn/` 目录下：
- `fmnist-a100_0_fedsga.csv`  → FashionMNIST, α=100, FedSGA
- `fmnist-a100_0_neurotoxin.csv` → FashionMNIST, α=100, Neurotoxin
- `fmist-a0_1_fedsga.csv`    → FashionMNIST, α=0.1, FedSGA
- ...
- `cifar10-a0_1_ddfu_sample.csv` → CIFAR10, α=0.1, ddfu_sample

### 运行注意事项

| 问题 | 说明 |
|------|------|
| **CUDA OOM** | 减小 `BATCH_SIZE`（在 `config.py` 第 26 行，默认 64）；QuickDrop 的 DSA 合成需额外 GPU 内存 |
| **QuickDrop + CIFAR10 显存不足** | 加上 `--no-dsa` 使用 DC（只合成单张图片），显存需求大幅降低 |
| **checkpoint 加载失败** | 确保 Step 2 先完成；checkpoint 路径为 `save/unified_unlearn/{scenario}_phase1.pt` |
| **同一 scenario 多算法数据划分一致** | 所有算法使用相同 `GLOBAL_SEED`（默认 42）生成相同的 Dirichlet 划分 |
| **遗忘用户选择一致** | Phase 2 的遗忘用户 index 由 `GLOBAL_SEED` 固定生成，同一 scenario 下所有算法遗忘同一用户 |
| **断点续跑** | 每个 `(dataset, alpha, algo)` 组合对应单独的 CSV 文件；已完成的算法可直接跳过，重新跑不会覆盖 Phase 1 checkpoint（除非加 `--force-pretrain`） |
| **Python 版本** | 建议 3.8+；`Backdoor Unlearning` 子项目需 3.12，但 `unified_unlearn` 不依赖该子项目 |
| **网络下载** | 首次运行会自动下载 FashionMNIST / CIFAR-10 数据集到 `data/` 目录 |

### CLI 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--dataset` | ✓ | — | `FashionMNIST` 或 `CIFAR10` |
| `--alpha` | ✓ | — | Dirichlet 分布 α 值（如 `100.0` / `0.1`） |
| `--algo` | ✓ | — | 算法名，支持：`Baseline_FedSGA`、`Baseline_Neurotoxin`、`Baseline_FedQuickDrop`、`Proposed_ddfu_client`、`Proposed_ddfu_sample`（及缩写 `fedsga`、`neurotoxin`、`quickdrop`、`ddfu_client`、`ddfu_sample`） |
| `--pretrain` | | `30` | Phase 1 预训练轮数 |
| `--unlearn` | | `20` | Phase 2 遗忘轮数 |
| `--seed` | | `42` | 随机种子 |
| `--force-pretrain` | | `False` | 强制重新预训练（否则优先从 checkpoint 加载） |
| `--no-dsa` | | `False` | QuickDrop 不使用 DSA 合成（使用 DC） |

### 输出结果

- **CSV 日志**：`result/unified_unlearn/{scenario_id}_{algo}.csv`
- **Checkpoint**：`save/unified_unlearn/{scenario_id}_phase1.pt`

其中 `scenario_id` 格式为 `fmnist-a100_0`（FashionMNIST α=100.0）或 `cifar10-a0_1`（CIFAR10 α=0.1）。
> 注意：结果 CSV 保存在 `result/unified_unlearn/` 下，Phase 1 checkpoint 保存在 `save/unified_unlearn/` 下（而非 `check_point/`）。

CSV 列名：`round, phase, test_acc, test_loss, forget_acc, forget_loss, remain_acc, remain_loss`

- `phase=1`：预训练阶段日志
- `phase=2`：遗忘阶段日志
- `forget_*`：遗忘用户数据的准确率/损失
- `remain_*`：保留用户数据的准确率/损失

---

## 环境要求

- **Python**: 3.8+（Backdoor 子项目建议 3.12）
- **PyTorch**: 1.6+（推荐 2.0+）
- **CUDA**: 可选（12.4 已测试通过）

### 依赖安装

```bash
# FedEraser
pip install -r FedEraser-Code/../requirements.txt

# Backdoor Unlearning
pip install -r federated_backdoor_unlearning-main/requirements.txt

# QuickDrop
pip install -r quickdrop-main/conda_env/requirements.txt
```

### 环境检查

```bash
bash check_env.sh
```

---

## 快速开始

### 1. FedEraser — 联邦遗忘

```bash
# Fashion-MNIST 实验
python FedEraser-Code/Fed_Unlearn_main.py --data_name fashion-mnist

# CIFAR-10 实验
python FedEraser-Code/Fed_Unlearn_main.py --data_name cifar10 --n_clients 20 --global_epoch 50
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_name` | mnist | 数据集（mnist / fashion-mnist / cifar10 / imagenet / purchase / adult） |
| `--n_clients` | 10 | 每轮参与客户端数 |
| `--n_total_clients` | 100 | 总客户端数 |
| `--global_epoch` | 20 | 全局训练轮数 |
| `--local_epoch` | 10 | 本地训练轮数 |
| `--local_lr` | 0.005 | 本地学习率 |
| `--local_batch_size` | 64 | 本地批次大小 |
| `--seed` | 1 | 随机种子 |

### 2. Backdoor Unlearning — 后门擦除

```bash
# 标准训练（无后门）
python federated_backdoor_unlearning-main/main.py --num_rounds 3000

# 投毒模式
python federated_backdoor_unlearning-main/main.py \
    --num_rounds 3000 \
    --poison \
    --poison_strategy random \
    --poison_start_round 300 \
    --poison_duration 100

# 投毒 + 遗忘
python federated_backdoor_unlearning-main/main.py \
    --num_rounds 3000 \
    --poison \
    --poison_strategy random \
    --poison_start_round 300 \
    --poison_duration 100 \
    --unlearn \
    --unlearn_duration 30
```

**关键参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `--num_rounds` | int | 总通信轮数 |
| `--poison` | flag | 启用后门投毒 |
| `--unlearn` | flag | 启用遗忘 |
| `--poison_strategy` | str | 投毒策略（continuous / fixed / random） |
| `--poison_start_round` | int | 投毒开始轮次 |
| `--poison_duration` | int | 投毒持续轮数 |
| `--unlearn_duration` | int | 遗忘持续轮数 |

### 3. QuickDrop — 快速丢弃

QuickDrop 采用三步流程：

1. **生成 FL 环境**（DiLICHET 非独立同分布分配）
2. **训练/加载基线模型**
3. **执行遗忘 + 恢复**

```bash
# Step 1: 生成环境
cd quickdrop-main
python env_generator/dilichlet_allocator/dilichlet_allocator.py \
    --dataset_name CIFAR10 \
    --num_clients 20 \
    --alpha 0.1 \
    --seed 42

# Step 2+3: 运行遗忘实验
cd reproduce/fig2_with_recovering/code
python reproduce_single_unlearning.py \
    --dataset CIFAR10 \
    --model ConvNet \
    --env CIFAR-10-seed42-u20-alpha0.1 \
    --strategy quickdrop-affine \
    --device cpu \
    --communication_round 200 \
    --local_epoch 5 \
    --forgetting_epoch 1 \
    --recovering_round 5 \
    --seed 42
```

> 详细步骤见 `quickdrop-main/readMe.md`

---

## 数据集支持

| 数据集 | FedEraser | Backdoor | QuickDrop |
|--------|:---------:|:--------:|:---------:|
| MNIST | ✓ | | ✓ |
| Fashion-MNIST | ✓ | | |
| CIFAR-10 | ✓ | ✓ | ✓ |
| CIFAR-100 | | | ✓ |
| ImageNet | ✓ | | |
| SVHN | | | ✓ |
| Purchase | ✓ | | |
| Adult | ✓ | | |

---

## 冒烟测试

四个冒烟测试脚本确保代码库在最小配置下可运行：

```bash
# 单独运行
bash smoke_test_federaser.sh       # FedEraser (MNIST, 2 轮)
bash smoke_test_backdoor.sh        # Backdoor (CIFAR-10, 5 轮)
bash smoke_test_quickdrop.sh       # QuickDrop (MNIST, 2 轮)

# 运行全部
bash smoke_test_all.sh
```

冒烟测试结果会汇总显示通过/失败状态。

---

## 实验复现

- **FedEraser**: `python FedEraser-Code/run_experiments.py`
- **Backdoor**: 参考 `federated_backdoor_unlearning-main/examples/` 中的示例日志
- **QuickDrop**: `cd quickdrop-main/reproduce/fig2_with_recovering/code && python reproduce_single_unlearning.py`

---

## 引用

- **FedEraser**: Liu et al., "FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models", IEEE IWQoS 2021.
- **Backdoor Unlearning**: Alam et al., "Get Rid Of Your Trail: Remotely Erasing Backdoors in Federated Learning", IEEE TAI 2024.
- **QuickDrop**: 见 `quickdrop-main/readMe.md`

---

## 联系方式

| 子项目 | 联系人 |
|--------|--------|
| FedEraser | 见 `FedEraser-Code/README.md` |
| Backdoor Unlearning | alam.manaar@nyu.edu |
| QuickDrop | 见 `quickdrop-main/readMe.md` |