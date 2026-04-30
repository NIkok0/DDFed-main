# FedEraser (Federated Unlearning)

## 项目简介

FedEraser 是一个联邦学习遗忘框架，允许联邦学习系统中的客户端退出，并消除其数据对全局模型的影响。本项目实现了三种联邦遗忘方法：

| 方法 | 描述 |
|------|------|
| **FedEraser** | 使用未遗忘客户端的模型参数作为全局模型迭代的步长，以新全局模型为起点进行少量训练，将新客户端模型作为新全局模型迭代的方向 |
| **Unlearning without Cali** | 直接使用标准联邦学习中保存的每轮本地模型，移除被遗忘客户端的模型后直接聚合其他客户端模型 |
| **Retrain** | 重新训练，不使用需要被遗忘的客户端数据 |

此外，本项目还提供成员推断攻击（Membership Inference Attack）功能，用于评估被遗忘客户端的数据是否真正被遗忘。

---

## 支持的数据集

| 数据集 | 说明 |
|--------|------|
| `mnist` | MNIST 手写数字识别 |
| `fashion-mnist` | Fashion-MNIST 服装分类 |
| `cifar10` | CIFAR-10 图像分类 |
| `imagenet` | ImageNet（需自行准备数据） |
| `purchase` | Purchase 数据集（需自行准备） |
| `adult` | Adult 收入数据集（需自行准备） |

---

## 环境要求

- Python 3.8+
- PyTorch 1.6+
- scikit-learn 0.23+
- NumPy 1.18+
- SciPy 1.5+

---

## 快速开始

### 1. 单个实验运行

```bash
# 激活 conda 环境
conda activate myenv

# 运行 Fashion-MNIST 实验
python "Federated unlearning/FedEraser-Code/Fed_Unlearn_main.py" --data_name fashion-mnist

# 运行 ImageNet 实验
python "Federated unlearning/FedEraser-Code/Fed_Unlearn_main.py" --data_name imagenet
```

### 2. 批量实验运行（调度脚本）

使用调度脚本可以并行运行多个实验，支持控制变量法：

```bash
python "Federated unlearning/FedEraser-Code/run_experiments.py"
```

默认配置：
- 数据集：fashion-mnist, imagenet
- 客户端数量：10, 20, 30, 40, 50
- 共 10 个实验并行执行

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_name` | mnist | 数据集名称 |
| `--n_clients` | 10 | 每轮参与的客户端数量 |
| `--n_total_clients` | 100 | 总客户端数量 |
| `--global_epoch` | 20 | 全局训练轮数 |
| `--local_epoch` | 10 | 本地训练轮数 |
| `--local_lr` | 0.005 | 本地学习率 |
| `--local_batch_size` | 64 | 本地批次大小 |
| `--seed` | 1 | 随机种子 |

### 示例命令

```bash
# 使用 20 个客户端运行 Fashion-MNIST
python Fed_Unlearn_main.py --data_name fashion-mnist --n_clients 20

# 使用 50 个客户端运行 CIFAR-10，全局训练 50 轮
python Fed_Unlearn_main.py --data_name cifar10 --n_clients 50 --global_epoch 50
```

---

## 文件结构

```
FedEraser-Code/
├── Fed_Unlearn_main.py      # 主程序入口
├── Fed_Unlearn_base.py     # 联邦遗忘核心算法
├── FL_base.py              # 联邦学习基础函数（FedAvg等）
├── data_preprocess.py      # 数据预处理
├── model_initiation.py     # 模型定义
├── membership_inference.py # 成员推断攻击
├── run_experiments.py      # 批量实验调度脚本
├── README.md               # 本说明文档
└── data/                   # 数据目录（自动下载）
```

---

## 实验结果

实验运行后，结果会保存在 `logs/` 目录下：
- 每个实验的日志文件（`.log`）
- 汇总结果文件（`experiment_results_*.txt`）

---

## 自定义实验

### 修改调度脚本配置

编辑 `run_experiments.py` 顶部的配置：

```python
# 实验配置
DATASETS = ["fashion-mnist", "imagenet"]  # 可添加更多数据集
CLIENT_COUNTS = [10, 20, 30, 40, 50]      # 可调整客户端数量
```

### 添加新数据集

1. 在 `data_preprocess.py` 的 `data_set()` 函数中添加数据加载代码
2. 在 `model_initiation.py` 的 `model_init()` 函数中添加对应的模型结构

---

## 注意事项

1. **ImageNet 数据集**：需要自行下载并放置在 `./data/imagenet/train` 和 `./data/imagenet/val` 目录下
2. **内存要求**：较大的客户端数量和数据集可能需要更多内存
3. **并行实验**：调度脚本会根据 CPU 核心数自动限制并行任务数，避免内存溢出




