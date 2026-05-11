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