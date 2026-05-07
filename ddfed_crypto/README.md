# DDFed: Dynamic SA for Decentralized FL & FU via FE

本项目是完全去中心化联邦学习与遗忘（Fully Decentralized FL & FU）方案的 Python 实现。

目前包含所用的核心 FDFE 方案 **Rodot+**，以及用于实验对比评估（Evaluation）的基线方案。

方案基于未知阶群（Paillier 密文域）和整数上的 Shamir 秘密共享（Secret Sharing over the Integers, SSoI）构建，无需任何可信第三方（Trusted Dealer），支持在高度不稳定的网络环境中进行动态的安全模型聚合。

---

## 📑 目录

- [📁 项目结构与文件说明](#project-structure)
- [🚀 环境配置指南 (仅支持 Linux)](#env-setup)
- [⚙️ 修改参数指南](#config-guide)
- [🚀 真实联邦学习 (FL) 框架集成指南](#fl-integration)

---

## <span id="project-structure"></span>📁 项目结构与文件说明

本项目主要分为密码学核心库 `ddfed_crypto` 和测试用例 `tests` 两部分：

### `ddfed_crypto/` (密码学核心模块)
* **`config.py`**: 密码层面的全局配置文件。存放了所有密码学的安全参数（如安全强度 $\lambda$）、网络拓扑参数（加密/解密节点总数、阈值 $t$）以及用于模拟环境的掉线率配置。
* **`math_utils.py`**: 基础数学工具箱。包含大素数生成、安全素数（Safe Prime）生成逻辑，以及将标签 $l$ 安全哈希映射到密文群 $\mathbb{Z}_{N^2}^*$ 的函数。
* **`shamir_ss.py`**: 门限秘密共享工具箱。同时包含了**纯整数上的秘密共享（SSoI）**（完全不取模，通过注入 $\Delta$ 修正和巨大随机高斯噪声边界保证统计安全）与**标准的有限域秘密共享**（Finite Field SS）两种并列的函数实现，便于扩展与对比。
* **`rodot_plus.py`**: 协议的核心类 `RodotPlus`。严格按照密码学顶会标准实现了协议的完整生命周期：
  - `setup`: 生成 RSA 模数与统一拒绝采样边界 $I$ 与预计算 $\Delta$。
  - `kgen`: 基于高斯分布 $\mathcal{D}_{\mathbb{Z},\sigma}$ 的私钥生成与拒绝采样。
  - `dkshare` / `dkcom`: 份额的生成与接收合并。
  - `enc`: 本地模型更新量的加密。
  - `pardec` / `comdec`: 基于同态抵消与拉格朗日重构（带有 $\Delta$ 乘子修正）的联合解密与聚合。
* **`baselines/`**: 存放密码学层的各类对比基线方案（Baseline）。
  - `tmcfe.py`: 阈值多客户端函数加密（TMCFE）对比方案。复现自论文 *TAPFed*，基于素数域的 DDH 假设构建。与 Rodot+ 不同，该方案高度依赖每轮动态生成的解密密钥（dk每轮生成），并内置了大步小步法（BSGS）用于在合并解密阶段求解离散对数。
  - `dmcfe_ip.py`: 完全去中心化多客户端函数加密（DMCFE-IP）对比方案。复现自 Qian 等人的论文 *Lepcat*，基于 DCR 假设构建。与 Rodot+ 极低开销的本地操作不同，该方案高度依赖客户端之间 $O(N)$ 复杂度的 Diffie-Hellman (DH) 掩码协商与数字签名，并在服务器端通过有限域拉格朗日重构来实现节点掉线容错（双掩码容错原理）。
  - `ddmcfe.py`: 动态分布式多客户端函数加密（DDMCFE）对比方案。基于双线性映射（Bilinear Pairings）构建，底层高度依赖斯坦福 PBC 库与 Charm-Crypto 引擎，且解密时需求解离散对数。

### `tests/` (测试与模拟模块)
* **`test_rodot_plus.py` (基础密码学流水线测试)**:
  - **功能**: 测试密码学底层公式的严密性与正确性。
  - **内容**: 单次穿透测试（所有阶段过一遍）。验证同态加法、纯整数拉格朗日插值、 $\Delta$ 乘子能否在代数层面上完美抵消并还原出正确的聚合整数。适合用于排查底层数学逻辑和参数边界 Bug。
* **`test_rodot_plus_fl.py` (FL 多轮动态场景测试)**:
  - **功能**: 测试系统在真实联邦学习（FL）生命周期中的健壮性。
  - **内容**: 严格区分了“一次性初始化阶段”（Setup, KGen, DKShare, DKCom）和“多轮迭代训练阶段”（Enc, ParDec, ComDec）。**每轮训练中动态模拟不同的加密节点和解密节点掉线**，验证跨 Epoch 的标签防重放机制以及动态集合聚合能力。
* **`ex_rodot_plus_time.py` (性能基准测试)**:
  - **功能**: 测量并评估算法本身的极限计算开销和稳定耗时。
  - **内容**: 随机掉线逻辑由config中参数设置控制（设置为0表示不掉线的理想环境）。通过可配置的多次循环运算（`NUM_ITERATIONS`），精确计算并输出“平均每个参与（存活）节点”在各个密码学阶段的纯计算耗时（单位：毫秒 ms）。此脚本的输出数据可直接用于论文的 Evaluation 性能对比图表撰写。
* **`ex_tmcfe_time.py` (TMCFE 对比方案性能测试)**:
  - **功能**: 测试对比方案 TMCFE 的极限计算开销，与 Rodot+ 形成对照。
  - **内容**: 运行该脚本可直观展示 TMCFE 的生命周期结构差异，并暴露其性能瓶颈：基础设施需要在每轮动态生成功能密钥（`DKGenerate`），且客户端本地在恢复聚合模型（`CombineDecrypt`）时需进行极其耗时的离散对数搜索（BSGS），为论文的性能优势论证提供核心数据支撑。
* **`ex_dmcfe_ip_time.py` (DMCFE-IP 对比方案性能测试)**:
  - **功能**: 评估基线方案 DMCFE-IP 的极限计算开销，以及应对掉线攻击时的容错开销。
  - **内容**: 严格对齐原论文的生命周期逻辑。支持通过调整顶部参数 (`DROP_COUNT`) 无缝切换“理想无掉线环境（=0）”与“恶劣掉线环境（≠0）”。运行结果可直观暴露该方案将 $O(N)$ 级的协商与签名算力负担转嫁给客户端的瓶颈，以及在发生节点掉线时，服务器端必须执行拉格朗日插值重构掩码所带来的显著性能惩罚。
* **`ex_ddmcfe_time.py` (DDMCFE 对比方案性能测试)**:
  - **功能**: 测试基于配对密码学的对比方案 DDMCFE 的极限计算开销。
  - **内容**: 运行此脚本必须确保已在 Linux 环境下正确编译安装了 Charm-Crypto 引擎。该脚本主要用于测量各个生命周期阶段的耗时，通过测试结果将直观暴露出该方案在联合解密阶段（Decrypt）由于需要执行离散对数搜索（BSGS）而产生的严重计算瓶颈，为论证 Rodot+ 的极速性能提供跨流派的数据支撑。

---

## <span id="env-setup"></span>🚀 环境配置指南 (仅支持 Linux)

本项目中 **部分对比方案（如 DDMCFE）** 依赖高性能的 C++ 密码学引擎（基于 PBC 配对库）。由于底层编译机制的限制，**本项目仅支持在 Linux 或 macOS 环境下运行**，请勿在 Windows 下强行配置。

⚠️ 若无需这些对比方案可直接在环境中```pip install -r requirements.txt```。

为保证密码学运算的极速性能，请严格按照以下步骤从源码构建基础环境：

### 第一步：安装基础开发工具与 Python 3.9
建议使用 **python3.9**，不保证支持其他版本。请在终端执行：

```bash
# 1. 更新系统包列表
sudo apt-get update

# 2. 安装基础 C++ 编译工具（编译底层库必需）
sudo apt-get install -y gcc g++ make build-essential libgmp-dev libssl-dev flex bison wget unzip

# 3. 添加 Python 3.9 软件源并安装
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev
```

### 第二步：手工编译安装 PBC 库

libpbc-dev 在很多系统源中缺失或不完整，必须通过源码编译以确保头文件（pbc.h）正确安装。

```bash
# 1. 回到家目录下载源码（避免在挂载盘操作权限问题）
cd ~
wget https://crypto.stanford.edu/pbc/files/pbc-0.5.14.tar.gz

# 2. 解压并进入目录
tar -xvf pbc-0.5.14.tar.gz
cd pbc-0.5.14

# 3. 配置、编译、安装（若提示权限不足请加 sudo）
./configure
make
sudo make install

# 4. 更新动态链接库缓存
sudo ldconfig
```

### 第三步：配置项目虚拟环境与“降级”构建工具

这是最容易被忽视的一步。最新的安装工具无法识别老旧的密码学框架，必须进行“降级”处理。

```bash
# 1. 进入你的项目根目录 (DDFed)
cd /你的项目路径/DDFed

# 2. 创建并激活 Python 3.9 虚拟环境
python3.9 -m venv venv
source venv/bin/activate

# 3. 核心步骤：升级 pip 并强制“降级” setuptools（必须低于 58 版本）
pip install --upgrade pip
pip install "setuptools<58.0.0" wheel
```

### 第四步：编译安装 Charm-Crypto

建议直接下载源码并手动指定库路径进行编译，这是最稳妥的方式。

```bash
# 1. 下载源码压缩包（若 GitHub 连接慢，可手动下载后放入项目目录）
wget https://github.com/JHUISI/charm/archive/refs/heads/dev.zip

# 2. 解压并进入目录
unzip dev.zip
cd charm-dev

# 3. 清理旧残余并重新配置
make clean
./configure.sh

# 4. 关键：手动指定 PBC 路径并执行安装（将 C++ 引擎装入 venv）
CFLAGS="-I/usr/local/include -I/usr/local/include/pbc" LDFLAGS="-L/usr/local/lib" python setup.py install

# 5. 回到项目根目录并清理临时文件
cd ..
rm -rf charm-dev dev.zip
```

### 第五步：安装剩余 Python 依赖

```bash
# 安装 requirements.txt 中的核心算法包
pip install -r requirements.txt
```

当上述步骤全部完成且无报错时，完成环境配置。

---

## <span id="config-guide"></span>⚙️ 修改参数指南

如果需要调整安全强度或网络拓扑，**只需修改 `ddfed_crypto/config.py` 文件**：

1. **调整密码学安全强度**:
   修改 `LAMBDA_SEC` (默认 128)。这会级联影响大素数的长度、高斯分布的 $\sigma$ 以及随机系数的采样边界。
2. **调整网络拓扑与容错率**:
   - `N_ENCRYPTORS`: 加密节点（FL Client / 训练者）总数。
   - `N_DECRYPTORS`: 解密节点（聚合服务器集群）总数。
   - `T_THRESHOLD`: 门限解密阈值 $t$（必须满足 $t \leq N_{DECRYPTORS}$）。
3. **模拟掉线测试**:
   调整 `N_DROPPED_K`, `N_DROPPED_M`, `N_DROPPED_D` 的值来测试系统在不同丢包率下的表现。（注意：实际存活的解密节点数不能低于 `T_THRESHOLD`，否则聚合会失败）。

---

## <span id="fl-integration"></span>🚀 真实联邦学习 (FL) 框架集成指南

本协议旨在作为安全聚合层（Secure Aggregation Layer）无缝嵌入到现有的 FL 框架中。集成时，请主要参考并修改 `test_rodot_plus_fl.py` 的第二阶段（多轮训练循环）逻辑。

### 集成指南与修改点：

#### 1. 浮点数梯度的编解码 (Quantization)
密码学算法（Paillier 或 DDH）只能在**整数域 $\mathbb{Z}$** 上运行，而深度学习模型的梯度/更新量（即 $x_i$）是**浮点数（Float）**。
* **加密前 (Enc 处修改)**: 
  
  每个加密节点在本地计算完模型差值之后，编码成大整数再加密；解密后解码回去。

#### 2. 明文打包加速 (Plaintext Packing)
由于联邦学习的模型更新通常包含成千上万的参数，逐个进行大数加密会导致极大的计算和通信延迟。基于 DCR（Paillier）的函数加密方案拥有远大于单个梯度空间的极大的明文空间（例如 2048 位）。因此，必须引入**明文打包技术**，将多个组件编码拼接成一个单一的明文进行批量处理。

* **加密前打包 (`PP.Encode`)**:

  具体可参考论文：Decentralized Multi-Client Functional Encryption  for Inner Product With Applications to  Federated Learning
  https://ieeexplore.ieee.org/document/10494860 中的明文打包