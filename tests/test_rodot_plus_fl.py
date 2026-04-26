# DDFed/tests/test_rodot_plus_fl.py

import sys
import os
import gmpy2
import random

# 1. 动态添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from ddfed_crypto.rodot_plus import RodotPlus
from ddfed_crypto import config

# ================= FL 配置 =================
NUM_EPOCHS = 5  # 设定的联邦学习总轮数
# ===========================================

if __name__ == "__main__":
    print("\n====== 开始测试 Rodot+ (联邦学习多轮场景) ======")

    rodot = RodotPlus()
    num_encryptors = config.N_ENCRYPTORS
    num_decryptors = config.N_DECRYPTORS
    all_encryptors = list(range(1, num_encryptors + 1))
    all_decryptors = list(range(1, num_decryptors + 1))

    # =========================================================================
    # 第一阶段：系统初始化 (FL 生命周期中仅执行一次)
    # =========================================================================
    print("\n[初始化] 正在执行 Setup & KGen...")
    rodot.setup()

    sk_dict = {}
    k_dict = {i: gmpy2.mpz(1) for i in all_encryptors}  # 权重默认为1
    # 在实际FL中，这里的权重设置为各方占比通分后的分子，最终的解密结果要除以分母。
    # 例如 三方归一化后占比分别是 1/2 1/3 1/6，这里的权重是 3 2 1，最后解密结果除以 6
    for i in all_encryptors:
        sk_dict[i] = rodot.kgen(i)

    print("[初始化] 正在执行 DKShare & DKCom (分发解密份额)...")
    # 模拟加密节点在密钥分发阶段掉线 (U_K 集合，仅在初始化时确定一次)
    dropped_k_nodes = set(random.sample(all_encryptors, config.N_DROPPED_K))
    active_k_nodes = [i for i in all_encryptors if i not in dropped_k_nodes]
    active_k_dict = {i: k_dict[i] for i in active_k_nodes}

    # 执行 DKShare
    all_dk_shares = {}
    for i in active_k_nodes:
        all_dk_shares[i] = rodot.dkshare(sk_dict[i], k_dict[i])

    # 转换数据维度并执行 DKCom
    shares_per_decryptor = {j: {} for j in all_decryptors}
    for i in active_k_nodes:
        for j in all_decryptors:
            shares_per_decryptor[j][i] = all_dk_shares[i][j]

    all_dk_j = {}
    for j in all_decryptors:
        all_dk_j[j] = rodot.dkcom(shares_per_decryptor[j], active_k_dict)

    print(f"[初始化完成] 成功建立加密环境！U_K 集合大小: {len(active_k_nodes)}\n")

    # =========================================================================
    # 第二阶段：联邦学习多轮训练循环 (每轮执行 Enc -> ParDec -> ComDec)
    # =========================================================================
    for epoch in range(1, NUM_EPOCHS + 1):
        label_l = f"Epoch_{epoch}"
        print(f"--------------------------------------------------")
        print(f"[*] 开始执行 {label_l} 训练循环 ...")

        # 1. 加密节点计算模型更新量并加密 (Enc 阶段)
        # 注意: 每轮训练中，加密节点可能因为网络原因掉线，所以 U_M 集合每轮重新随机生成
        dropped_m_nodes = set(random.sample(all_encryptors, config.N_DROPPED_M))
        active_m_nodes = [i for i in all_encryptors if i not in dropped_m_nodes]

        # 【FL 模型数据接入点】:
        # 这里的 x_i 代表加密节点 i 在本轮本地训练得到的模型梯度/更新量。
        # 后续接入真实 FL 时，将这里替换为将浮点数梯度编码、量化为整数的逻辑。
        # 且需要先进行明文打包再拿来加密，也就是把多个小明文编码成一个大明文，以减少总加密次数
        # 目前测试为了验证正确性，设定为 i * epoch * 10
        x_dict = {i: gmpy2.mpz(i * epoch * 10) for i in all_encryptors}

        all_cts = {}
        for i in active_m_nodes:
            all_cts[i] = rodot.enc(sk_dict[i], x_dict[i], label_l)

        # 2. 解密节点接收密文并进行部分解密 (ParDec 阶段)
        # 注意: 每轮通信中，解密节点也可能掉线，所以 U_D' 集合每轮重新随机生成
        dropped_d_nodes = set(random.sample(all_decryptors, config.N_DROPPED_D))
        active_d_nodes = [j for j in all_decryptors if j not in dropped_d_nodes]

        if len(active_d_nodes) < config.T_THRESHOLD:
            print(f"  [!] {label_l} 失败: 存活解密节点数({len(active_d_nodes)})低于阈值({config.T_THRESHOLD})")
            continue

        all_y_j = {}
        for j in active_d_nodes:
            all_y_j[j] = rodot.pardec(all_dk_j[j], active_k_dict, all_cts, label_l)

        # 3. 聚合中心重构最终聚合模型 (ComDec 阶段)
        u_m_intersect_u_k = set(active_m_nodes).intersection(set(active_k_nodes))
        final_y = rodot.comdec(all_y_j, active_k_dict, u_m_intersect_u_k)

        # --- 验证正确性 ---
        expected_y_prime = sum(k_dict[i] * x_dict[i] for i in u_m_intersect_u_k)
        sum_k_uk = sum(k_dict[i] for i in active_k_nodes)
        sum_k_intersect = sum(k_dict[i] for i in u_m_intersect_u_k)
        expected_y = (expected_y_prime * sum_k_uk) // sum_k_intersect

        if final_y == expected_y:
            print(f"  [✔] {label_l} 聚合解密成功！")
            print(f"      参与本轮加密的节点数(U_M): {len(active_m_nodes)}")
            print(f"      参与本轮解密的节点数(U_D'): {len(active_d_nodes)}")
            print(f"      => 提取的聚合模型更新量: {final_y}")
        else:
            print(f"  [✘] {label_l} 聚合解密失败！")
            print(f"      期望值: {expected_y}, 实际值: {final_y}")

    print(f"--------------------------------------------------")
    print("\n====== 联邦学习多轮测试完毕 ======")
