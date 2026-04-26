# DDFed/tests/test_rodot_plus.py

import sys
import os
import gmpy2
import random

# 1. 动态添加项目根目录到 Python 路径
# 获取当前测试文件所在的绝对路径的上一级（即 DDFed 根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. 导入 ddfed_crypto 包
from ddfed_crypto.rodot_plus import RodotPlus
from ddfed_crypto import config

if __name__ == "__main__":
    print("\n====== 开始测试 Rodot+ ======")

    # 实例化方案
    rodot = RodotPlus()

    # 1. 执行 Setup
    print(f"\n[*] 开始执行 Setup 阶段...")
    pp = rodot.setup()

    print("\n[+] 生成的公开参数 (pp):")
    for k, v in pp.items():
        val_str = str(v)
        # 对于超长的大整数，前后截断显示以便于查看
        if len(val_str) > 50:
            val_str = val_str[:20] + " ... " + val_str[-20:]
        print(f"  {k}: {val_str}")
    print("\n[*] 公开参数生成完毕.")

    # 2. 执行 KGen
    print("\n[*] 开始执行 KGen 阶段...")
    num_users = config.N_ENCRYPTORS  # 从 config 中读取节点数量
    sk_dict = {}  # 字典用于在内存中保存这50个私钥，格式 {i: sk_i}

    for i in range(1, num_users + 1):
        # 传入节点编号 i
        sk_i = rodot.kgen(i)
        sk_dict[i] = sk_i

        # 截断显示，避免控制台被大数刷屏
        if i <= 3 or i == num_users:
            val_str = str(sk_i)
            if len(val_str) > 50:
                val_str = val_str[:20] + " ... " + val_str[-20:]
            print(f"  -> 成功为节点 {i} 生成私钥 sk_{i}: {val_str}")
        elif i == 4:
            print("  -> ... (省略中间输出) ...")

    print(f"\n[*] 成功为 {num_users} 个节点生成了私钥！")

    # 3. 执行 DKShare (模拟 K 阶段掉线)
    print("\n[*] 开始执行 DKShare 阶段...")

    # 所有加密节点的 k_i 设为 1
    k_dict = {i: gmpy2.mpz(1) for i in range(1, num_users + 1)}
    all_encryptors = list(range(1, num_users + 1))

    # 根据 config.N_DROPPED_K 随机挑选掉线节点
    dropped_k_nodes = set(random.sample(all_encryptors, config.N_DROPPED_K))
    if config.N_DROPPED_K > 0:
        print(f"  [!] 模拟以下 {config.N_DROPPED_K} 个加密节点在 DKShare 阶段掉线: {sorted(list(dropped_k_nodes))}")
    else:
        print("  [!] 本次测试无加密节点在 DKShare 阶段掉线。")

    # U_K 集合：存活并成功发送份额的加密节点
    active_k_nodes = [i for i in all_encryptors if i not in dropped_k_nodes]

    # 存储存活节点发给解密节点的份额，keys 就天然代表了 U_K 集合
    all_dk_shares = {}

    for i in active_k_nodes:
        sk_i = sk_dict[i]
        k_i = k_dict[i]

        # 调用 DKShare 生成份额
        dk_shares_i = rodot.dkshare(sk_i, k_i)
        all_dk_shares[i] = dk_shares_i

        # 为了控制台简洁，只打印部分存活节点的份额信息
        if i == active_k_nodes[0] or i == active_k_nodes[-1]:
            print(f"  -> 存活加密节点 {i} (k_{i}={k_i}) 生成的解密份额如下:")
            for j, share in dk_shares_i.items():
                if j <= 2 or j == config.N_DECRYPTORS:
                    val_str = str(share)
                    if len(val_str) > 50:
                        val_str = val_str[:20] + " ... " + val_str[-20:]
                    print(f"     给解密节点 {j} 的份额 dk_{{{i},{j}}}: {val_str}")
                elif j == 3:
                    print("     ... (省略中间解密节点份额) ...")
        elif i == active_k_nodes[1]:
            print("  -> ... (省略中间存活加密节点的输出) ...")

    print(f"\n[*] 成功为 {len(active_k_nodes)} 个存活加密节点完成了 DKShare 份额分发！(U_K 集合大小: {len(all_dk_shares)})")

    # 4. 执行 Enc 阶段 (模拟 M 阶段掉线)
    print("\n[*] 开始执行 Enc 阶段...")

    # 设定一个全局共享的聚合标签 l
    label_l = "Epoch_1"

    # 模拟每个节点的原始数据 x_i (测试中简单设为 i * 10)
    x_dict = {i: gmpy2.mpz(i * 10) for i in all_encryptors}

    # 根据 config.N_DROPPED_M 随机挑选掉线节点
    dropped_m_nodes = set(random.sample(all_encryptors, config.N_DROPPED_M))
    if config.N_DROPPED_M > 0:
        print(f"  [!] 模拟 {config.N_DROPPED_M} 个加密节点在 Enc 阶段掉线: {sorted(list(dropped_m_nodes))}")
    else:
        print("  [!] 本次测试无加密节点在 Enc 阶段掉线。")

    # U_M 集合：存活并成功发送密文的加密节点
    active_m_nodes = [i for i in all_encryptors if i not in dropped_m_nodes]

    # 存储存活节点生成的密文，keys 天然代表了 U_M 集合
    all_cts = {}

    for i in active_m_nodes:
        sk_i = sk_dict[i]
        x_i = x_dict[i]

        # 调用 Enc 生成密文
        ct_i = rodot.enc(sk_i, x_i, label_l)
        all_cts[i] = ct_i

        # 为了控制台简洁，只打印部分存活节点的密文信息
        if i == active_m_nodes[0] or i == active_m_nodes[-1]:
            val_str = str(ct_i)
            if len(val_str) > 50:
                val_str = val_str[:20] + " ... " + val_str[-20:]
            print(f"  -> 存活加密节点 {i} (x_{i}={x_i}) 生成密文 ct_{i}: {val_str}")
        elif i == active_m_nodes[1]:
            print("  -> ... (省略中间存活加密节点的密文输出) ...")

    print(f"\n[*] 成功为 {len(active_m_nodes)} 个存活加密节点完成了 Enc 加密！(U_M 集合大小: {len(all_cts)})")

    # 5. 执行 DKCom 阶段
    print("\n[*] 开始执行 DKCom 阶段 ...")

    # 数据维度转换：从 all_dk_shares[加密节点 i][解密节点 j]
    # 转换为 shares_per_decryptor[解密节点 j][加密节点 i]
    all_decryptors = list(range(1, config.N_DECRYPTORS + 1))
    shares_per_decryptor = {j: {} for j in all_decryptors}

    for i in active_k_nodes:
        for j in all_decryptors:
            shares_per_decryptor[j][i] = all_dk_shares[i][j]

    # 提取 U_K 集合对应的 k_i 字典
    active_k_dict = {i: k_dict[i] for i in active_k_nodes}

    # 存储每个解密节点处理后的 dk_j
    all_dk_j = {}

    for j in all_decryptors:
        # 解密节点 j 执行 DKCom
        dk_j = rodot.dkcom(shares_per_decryptor[j], active_k_dict)
        all_dk_j[j] = dk_j

        # 仅打印首尾解密节点的处理结果
        if j == 1 or j == config.N_DECRYPTORS:
            print(f"  -> 解密节点 {j} 成功处理了来自 U_K 集合的份额 (共 {len(dk_j)} 份)。")

            # 直接使用 [0] 打印来自第一个存活加密节点的份额
            val_first = str(dk_j[active_k_nodes[0]])
            if len(val_first) > 50:
                val_first = val_first[:20] + " ... " + val_first[-20:]
            print(f"     来自首个存活加密节点 {active_k_nodes[0]} 的调整后份额: {val_first}")

            print("     ... (省略中间加密节点的调整后份额) ...")

            # 直接使用 [-1] 打印来自最后一个存活加密节点的份额
            val_last = str(dk_j[active_k_nodes[-1]])
            if len(val_last) > 50:
                val_last = val_last[:20] + " ... " + val_last[-20:]
            print(f"     来自末尾存活加密节点 {active_k_nodes[-1]} 的调整后份额: {val_last}")

        elif j == 2:
            print("  -> ... (省略中间解密节点的处理输出) ...")

    print(f"\n[*] 成功为 {config.N_DECRYPTORS} 个解密节点完成了 DKCom 份额调整！")

    # 6. 执行 ParDec 阶段
    print("\n[*] 开始执行 ParDec 阶段 ...")

    # 模拟 U_D' 集合 (参与部分解密的解密节点，模拟 D 阶段掉线)
    dropped_d_nodes = set(random.sample(all_decryptors, config.N_DROPPED_D))
    if config.N_DROPPED_D > 0:
        print(f"  [!] 模拟 {config.N_DROPPED_D} 个解密节点在 ParDec 阶段掉线: {sorted(list(dropped_d_nodes))}")
    else:
        print("  [!] 本次测试无解密节点在 ParDec 阶段掉线。")

    active_d_nodes = [j for j in all_decryptors if j not in dropped_d_nodes]

    # 确保存活的解密节点数满足阈值 t，否则最后的 ComDec 会失败
    if len(active_d_nodes) < config.T_THRESHOLD:
        print(
            f"  [!] 警告: 存活解密节点数 ({len(active_d_nodes)}) 小于阈值 t ({config.T_THRESHOLD})，后续 ComDec 将失败！")

    # 存储每个存活解密节点的部分解密结果 y_j
    all_y_j = {}

    for j in active_d_nodes:
        dk_j = all_dk_j[j]  # 节点 j 的调整后份额

        # 解密节点 j 计算部分解密结果
        y_j = rodot.pardec(dk_j, active_k_dict, all_cts, label_l)
        all_y_j[j] = y_j

        # 仅打印首尾存活解密节点的部分解密结果
        if j == active_d_nodes[0] or j == active_d_nodes[-1]:
            val_str = str(y_j)
            if len(val_str) > 50:
                val_str = val_str[:20] + " ... " + val_str[-20:]
            print(f"  -> 存活解密节点 {j} 成功生成部分解密结果 y_{j}: {val_str}")
        elif j == active_d_nodes[1]:
            print("  -> ... (省略中间存活解密节点的部分解密输出) ...")

    print(f"\n[*] 成功为 {len(active_d_nodes)} 个存活解密节点完成了 ParDec 部分解密！(U_D' 集合大小: {len(all_y_j)})")

    # 7. 执行 ComDec 阶段
    print("\n[*] 开始执行 ComDec 阶段 ...")

    # 提取交集 U_M \cap U_K
    u_m_intersect_u_k = set(active_m_nodes).intersection(set(active_k_nodes))

    # 调用 ComDec
    final_y = rodot.comdec(all_y_j, active_k_dict, u_m_intersect_u_k)
    print(f"\n[+] 系统解密出的最终结果 y: {final_y}")

    # --- 验证正确性 ---
    print("\n[+] 正在计算理论期望结果进行对比...")
    # 理论上的 y' = \sum_{i \in 交集} k_i * x_i
    expected_y_prime = sum(k_dict[i] * x_dict[i] for i in u_m_intersect_u_k)
    sum_k_uk = sum(k_dict[i] for i in active_k_nodes)
    sum_k_intersect = sum(k_dict[i] for i in u_m_intersect_u_k)

    # 理论最终 y = y' * sum_k_uk / sum_k_intersect
    expected_y = (expected_y_prime * sum_k_uk) // sum_k_intersect

    print(f"[+] 理论期望的最终结果 : {expected_y}")

    if final_y == expected_y:
        print("\n[✔] 测试通过！解密结果与理论值相符！")
    else:
        print("\n[✘] 测试失败！解密结果与理论值不符！")

