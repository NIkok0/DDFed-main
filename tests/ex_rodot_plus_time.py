# DDFed/tests/ex_rodot_plus_time.py

import sys
import os
import gmpy2
import time
import random  # 重新引入 random 以处理随机掉线

# 1. 动态添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. 导入 ddfed_crypto 包
from ddfed_crypto.rodot_plus import RodotPlus
from ddfed_crypto import config

# 可修改的测试运行次数
NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS_OVERRIDE", "1000"))

if __name__ == "__main__":
    print(f"\n====== 开始执行 Rodot+ 性能测试 (共 {NUM_ITERATIONS} 次) ======")

    # 初始化各阶段单节点的累计耗时变量
    total_time_setup = 0.0
    total_time_kgen = 0.0
    total_time_dkshare = 0.0
    total_time_enc = 0.0
    total_time_dkcom = 0.0
    total_time_pardec = 0.0
    total_time_comdec = 0.0

    # 实例化方案与准备基础变量
    rodot = RodotPlus()
    num_users = config.N_ENCRYPTORS
    all_encryptors = list(range(1, num_users + 1))
    all_decryptors = list(range(1, config.N_DECRYPTORS + 1))

    # 开始多次迭代测试
    for it in range(1, NUM_ITERATIONS + 1):
        print(f"[*] 正在执行第 {it}/{NUM_ITERATIONS} 轮测试，请稍候...")

        # ----------------------------------------------------------------
        # 1. 执行 Setup (系统初始化，无节点差异)
        # ----------------------------------------------------------------
        start_time = time.time()
        pp = rodot.setup()
        setup_time = time.time() - start_time
        total_time_setup += setup_time

        # ----------------------------------------------------------------
        # 2. 执行 KGen (全体加密节点参与)
        # ----------------------------------------------------------------
        sk_dict = {}
        start_time = time.time()
        for i in all_encryptors:
            sk_dict[i] = rodot.kgen(i)
        kgen_total_time = time.time() - start_time

        # 计算平均每个加密节点的耗时
        total_time_kgen += (kgen_total_time / num_users)

        # ----------------------------------------------------------------
        # 3. 执行 DKShare (模拟 K 阶段掉线)
        # ----------------------------------------------------------------
        k_dict = {i: gmpy2.mpz(1) for i in all_encryptors}
        all_dk_shares = {}

        # 随机掉线逻辑
        dropped_k_nodes = set(random.sample(all_encryptors, config.N_DROPPED_K))
        active_k_nodes = [i for i in all_encryptors if i not in dropped_k_nodes]

        start_time = time.time()
        for i in active_k_nodes:
            all_dk_shares[i] = rodot.dkshare(sk_dict[i], k_dict[i])
        dkshare_total_time = time.time() - start_time

        # 计算平均每个存活加密节点的耗时
        if active_k_nodes:
            total_time_dkshare += (dkshare_total_time / len(active_k_nodes))

        # ----------------------------------------------------------------
        # 4. 执行 Enc 阶段 (模拟 M 阶段掉线)
        # ----------------------------------------------------------------
        label_l = f"Epoch_{it}"
        x_dict = {i: gmpy2.mpz(i * 10) for i in all_encryptors}
        all_cts = {}

        # 随机掉线逻辑
        dropped_m_nodes = set(random.sample(all_encryptors, config.N_DROPPED_M))
        active_m_nodes = [i for i in all_encryptors if i not in dropped_m_nodes]

        start_time = time.time()
        for i in active_m_nodes:
            all_cts[i] = rodot.enc(sk_dict[i], x_dict[i], label_l)
        enc_total_time = time.time() - start_time

        # 计算平均每个存活加密节点的耗时
        if active_m_nodes:
            total_time_enc += (enc_total_time / len(active_m_nodes))

        # ----------------------------------------------------------------
        # 5. 执行 DKCom 阶段 (全体解密节点处理存活加密节点的份额)
        # ----------------------------------------------------------------
        # 维度转换：只转换活跃的加密节点
        shares_per_decryptor = {j: {} for j in all_decryptors}
        for i in active_k_nodes:
            for j in all_decryptors:
                shares_per_decryptor[j][i] = all_dk_shares[i][j]

        active_k_dict = {i: k_dict[i] for i in active_k_nodes}
        all_dk_j = {}

        start_time = time.time()
        for j in all_decryptors:
            all_dk_j[j] = rodot.dkcom(shares_per_decryptor[j], active_k_dict)
        dkcom_total_time = time.time() - start_time

        # 计算平均每个解密节点的耗时
        total_time_dkcom += (dkcom_total_time / config.N_DECRYPTORS)

        # ----------------------------------------------------------------
        # 6. 执行 ParDec 阶段 (模拟 D 阶段解密节点掉线)
        # ----------------------------------------------------------------
        # 随机掉线逻辑
        dropped_d_nodes = set(random.sample(all_decryptors, config.N_DROPPED_D))
        active_d_nodes = [j for j in all_decryptors if j not in dropped_d_nodes]

        if len(active_d_nodes) < config.T_THRESHOLD:
            print(f"  [!] 警告: 存活解密节点数 ({len(active_d_nodes)}) 小于阈值 t ({config.T_THRESHOLD})，ComDec 将失败！跳过后续测试。")
            continue

        all_y_j = {}
        start_time = time.time()
        for j in active_d_nodes:
            all_y_j[j] = rodot.pardec(all_dk_j[j], active_k_dict, all_cts, label_l)
        pardec_total_time = time.time() - start_time

        # 计算平均每个存活解密节点的耗时
        if active_d_nodes:
            total_time_pardec += (pardec_total_time / len(active_d_nodes))

        # ----------------------------------------------------------------
        # 7. 执行 ComDec 阶段 (基于存活节点的交集)
        # ----------------------------------------------------------------
        # 交集 U_M \cap U_K
        u_m_intersect_u_k = set(active_m_nodes).intersection(set(active_k_nodes))

        start_time = time.time()
        final_y = rodot.comdec(all_y_j, active_k_dict, u_m_intersect_u_k)
        comdec_time = time.time() - start_time
        total_time_comdec += comdec_time

        # --- 验证正确性 ---
        if u_m_intersect_u_k:
            expected_y_prime = sum(k_dict[i] * x_dict[i] for i in u_m_intersect_u_k)
            sum_k_uk = sum(k_dict[i] for i in active_k_nodes)
            sum_k_intersect = sum(k_dict[i] for i in u_m_intersect_u_k)
            expected_y = (expected_y_prime * sum_k_uk) // sum_k_intersect
        else:
            expected_y = 0

        if final_y != expected_y:
            print(f"  [✘] 严重警告: 第 {it} 次测试结果解密失败，理论值({expected_y})与实际值({final_y})不符！")

    # ====================================================================
    # === 统计平均值并格式化输出结果 ===
    # ====================================================================

    def to_ms(seconds):
        return (seconds / NUM_ITERATIONS) * 1000

    # 计算均值 (累计的单节点平均值再除以总迭代次数)
    avg_setup_ms = to_ms(total_time_setup)
    avg_kgen_ms = to_ms(total_time_kgen)
    avg_dkshare_ms = to_ms(total_time_dkshare)
    avg_enc_ms = to_ms(total_time_enc)
    avg_dkcom_ms = to_ms(total_time_dkcom)
    avg_pardec_ms = to_ms(total_time_pardec)
    avg_comdec_ms = to_ms(total_time_comdec)

    total_pipeline_ms = (avg_setup_ms + avg_kgen_ms + avg_dkshare_ms +
                         avg_enc_ms + avg_dkcom_ms + avg_pardec_ms + avg_comdec_ms)

    print("\n==================== 性能测试结果 (单位: ms) ====================")
    print(f"测试轮数    : {NUM_ITERATIONS} 轮")
    print(f"网络环境    : 加密节点数 {config.N_ENCRYPTORS} | 解密节点数 {config.N_DECRYPTORS} | 门限 {config.T_THRESHOLD}")
    print(f"掉线配置    : DKShare阶段掉线 {config.N_DROPPED_K} 个 | Enc阶段掉线 {config.N_DROPPED_M} 个 | ParDec阶段掉线 {config.N_DROPPED_D} 个")
    print("---------------------------------------------------------------")
    print("平均每个【存活】参与者的阶段耗时 (单位: ms):")
    print(f"  1. Setup    (任何节点单次执行)       : {avg_setup_ms:.3f} ms")
    print(f"  2. KGen     (各个加密节点执行)       : {avg_kgen_ms:.3f} ms")
    print(f"  3. DKShare  (存活加密节点执行)       : {avg_dkshare_ms:.3f} ms")
    print(f"  4. Enc      (存活加密节点执行)       : {avg_enc_ms:.3f} ms")
    print(f"  5. DKCom    (各个解密节点执行)       : {avg_dkcom_ms:.3f} ms")
    print(f"  6. ParDec   (存活解密节点执行)       : {avg_pardec_ms:.3f} ms")
    print(f"  7. ComDec   (存活加密节点执行)       : {avg_comdec_ms:.3f} ms")
    print("---------------------------------------------------------------")
    print(f"  单次全流程流水线耗时总计         : {total_pipeline_ms:.3f} ms")
    print("===============================================================")
