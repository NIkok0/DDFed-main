# DDFed/tests/ex_tmcfe_time.py

import sys
import os
import gmpy2
import time

# 动态添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 引入基线方案 TMCFE 和配置
from ddfed_crypto.baselines.tmcfe import TMCFE
from ddfed_crypto import config

# 可修改的测试运行次数
NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS_OVERRIDE", "1000"))

if __name__ == "__main__":
    print(f"\n====== 开始执行 对比方案 TMCFE 性能测试 (共 {NUM_ITERATIONS} 次) ======")
    print("注: 已移除随机掉线，时间单位已转换为 毫秒 (ms)。阶段命名已完全对齐 TAPFed 论文。")

    # 初始化各阶段的累计耗时变量 (单位: 秒)
    total_time_setup = 0.0
    total_time_sk_distribute = 0.0
    total_time_dk_generate = 0.0  # TMCFE 特有：每轮生成解密密钥
    total_time_encrypt = 0.0
    total_time_share_decrypt = 0.0
    total_time_combine_decrypt = 0.0

    tmcfe = TMCFE()
    num_users = config.N_ENCRYPTORS
    num_decryptors = config.N_DECRYPTORS
    all_encryptors = list(range(1, num_users + 1))
    all_decryptors = list(range(1, num_decryptors + 1))

    for it in range(1, NUM_ITERATIONS + 1):
        print(f"[*] 正在执行第 {it}/{NUM_ITERATIONS} 轮测试...")

        # ----------------------------------------------------------------
        # 1. Setup (系统初始化，密码基础设施全局执行 1 次)
        # ----------------------------------------------------------------
        start_time = time.time()
        pp = tmcfe.setup()
        total_time_setup += (time.time() - start_time)

        # ----------------------------------------------------------------
        # 2. SKDistribute (密码基础设施参与)
        # ----------------------------------------------------------------
        sk_dict = {}
        start_time = time.time()
        for i in all_encryptors:
            sk_dict[i] = tmcfe.sk_distribute(i)
        total_time_sk_distribute += (time.time() - start_time)

        # --- 准备测试数据 ---
        label_l = f"Epoch_{it}"
        x_dict = {i: gmpy2.mpz(i * 10) for i in all_encryptors}
        y_dict = {i: gmpy2.mpz(1) for i in all_encryptors}

        # ----------------------------------------------------------------
        # 3. DKGenerate (TMCFE特有：每轮根据标签和权重动态生成)
        # ----------------------------------------------------------------
        start_time = time.time()
        all_dk_dict = tmcfe.dk_generate(y_dict, label_l)
        total_time_dk_generate += (time.time() - start_time)

        # ----------------------------------------------------------------
        # 4. Encrypt (全体加密节点参与)
        # ----------------------------------------------------------------
        all_cts = {}
        start_time = time.time()
        for i in all_encryptors:
            all_cts[i] = tmcfe.encrypt(sk_dict[i], x_dict[i], label_l)
        total_time_encrypt += ((time.time() - start_time) / num_users)

        # ----------------------------------------------------------------
        # 5. ShareDecrypt (全体解密节点参与)
        # ----------------------------------------------------------------
        all_y_j = {}
        active_d_nodes = all_decryptors
        start_time = time.time()
        for j in active_d_nodes:
            all_y_j[j] = tmcfe.share_decrypt(all_dk_dict[j], j, active_d_nodes, y_dict, all_cts)
        total_time_share_decrypt += ((time.time() - start_time) / num_decryptors)

        # ----------------------------------------------------------------
        # 6. CombineDecrypt (包含离散对数求解)
        # ----------------------------------------------------------------
        start_time = time.time()
        final_y = tmcfe.combine_decrypt(all_y_j)
        total_time_combine_decrypt += (time.time() - start_time)

        # --- 验证正确性 ---
        expected_y = sum(y_dict[i] * x_dict[i] for i in all_encryptors)
        if final_y != expected_y:
            print(f"  [✘] 严重警告: 第 {it} 次测试结果解密失败，理论值({expected_y})与实际值({final_y})不符！")

    # ====================================================================
    # === 统计平均值并将秒转换为毫秒 (ms) ===
    # ====================================================================
    def to_ms(seconds):
        return (seconds / NUM_ITERATIONS) * 1000

    avg_setup_ms = to_ms(total_time_setup)
    avg_sk_distribute_ms = to_ms(total_time_sk_distribute)
    avg_dk_generate_ms = to_ms(total_time_dk_generate)
    avg_encrypt_ms = to_ms(total_time_encrypt)
    avg_share_decrypt_ms = to_ms(total_time_share_decrypt)
    avg_combine_decrypt_ms = to_ms(total_time_combine_decrypt)

    total_pipeline_ms = (avg_setup_ms + avg_sk_distribute_ms + avg_dk_generate_ms +
                         avg_encrypt_ms + avg_share_decrypt_ms + avg_combine_decrypt_ms)

    print("\n==================== TMCFE 对比方案测试结果 (单位: ms) ====================")
    print(f"测试轮数    : {NUM_ITERATIONS} 轮 (无随机掉线)")
    print(f"加密节点数  : {config.N_ENCRYPTORS} | 解密节点数: {config.N_DECRYPTORS} | 门限: {config.T_THRESHOLD}")
    print("---------------------------------------------------------------")
    print("平均每个参与者的阶段耗时 (单位: ms):")
    print(f"  1. Setup            (密码基础设施执行)       : {avg_setup_ms:.3f} ms")
    print(f"  2. SKDistribute     (密码基础设施执行)       : {avg_sk_distribute_ms:.3f} ms")
    print(f"  3. DKGenerate       (密码基础设施执行)       : {avg_dk_generate_ms:.3f} ms")
    print(f"  4. Encrypt          (各个加密节点执行)       : {avg_encrypt_ms:.3f} ms")
    print(f"  5. ShareDecrypt     (各个解密节点执行)       : {avg_share_decrypt_ms:.3f} ms")
    print(f"  6. CombineDecrypt   (各个加密节点执行)       : {avg_combine_decrypt_ms:.3f} ms")
    print("---------------------------------------------------------------")
    print(f"  单次全流程流水线耗时总计                   : {total_pipeline_ms:.3f} ms")
    print("===============================================================")
