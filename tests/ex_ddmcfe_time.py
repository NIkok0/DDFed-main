# DDFed/tests/ex_ddmcfe_time.py

import sys
import os
import gmpy2
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from ddfed_crypto.baselines.ddmcfe import DDMCFE
from ddfed_crypto import config

NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS_OVERRIDE", "1000"))

if __name__ == "__main__":
    print(f"\n====== 开始执行 DDMCFE 性能测试 (共 {NUM_ITERATIONS} 次) ======")

    total_time_setup = 0.0
    total_time_keygen = 0.0
    total_time_encrypt = 0.0
    total_time_dkgenshare = 0.0
    total_time_dkcomb = 0.0
    total_time_decrypt = 0.0

    ddmcfe = DDMCFE()
    num_users = config.N_ENCRYPTORS
    all_users = list(range(1, num_users + 1))

    for it in range(1, NUM_ITERATIONS + 1):
        print(f"[*] 正在执行第 {it}/{NUM_ITERATIONS} 轮测试...")

        label_l = f"Round_{it}"
        x_dict = {i: gmpy2.mpz(i * 10) for i in all_users}
        y_dict = {i: gmpy2.mpz(1) for i in all_users}

        start_time = time.time()
        ddmcfe.Setup(lam=config.LAMBDA_SEC, n_encryptors=num_users)
        total_time_setup += (time.time() - start_time)

        pk_dict, sk_dict, ek_dict = {}, {}, {}
        start_time = time.time()
        for i in all_users:
            pk_dict[i], sk_dict[i], ek_dict[i] = ddmcfe.KeyGen()
        total_time_keygen += ((time.time() - start_time) / num_users)

        c_dict = {}
        start_time = time.time()
        for i in all_users:
            c_dict[i] = ddmcfe.Encrypt(ek_dict[i], x_dict[i], label_l)
        total_time_encrypt += ((time.time() - start_time) / num_users)

        dk_share_dict = {}
        aone_pks = {i: pk_dict[i]['aone_pk'] for i in all_users}
        start_time = time.time()
        for i in all_users:
            dk_share_dict[i] = ddmcfe.DKGenShare(i, sk_dict[i], y_dict[i], all_users, aone_pks, y_dict, label_l)
        total_time_dkgenshare += ((time.time() - start_time) / num_users)

        start_time = time.time()
        d_2 = ddmcfe.DKComb(all_users, dk_share_dict, y_dict, label_l)
        total_time_dkcomb += (time.time() - start_time)

        start_time = time.time()
        res = ddmcfe.Decrypt(all_users, c_dict, d_2, y_dict, label_l)
        total_time_decrypt += (time.time() - start_time)

        expected_y = sum(y_dict[i] * x_dict[i] for i in all_users)
        if res != expected_y:
            print(f"  [✘] 严重警告: 测试失败！预期 ({expected_y}) vs 实际 ({res})")

    # ====================================================================
    # === 统计平均值并格式化输出结果 ===
    # ====================================================================
    def to_ms(seconds):
        return (seconds / NUM_ITERATIONS) * 1000

    avg_setup_ms = to_ms(total_time_setup)
    avg_keygen_ms = to_ms(total_time_keygen)
    avg_encrypt_ms = to_ms(total_time_encrypt)
    avg_dkgenshare_ms = to_ms(total_time_dkgenshare)
    avg_dkcomb_ms = to_ms(total_time_dkcomb)
    avg_decrypt_ms = to_ms(total_time_decrypt)

    total_pipeline_ms = (avg_setup_ms + avg_keygen_ms + avg_encrypt_ms +
                         avg_dkgenshare_ms + avg_dkcomb_ms + avg_decrypt_ms)

    print("\n==================== DDMCFE 性能测试结果 ====================")
    print(f"测试轮数    : {NUM_ITERATIONS} 轮 (无随机掉线)")
    print(f"加密节点数  : {config.N_ENCRYPTORS} | (注: 本方案为单聚合服务器架构)")
    print("---------------------------------------------------------------------")
    print("平均每个参与者的阶段耗时 (单位: ms):")
    print(f"  1. Setup       (中心服务器单次执行)   : {avg_setup_ms:.3f} ms")
    print(f"  2. KeyGen      (各个加密节点并行)     : {avg_keygen_ms:.3f} ms")
    print(f"  3. Encrypt     (各个加密节点并行)     : {avg_encrypt_ms:.3f} ms")
    print(f"  4. DKGenShare  (各个加密节点并行)     : {avg_dkgenshare_ms:.3f} ms (含 AoNE 封装)")
    print(f"  5. DKComb      (中心服务器单次执行)   : {avg_dkcomb_ms:.3f} ms (含 AoNE 解封装)")
    print(f"  6. Decrypt     (中心服务器单次执行)   : {avg_decrypt_ms:.3f} ms (含 Pairing 聚合与 DLOG 求解)")
    print("---------------------------------------------------------------------")
    print(f"  单次全流程流水线耗时总计              : {total_pipeline_ms:.3f} ms")
    print("=====================================================================")
