# DDFed/tests/ex_dmcfe_ip_time_dropout.py

import sys
import os
import gmpy2
import time
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from ddfed_crypto.baselines.dmcfe_ip import DMCFE_IP
from ddfed_crypto import config

NUM_BENCHMARK_ITERATIONS = int(os.getenv("NUM_BENCHMARK_ITERATIONS", "1000"))

# 定义掉线人数（可以根据 config 引入，或者直接在这里修改测试）
# 确保 存活人数 = config.N_ENCRYPTORS - DROP_COUNT >= config.T_THRESHOLD
DROP_COUNT = int(os.getenv("DROP_COUNT_OVERRIDE", "5"))

if __name__ == "__main__":
    print(f"\n====== 开始执行 DMCFE-IP [掉线容错] 性能基准测试 ======")

    time_global_setup = 0.0
    time_client_setup = 0.0

    time_agree_on_weight_y_client = 0.0
    time_key_sharing = 0.0
    time_encryption = 0.0

    time_agree_on_weight_y_server = 0.0
    time_aggregation = 0.0

    dmcfe = DMCFE_IP()
    num_users = config.N_ENCRYPTORS
    all_encryptors = list(range(1, num_users + 1))
    num_active = num_users - DROP_COUNT

    if num_active < config.T_THRESHOLD:
        print(f"错误: 存活人数 ({num_active}) 小于门限阈值 t ({config.T_THRESHOLD})，无法解密！")
        sys.exit(1)

    for it in range(1, NUM_BENCHMARK_ITERATIONS + 1):
        print(f"\n[*] 正在执行第 {it}/{NUM_BENCHMARK_ITERATIONS} 轮测试，请稍候...")

        label_l = f"Task_ID_{it}_Drop"
        x_dict = {i: gmpy2.mpz(i * 10) for i in all_encryptors}
        y_dict = {i: gmpy2.mpz(1) for i in all_encryptors}

        # 随机挑选本轮的存活者
        active_users = random.sample(all_encryptors, num_active)
        dropped_users = [i for i in all_encryptors if i not in active_users]
        print(f"    [!] 本轮掉线节点: {dropped_users}")

        # =========================================================================
        # [Initialization Phase] (无掉线)
        # =========================================================================
        start_time = time.time()
        pp = dmcfe.GlobalSetup(lam=config.LAMBDA_SEC, n_encryptors=num_users, t=config.T_THRESHOLD)
        time_global_setup += (time.time() - start_time)

        sk_dict = {}
        pk_dict = {}
        start_time = time.time()
        for i in all_encryptors:
            sk_dict[i], pk_dict[i] = dmcfe.ClientSetup()
        time_client_setup += ((time.time() - start_time) / num_users)

        # =========================================================================
        # [Aggregation Phase] (存在掉线)
        # =========================================================================

        # --- Step 1: Agree on weight Y ---
        y_payloads = {}
        start_time = time.time()
        for i in active_users:  # 仅存活者发送签名
            y_payloads[i] = dmcfe.AgreeOnWeightY_Sign(y_dict[i], sk_dict[i])
        time_agree_on_weight_y_client += ((time.time() - start_time) / num_active)

        start_time = time.time()
        for i in active_users:  # 服务器仅验证存活者的签名
            if not dmcfe.AgreeOnWeightY_Verify(y_payloads[i], pk_dict[i]['sig_pk_i']):
                raise ValueError(f"权重验签失败: 节点 {i}")
        time_agree_on_weight_y_server += (time.time() - start_time)

        # --- Step 2: Key Sharing ---
        shares_dh_dict = {}
        start_time = time.time()
        # 注意：这里必须是所有人执行。在真实场景中，KeySharing通常在模型训练开始前就完成了，
        # 以确保当节点在未来掉线时，它的份额已经被安全分发。
        for i in all_encryptors:
            res = dmcfe.KeySharing(sk_dict[i])
            shares_dh_dict[i] = res['shares_dh']
        time_key_sharing += ((time.time() - start_time) / num_users)

        # --- Step 3: Encryption ---
        enc_payloads = {}
        start_time = time.time()
        for i in active_users:  # 掉线者不发送密文
            # 存活者依然会生成包括对掉线者在内的所有 PRG 掩码 (因为他们不知道谁掉线了)
            enc_payloads[i] = dmcfe.Encryption(i, sk_dict[i], x_dict[i], y_dict[i], pk_dict, label_l)
        time_encryption += ((time.time() - start_time) / num_active)

        # --- Step 4: Aggregation (服务器独占，并触发拉格朗日重构) ---
        start_time = time.time()
        final_y = dmcfe.Aggregation(active_users, all_encryptors, enc_payloads, shares_dh_dict, pk_dict, label_l,
                                    y_dict)
        time_aggregation += (time.time() - start_time)

        # 验证仅聚合存活节点的数据
        expected_y = sum(y_dict[i] * x_dict[i] for i in active_users)
        if final_y != expected_y:
            print(f"  [✘] 严重警告: Benchmark 采样失败，掉线恢复后解密不一致！期望: {expected_y}, 实际: {final_y}")
        else:
            print(f"    [√] 服务器成功恢复掩码，解密结果一致 ({final_y})。")


    # =========================================================================
    # 统计与输出
    # =========================================================================
    def to_ms(seconds):
        return (seconds / NUM_BENCHMARK_ITERATIONS) * 1000


    print("\n================= DMCFE-IP 性能测试结果 (含掉线恢复) =================")
    print(f"测试轮数    ： {NUM_BENCHMARK_ITERATIONS} 轮")
    print(f"系统架构    : 1 台服务器 | {config.N_ENCRYPTORS} 节点总数 | {DROP_COUNT} 节点掉线")
    print("----------------------------------------------------------------------")
    print("各阶段单节点平均计算耗时 (单位: ms):")

    print(f"\n  [Initialization Phase]")
    print(f"      Global Setup                 : {to_ms(time_global_setup):.3f} ms (单次执行)")
    print(f"      Client Setup                 : {to_ms(time_client_setup):.3f} ms (并行)")

    print(f"\n  [Aggregation Phase]")
    print(f"    > 存活的加密节点 (Client) 本地执行:")
    print(f"      Step 1: Agree on weight Y    : {to_ms(time_agree_on_weight_y_client):.3f} ms")
    print(f"      Step 2: Key Sharing          : {to_ms(time_key_sharing):.3f} ms")
    print(f"      Step 3: Encryption           : {to_ms(time_encryption):.3f} ms")

    print(f"    > 服务器 (Cloud Server) 独占计算:")
    print(f"      Step 1: Verify weight Y      : {to_ms(time_agree_on_weight_y_server):.3f} ms")
    print(f"      Step 4: Aggregation          : {to_ms(time_aggregation):.3f} ms")
    print(f"        *(注:包含 {DROP_COUNT} 次拉格朗日重构及冗余掩码消除)*")

    client_agg_total = to_ms(time_agree_on_weight_y_client) + to_ms(time_key_sharing) + to_ms(time_encryption)
    server_agg_total = to_ms(time_agree_on_weight_y_server) + to_ms(time_aggregation)

    print("----------------------------------------------------------------------")
    print(f"  客户端 (Client) Aggregation Phase 纯计算负担: {client_agg_total:.3f} ms")
    print(f"  服务器 (Server) Aggregation Phase 纯计算负担: {server_agg_total:.3f} ms")
    print("======================================================================")