# DDFed/ddfed_crypto/baselines/ddmcfe.py

import os
import gmpy2
try:
    from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair
except ImportError:
    from .pairing_mock import PairingGroup, ZR, G1, G2, GT, pair
from .aone import AoNE_Charm


class DDMCFE:
    """
    Dynamic Decentralized MCFE (C++ 极速版)
    复现自: Yu et al. [2025] Section III-B
    """

    def __init__(self):
        # 初始化底层 C++ 配对群 (使用 SS512 曲线，最标准的对称配对曲线)
        self.group = PairingGroup('SS512')
        self.aone = AoNE_Charm(self.group)
        self.g1 = self.group.hash('generator_g1', G1)
        self.g2 = self.group.hash('generator_g2', G2)

    def hash_to_G1_vec2(self, data_str):
        h1 = self.group.hash(data_str + "_1", G1)
        h2 = self.group.hash(data_str + "_2", G1)
        return (h1, h2)

    def hash_to_G2_vec2(self, y_dict):
        y_str = str(sorted(y_dict.items()))
        h1 = self.group.hash(y_str + "_1", G2)
        h2 = self.group.hash(y_str + "_2", G2)
        return (h1, h2)

    def dlog_bsgs(self, target_GT, max_range=100000):
        """利用 C++ 极速序列化的 BSGS 离散对数求解"""
        GT_gen = pair(self.g1, self.g2)
        m = int(max_range ** 0.5) + 1

        value_table = {}
        current = self.group.init(GT, 1)
        for j in range(m):
            key = self.group.serialize(current)
            value_table[key] = j
            current = current * GT_gen

        # ======== 修复点：拆解负数幂运算，迎合 C++ 底层接口 ========
        GT_gen_m = GT_gen ** m
        GT_gen_m_inv = GT_gen_m ** -1  # 先算 m 次方，再统一求逆元
        # ==========================================================

        giant_current = target_GT

        for i in range(m):
            key = self.group.serialize(giant_current)
            if key in value_table:
                return i * m + value_table[key]
            giant_current = giant_current * GT_gen_m_inv

        raise ValueError("DLOG range exceeded: 掩码抵消失败，密文变为乱码！")

    def Setup(self, lam=128, n_encryptors=10):
        return {'n': n_encryptors}

    def KeyGen(self):
        s = (self.group.random(ZR), self.group.random(ZR))
        aone_sk, aone_pk = self.aone.keygen()
        sk_pk = {'s': s, 'aone_sk': aone_sk}
        return {'aone_pk': aone_pk}, sk_pk, s

    def Encrypt(self, ek_pk, x_pk, label_l):
        u_l_1 = self.hash_to_G1_vec2(str(label_l))
        s = ek_pk
        u_s_G1 = (u_l_1[0] ** s[0]) * (u_l_1[1] ** s[1])
        c_pk_1 = u_s_G1 * (self.g1 ** int(x_pk))
        return c_pk_1

    def DKGenShare(self, i, sk_pk, y, active_users, aone_pk_dict, y_dict, label_l):
        s = sk_pk['s']
        y_zr = self.group.init(ZR, int(y))
        v_y_2 = self.hash_to_G2_vec2(y_dict)

        T_pk = (
            (self.group.random(ZR), self.group.random(ZR)),
            (self.group.random(ZR), self.group.random(ZR))
        )
        ys = (y_zr * s[0], y_zr * s[1])

        Tv_0 = (v_y_2[0] ** T_pk[0][0]) * (v_y_2[1] ** T_pk[0][1])
        Tv_1 = (v_y_2[0] ** T_pk[1][0]) * (v_y_2[1] ** T_pk[1][1])
        d_pk_1 = ((self.g2 ** ys[0]) * Tv_0, (self.g2 ** ys[1]) * Tv_1)

        # 转换 C++ ZR 标量为 Python 整数进行封装
        t00 = int(T_pk[0][0]);
        t01 = int(T_pk[0][1])
        t10 = int(T_pk[1][0]);
        t11 = int(T_pk[1][1])
        T_int = (t00 << 768) | (t01 << 512) | (t10 << 256) | t11

        d_pk_2 = self.aone.encapsulate(i, sk_pk['aone_sk'], T_int, active_users, aone_pk_dict, label_l)
        return (d_pk_1, d_pk_2)

    def DKComb(self, active_users, dk_share_dict, y_dict, label_l):
        v_y_2 = self.hash_to_G2_vec2(y_dict)

        aone_payloads = {i: dk[1] for i, dk in dk_share_dict.items()}
        T_int_dict = self.aone.decapsulate(active_users, aone_payloads, label_l)

        sum_T = [[0, 0], [0, 0]]
        for T_int in T_int_dict.values():
            sum_T[0][0] += (T_int >> 768)
            sum_T[0][1] += ((T_int >> 512) & ((1 << 256) - 1))
            sum_T[1][0] += ((T_int >> 256) & ((1 << 256) - 1))
            sum_T[1][1] += (T_int & ((1 << 256) - 1))

        order_attr = self.group.order
        p_order = int(order_attr() if callable(order_attr) else order_attr)
        sum_T_ZR = (
            (self.group.init(ZR, sum_T[0][0] % p_order), self.group.init(ZR, sum_T[0][1] % p_order)),
            (self.group.init(ZR, sum_T[1][0] % p_order), self.group.init(ZR, sum_T[1][1] % p_order))
        )

        sum_Tv_0 = (v_y_2[0] ** sum_T_ZR[0][0]) * (v_y_2[1] ** sum_T_ZR[0][1])
        sum_Tv_1 = (v_y_2[0] ** sum_T_ZR[1][0]) * (v_y_2[1] ** sum_T_ZR[1][1])

        sum_d1_0 = self.group.init(G2, 1)
        sum_d1_1 = self.group.init(G2, 1)

        for i in active_users:
            dk_1 = dk_share_dict[i][0]
            sum_d1_0 = sum_d1_0 * dk_1[0]
            sum_d1_1 = sum_d1_1 * dk_1[1]

        # 减去掩码 (乘以逆元)
        d_2 = (sum_d1_0 * (sum_Tv_0 ** -1), sum_d1_1 * (sum_Tv_1 ** -1))
        return d_2

    def Decrypt(self, active_users, c_dict, d_2, y_dict, label_l):
        u_l_1 = self.hash_to_G1_vec2(str(label_l))

        sum_pair_c_y = self.group.init(GT, 1)
        for i in active_users:
            y_i_G2 = self.g2 ** int(y_dict[i])
            sum_pair_c_y = sum_pair_c_y * pair(c_dict[i], y_i_G2)

        sum_pair_u_d = pair(u_l_1[0], d_2[0]) * pair(u_l_1[1], d_2[1])
        target_GT = sum_pair_c_y / sum_pair_u_d

        return self.dlog_bsgs(target_GT)
