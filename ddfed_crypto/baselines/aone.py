# DDFed/ddfed_crypto/baselines/aone.py

import hashlib
try:
    from charm.toolbox.pairinggroup import ZR, G1, G2, pair
except ImportError:
    from .pairing_mock import ZR, G1, G2, pair, PairingGroup


class AoNE_Charm:
    """
    All-or-Nothing Encapsulation (AoNE) 极速 C++ 库实现
    底层引擎: Stanford PBC (通过 Charm-Crypto)
    """

    def __init__(self, group):
        self.group = group

    def derive_symmetric_key(self, GT_element, length_bytes=256):
        """将 C++ 层面的 GT 元素序列化，并哈希派生出 2048-bit 密钥流"""
        gt_bytes = self.group.serialize(GT_element)
        mask_bytes = hashlib.shake_256(gt_bytes).digest(length_bytes)
        return int.from_bytes(mask_bytes, 'big')

    def keygen(self):
        """生成 sk_t \in ZR 和 pk_T \in G2"""
        sk_t = self.group.random(ZR)
        g2_gen = self.group.hash('g2_gen', G2)
        pk_T = g2_gen ** sk_t
        return sk_t, pk_T

    def encapsulate(self, i, sk_t, x_i, active_users, pk_dict, label):
        group_label_str = str(sorted(active_users)) + "_" + str(label)
        H_G1 = self.group.hash(group_label_str, G1)

        # 客户端份额 S_pk \in G1
        S_pk = H_G1 ** sk_t

        # 随机数 r_pk 和 R_pk \in G2
        r_pk = self.group.random(ZR)
        g2_gen = self.group.hash('g2_gen', G2)
        R_pk = g2_gen ** r_pk

        # 聚合所有的 pk_j (G2 上的加法在 Charm 中用 * 表示)
        sum_pk = pk_dict[active_users[0]]
        for j in active_users[1:]:
            sum_pk = sum_pk * pk_dict[j]

        r_sum_pk = sum_pk ** r_pk

        # C++ 极速双线性配对
        K_sym_GT = pair(H_G1, r_sum_pk)

        # 异或加密
        otp_mask = self.derive_symmetric_key(K_sym_GT, length_bytes=256)
        c_i = int(x_i) ^ otp_mask

        return {'c_i': c_i, 'R_pk': R_pk, 'S_pk': S_pk}

    def decapsulate(self, active_users, enc_payloads, label):
        # 聚合所有人的份额 sum_S
        sum_S = enc_payloads[active_users[0]]['S_pk']
        for i in active_users[1:]:
            sum_S = sum_S * enc_payloads[i]['S_pk']

        recovered_x_dict = {}
        for i in active_users:
            R_pk = enc_payloads[i]['R_pk']
            c_i = enc_payloads[i]['c_i']

            # C++ 极速双线性配对恢复密钥
            K_sym_GT = pair(sum_S, R_pk)
            otp_mask = self.derive_symmetric_key(K_sym_GT, length_bytes=256)
            x_i = c_i ^ otp_mask

            recovered_x_dict[i] = x_i

        return recovered_x_dict
