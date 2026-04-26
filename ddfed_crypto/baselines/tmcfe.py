# DDFed/ddfed_crypto/baselines/tmcfe.py

import gmpy2
import os
import hashlib
import math

from .. import config
from ..math_utils import generate_safe_prime
from ..shamir_ss import ShamirSS


def hash_to_zp(l_bytes, p):
    """将标签 l 哈希映射到 Z_p 域"""
    h = hashlib.sha256(l_bytes).hexdigest()
    return gmpy2.f_mod(gmpy2.mpz(int(h, 16)), p)


class TMCFE:
    def __init__(self):
        self.pp = None
        self.msk = None

    def setup(self, lam=config.LAMBDA_SEC, n_encryptors=config.N_ENCRYPTORS, n_decryptors=config.N_DECRYPTORS,
              t=config.T_THRESHOLD):
        """
        Setup(\lambda): 生成公共参数和主密钥
        对应论文: Section III.D Setup
        """
        seed = int.from_bytes(os.urandom(16), 'big')
        self.rs = gmpy2.random_state(seed)

        P, p = generate_safe_prime(lam)

        while True:
            h = gmpy2.mpz_random(self.rs, P - 2) + 2
            g = gmpy2.powmod(h, 2, P)
            if g != 1:
                break

        alpha = gmpy2.mpz_random(self.rs, p)
        g_alpha = gmpy2.powmod(g, alpha, P)

        # === 修正点: 按照论文，W 和 U 必须在 Setup 阶段预先生成并存入 MSK  ===
        W_matrix = {i: gmpy2.mpz_random(self.rs, p) for i in range(1, n_encryptors + 1)}
        U_matrix = {i: gmpy2.mpz_random(self.rs, p) for i in range(1, n_encryptors + 1)}

        self.pp = {
            'P': P, 'p': p, 'g': g,
            'n': n_encryptors, 's': n_decryptors, 't': t
        }
        self.msk = {
            'alpha': alpha,
            'g_alpha': g_alpha,
            'W': W_matrix,
            'U': U_matrix
        }
        return self.pp

    def sk_distribute(self, i):
        """
        SKDistribute: 基础设施将 MSK 中对应的行分发给加密节点 i
        对应论文: Section III.D SKDistribute
        """
        P = self.pp['P']

        # === 修正点: 从 MSK 中直接提取预生成的 W_i 和 U_i  ===
        W_i = self.msk['W'][i]
        U_i = self.msk['U'][i]

        # 计算 g^{\alpha * W_i} mod P
        g_alpha_Wi = gmpy2.powmod(self.msk['g_alpha'], W_i, P)

        sk_i = {
            'i': i,
            'g_alpha': self.msk['g_alpha'],
            'g_alpha_Wi': g_alpha_Wi,
            'U_i': U_i
        }
        return sk_i

    def dk_generate(self, y_dict, label_l):
        """
        DKGenerate: 基础设施根据 MSK 为聚合器生成功能解密密钥份额
        对应论文: Section III.D DKGenerate
        """
        p = self.pp['p']
        H_l = hash_to_zp(str(label_l).encode('utf-8'), p)

        # 1. 计算常数项 f^{(0)}(0) = H(l) * \sum y_i U_i mod p
        sum_yU = gmpy2.mpz(0)
        for i, y_i in y_dict.items():
            U_i = self.msk['U'][i]  # 必须使用 MSK 中的 U
            sum_yU = gmpy2.f_mod(gmpy2.add(sum_yU, gmpy2.mul(gmpy2.mpz(y_i), U_i)), p)
        f0_0 = gmpy2.f_mod(gmpy2.mul(H_l, sum_yU), p)

        # 2. 计算常数项 f^{(i)}(0) = y_i W_i mod p
        fi_0 = {}
        for i, y_i in y_dict.items():
            W_i = self.msk['W'][i]  # 必须使用 MSK 中的 W
            fi_0[i] = gmpy2.f_mod(gmpy2.mul(gmpy2.mpz(y_i), W_i), p)

        shares_f0 = ShamirSS.share_field(f0_0, self.pp['s'], self.pp['t'], p)
        shares_fi = {}
        for i in y_dict.keys():
            shares_fi[i] = ShamirSS.share_field(fi_0[i], self.pp['s'], self.pp['t'], p)

        dk_dict = {}
        for j in range(1, self.pp['s'] + 1):
            v_j1 = {i: shares_fi[i][j] for i in y_dict.keys()}
            dk_dict[j] = {'v_j0': shares_f0[j], 'v_j1': v_j1}
        return dk_dict

    # Encrypt, ShareDecrypt, CombineDecrypt 逻辑保持不变，已与论文对齐
    def encrypt(self, sk_i, x_i, label_l):
        P = self.pp['P'];
        p = self.pp['p'];
        g = self.pp['g']
        H_l = hash_to_zp(str(label_l).encode('utf-8'), p)
        r_i = gmpy2.mpz_random(self.rs, p)
        exp1 = gmpy2.f_mod(gmpy2.add(gmpy2.mpz(x_i), gmpy2.mul(H_l, sk_i['U_i'])), p)
        ct_i0 = gmpy2.f_mod(gmpy2.mul(gmpy2.powmod(g, exp1, P), gmpy2.powmod(sk_i['g_alpha_Wi'], r_i, P)), P)
        ct_i1 = gmpy2.powmod(sk_i['g_alpha'], r_i, P)
        return {'ct_0': ct_i0, 'ct_1': ct_i1}

    def share_decrypt(self, dk_j, j, S_active, y_dict, ct_dict):
        P = self.pp['P'];
        p = self.pp['p'];
        g = self.pp['g']
        num = gmpy2.mpz(1);
        den = gmpy2.mpz(1)
        for m in S_active:
            if m != j:
                num = gmpy2.f_mod(gmpy2.mul(num, gmpy2.mpz(-m)), p)
                den = gmpy2.f_mod(gmpy2.mul(den, gmpy2.mpz(j - m)), p)
        L_j = gmpy2.f_mod(gmpy2.mul(num, gmpy2.invert(den, p)), p)
        ct_j0_prime = gmpy2.mpz(1)
        for i, ct in ct_dict.items():
            ct_j0_prime = gmpy2.f_mod(gmpy2.mul(ct_j0_prime, gmpy2.powmod(ct['ct_0'], gmpy2.mpz(y_dict[i]), P)), P)
        ct_j1_prime = gmpy2.mpz(1)
        for i, ct in ct_dict.items():
            exp = gmpy2.f_mod(gmpy2.mul(dk_j['v_j1'][i], L_j), p)
            ct_j1_prime = gmpy2.f_mod(gmpy2.mul(ct_j1_prime, gmpy2.powmod(ct['ct_1'], exp, P)), P)
        exp_j2 = gmpy2.f_mod(gmpy2.mul(dk_j['v_j0'], L_j), p)
        ct_j2_prime = gmpy2.powmod(g, exp_j2, P)
        return {'ct_0': ct_j0_prime, 'ct_1': ct_j1_prime, 'ct_2': ct_j2_prime}

    def combine_decrypt(self, pardec_dict):
        P = self.pp['P']
        first_j = list(pardec_dict.keys())[0]
        C = pardec_dict[first_j]['ct_0']
        D1 = gmpy2.mpz(1);
        D2 = gmpy2.mpz(1)
        for res in pardec_dict.values():
            D1 = gmpy2.f_mod(gmpy2.mul(D1, res['ct_1']), P)
            D2 = gmpy2.f_mod(gmpy2.mul(D2, res['ct_2']), P)
        D = gmpy2.f_mod(gmpy2.mul(C, gmpy2.invert(gmpy2.f_mod(gmpy2.mul(D1, D2), P), P)), P)
        return self._solve_dlog_bsgs(D)

    def _solve_dlog_bsgs(self, target, max_val=5000000):
        P = self.pp['P'];
        g = self.pp['g'];
        m = int(math.isqrt(max_val)) + 1
        table = {};
        cur = gmpy2.mpz(1)
        for j in range(m):
            table[cur] = j
            cur = gmpy2.f_mod(gmpy2.mul(cur, g), P)
        g_inv_m = gmpy2.invert(gmpy2.powmod(g, m, P), P);
        gamma = target
        for i in range(m):
            if gamma in table: return i * m + table[gamma]
            gamma = gmpy2.f_mod(gmpy2.mul(gamma, g_inv_m), P)
        raise ValueError("离散对数求解失败")
    