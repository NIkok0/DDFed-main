# DDFed/ddfed_crypto/baselines/dmcfe_ip.py

import gmpy2
import os
import hashlib

from .. import config
from ..math_utils import generate_safe_prime
from ..shamir_ss import ShamirSS


# =====================================================================
# 工具库: Hash, PRG, Schnorr 签名, 以及 Shamir 恢复
# =====================================================================
def hash_to_zn2_star(l_str, N, N2):
    h = int(hashlib.sha256(l_str.encode('utf-8')).hexdigest(), 16)
    h_mpz = gmpy2.f_mod(gmpy2.mpz(h), N2)
    while gmpy2.gcd(h_mpz, N) != 1 or h_mpz == 0:
        h_mpz = gmpy2.f_mod(h_mpz + 1, N2)
    return h_mpz


def PRG(seed, bound):
    h = hashlib.sha256(str(seed).encode('utf-8')).hexdigest()
    return gmpy2.f_mod(gmpy2.mpz(int(h, 16)), bound)


def schnorr_keygen(P, p, g, rs):
    sk = gmpy2.mpz_random(rs, p)
    pk = gmpy2.powmod(g, sk, P)
    return sk, pk


def schnorr_sign(message, sk, P, p, g, rs):
    k = gmpy2.mpz_random(rs, p)
    r = gmpy2.powmod(g, k, P)
    h_input = str(r) + "_" + str(message)
    e = gmpy2.f_mod(gmpy2.mpz(int(hashlib.sha256(h_input.encode('utf-8')).hexdigest(), 16)), p)
    s = gmpy2.f_mod(gmpy2.sub(k, gmpy2.mul(sk, e)), p)
    return {'e': e, 's': s}


def schnorr_verify(message, sig, pk, P, p, g):
    r_prime = gmpy2.f_mod(gmpy2.mul(gmpy2.powmod(g, sig['s'], P), gmpy2.powmod(pk, sig['e'], P)), P)
    h_input = str(r_prime) + "_" + str(message)
    e_prime = gmpy2.f_mod(gmpy2.mpz(int(hashlib.sha256(h_input.encode('utf-8')).hexdigest(), 16)), p)
    return sig['e'] == e_prime


def recover_field(shares_dict, P):
    secret = gmpy2.mpz(0)
    for j, y_j in shares_dict.items():
        num = gmpy2.mpz(1);
        den = gmpy2.mpz(1)
        for m in shares_dict.keys():
            if m != j:
                num = gmpy2.f_mod(gmpy2.mul(num, gmpy2.mpz(-m)), P)
                den = gmpy2.f_mod(gmpy2.mul(den, gmpy2.sub(j, m)), P)
        L_j = gmpy2.f_mod(gmpy2.mul(num, gmpy2.invert(den, P)), P)
        secret = gmpy2.f_mod(gmpy2.add(secret, gmpy2.f_mod(gmpy2.mul(y_j, L_j), P)), P)
    return secret


# =====================================================================
# Section III.A: Modified MCFE Primitives
# =====================================================================
def MCFE_Setup(lam, rs):
    p_dcr, _ = generate_safe_prime(lam)
    q_dcr, _ = generate_safe_prime(lam)
    N = gmpy2.mul(p_dcr, q_dcr)
    N2 = gmpy2.mul(N, N)
    return N, N2


def MCFE_KeyGen(N, rs):
    return gmpy2.mpz_random(rs, gmpy2.f_div(N, 4))


def MCFE_Encrypt(N, N2, s_i, x_i, label_l):
    h_l = hash_to_zn2_star(str(label_l), N, N2)
    term1 = gmpy2.powmod(1 + N, gmpy2.mpz(x_i), N2)
    term2 = gmpy2.powmod(h_l, s_i, N2)
    return gmpy2.f_mod(gmpy2.mul(term1, term2), N2)


def MCFE_Decrypt(N, N2, c_list, sum_dk, label_l, y_list):
    h_l = hash_to_zn2_star(str(label_l), N, N2)
    C = gmpy2.mpz(1)
    for i, c_i in enumerate(c_list):
        C = gmpy2.f_mod(gmpy2.mul(C, gmpy2.powmod(c_i, gmpy2.mpz(y_list[i]), N2)), N2)
    denom_inv = gmpy2.invert(gmpy2.powmod(h_l, sum_dk, N2), N2)
    V = gmpy2.f_mod(gmpy2.mul(C, denom_inv), N2)
    return gmpy2.divexact(gmpy2.sub(V, 1), N)


# =====================================================================
# DMCFE-IP 协议调度器
# =====================================================================
class DMCFE_IP:
    def __init__(self):
        self.pp = None
        self.rs = None

    # --- Phase I: Initialization ---
    def GlobalSetup(self, lam=config.LAMBDA_SEC, n_encryptors=config.N_ENCRYPTORS, t=config.T_THRESHOLD):
        seed = int.from_bytes(os.urandom(16), 'big')
        self.rs = gmpy2.random_state(seed)

        N, N2 = MCFE_Setup(lam, self.rs)

        P, p_dh = generate_safe_prime(lam)
        while True:
            h = gmpy2.mpz_random(self.rs, P - 2) + 2
            g = gmpy2.powmod(h, 2, P)
            if g != 1: break

        delta = gmpy2.fac(n_encryptors)

        self.pp = {
            'N': N, 'N2': N2, 'P': P, 'p': p_dh, 'g': g,
            'n': n_encryptors, 't': t, 'lam': lam,
            'I': gmpy2.mpz(2) ** 40, 'delta': delta,
            'mask_bound': gmpy2.mul(N, gmpy2.mpz(2) ** lam)
        }
        return self.pp

    def ClientSetup(self):
        s_i = MCFE_KeyGen(self.pp['N'], self.rs)
        dh_sk_i, dh_pk_i = schnorr_keygen(self.pp['P'], self.pp['p'], self.pp['g'], self.rs)
        sig_sk_i, sig_pk_i = schnorr_keygen(self.pp['P'], self.pp['p'], self.pp['g'], self.rs)

        sk_i = {'s_i': s_i, 'dh_sk_i': dh_sk_i, 'sig_sk_i': sig_sk_i}
        pk_i = {'dh_pk_i': dh_pk_i, 'sig_pk_i': sig_pk_i}
        return sk_i, pk_i

    # --- Phase II: Aggregation ---

    def AgreeOnWeightY_Sign(self, y_i, sk_i):
        """Step 1 (Client): 客户端对自己的权重 y_i 进行签名"""
        sig_y = schnorr_sign(y_i, sk_i['sig_sk_i'], self.pp['P'], self.pp['p'], self.pp['g'], self.rs)
        return {'y_i': gmpy2.mpz(y_i), 'sig_y': sig_y}

    def AgreeOnWeightY_Verify(self, y_payload, pk_sig):
        """Step 1 (Server): 云服务器统一验签所有客户端上传的权重"""
        return schnorr_verify(y_payload['y_i'], y_payload['sig_y'], pk_sig, self.pp['P'], self.pp['p'], self.pp['g'])

    def KeySharing(self, sk_i):
        """Step 2: 分发 Shares 用于容错"""
        shares_s = ShamirSS.share_int(sk_i['s_i'], self.pp['n'], self.pp['t'], self.pp['I'], self.pp['lam'],
                                      self.pp['delta'])
        shares_dh = ShamirSS.share_field(sk_i['dh_sk_i'], self.pp['n'], self.pp['t'], self.pp['P'])
        return {'shares_s': shares_s, 'shares_dh': shares_dh}

    def Encryption(self, i, sk_i, x_i, y_i, pk_dict, label_l):
        """Step 3: 计算 dk_i (含掩码) 和密文 c_i"""
        dk_i = gmpy2.mul(gmpy2.mpz(y_i), sk_i['s_i'])
        for j, pk_j in pk_dict.items():
            if j == i: continue
            K_ij = gmpy2.powmod(pk_j['dh_pk_i'], sk_i['dh_sk_i'], self.pp['P'])
            seed = str(K_ij) + "_" + str(label_l)
            mask_ij = PRG(seed, self.pp['mask_bound'])
            if i < j:
                dk_i = gmpy2.add(dk_i, mask_ij)
            else:
                dk_i = gmpy2.sub(dk_i, mask_ij)

        c_i = MCFE_Encrypt(self.pp['N'], self.pp['N2'], sk_i['s_i'], x_i, label_l)

        return {'c_i': c_i, 'dk_i': dk_i}

    def Aggregation(self, U, all_encryptors, enc_payloads, shares_dh_dict, pk_dict, label_l, y_dict):
        """Step 4: 服务器处理掉线掩码、聚合解密"""
        sum_dk = gmpy2.mpz(0)

        for i in U:
            sum_dk = gmpy2.add(sum_dk, enc_payloads[i]['dk_i'])

        dropped = [k for k in all_encryptors if k not in U]
        for k in dropped:
            shares_for_k = {i: shares_dh_dict[k][i] for i in U[:self.pp['t']]}
            dh_sk_k = recover_field(shares_for_k, self.pp['P'])

            # 服务器模拟掉线者 k，与所有存活者 i 计算出对应的悬空掩码，并主动消除
            for i in U:
                K_ik = gmpy2.powmod(pk_dict[i]['dh_pk_i'], dh_sk_k, self.pp['P'])
                seed = str(K_ik) + "_" + str(label_l)
                mask_ik = PRG(seed, self.pp['mask_bound'])

                if i < k:
                    sum_dk = gmpy2.sub(sum_dk, mask_ik)
                else:
                    sum_dk = gmpy2.add(sum_dk, mask_ik)

        c_list = [enc_payloads[i]['c_i'] for i in U]
        y_list = [y_dict[i] for i in U]

        return MCFE_Decrypt(self.pp['N'], self.pp['N2'], c_list, sum_dk, label_l, y_list)
