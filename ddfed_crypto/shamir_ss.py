# DDFed/ddfed_crypto/shamir_ss.py

import gmpy2
import os


class ShamirSS:

    # =====================================================================
    # 1. 整数上的 Shamir 秘密共享 (Secret Sharing over the Integers)
    # =====================================================================
    @staticmethod
    def share_int(secret, n, t, I, lam, delta):
        """
        ShamirSS.Share(n, t, secret):
        整数上的 Shamir 秘密共享 (SSoI)。
        没有任何取模操作，通过注入极大的均匀随机噪声来保证统计安全。
        """
        secret = gmpy2.mpz(secret)
        I = gmpy2.mpz(I)
        delta = gmpy2.mpz(delta)

        # 1. 使用系统随机熵初始化随机数生成器
        seed = int.from_bytes(os.urandom(16), 'big')
        rs = gmpy2.random_state(seed)

        # 2. 构造多项式 f(x)，常数项 a_0 = \Delta * secret
        coeffs = [gmpy2.mul(delta, secret)]

        # 3. 计算多项式系数采样的绝对值边界: bound = 2^\lambda * \Delta^2 * I
        # 这是一个足够大的边界，能保证统计距离 < 2^{-\lambda}
        lam_pow = gmpy2.mpz(2) ** lam
        delta_sq = gmpy2.mul(delta, delta)
        bound = gmpy2.mul(gmpy2.mul(lam_pow, delta_sq), I)

        for _ in range(t - 1):
            # 在 [-bound, bound] 范围内均匀随机采样
            # 先生成 [0, 2*bound] 的随机数，再减去 bound
            rand_val = gmpy2.mpz_random(rs, gmpy2.add(gmpy2.mul(2, bound), 1))
            a_k = gmpy2.sub(rand_val, bound)
            coeffs.append(a_k)

        # 4. 计算纯整数 shares: f(1), f(2), ..., f(n)
        shares = {}
        for j in range(1, n + 1):
            x = gmpy2.mpz(j)
            y = gmpy2.mpz(0)
            x_pow = gmpy2.mpz(1)  # x^0

            # 标准大整数加法与乘法
            for coeff in coeffs:
                term = gmpy2.mul(coeff, x_pow)
                y = gmpy2.add(y, term)
                x_pow = gmpy2.mul(x_pow, x)

            shares[j] = y

        return shares

    # =====================================================================
    # 2. 有限域上的 Shamir 秘密共享 (Secret Sharing over Finite Field)
    # =====================================================================
    @staticmethod
    def share_field(secret, n, t, prime):
        """
        ShamirSS.share_field(secret, n, t, prime):
        标准的有限域 Z_{prime} 上的 Shamir 秘密共享。
        所有加法和乘法运算都严格在 mod prime 下进行。
        """
        secret = gmpy2.mpz(secret)
        prime = gmpy2.mpz(prime)

        # 1. 使用系统真随机熵初始化随机数生成器
        seed = int.from_bytes(os.urandom(16), 'big')
        rs = gmpy2.random_state(seed)

        # 2. 构造多项式 f(x)，常数项 a_0 = secret mod prime
        # (标准的 Shamir 不需要乘 \Delta)
        coeffs = [gmpy2.f_mod(secret, prime)]

        # 3. 随机生成其余系数 a_1 到 a_{t-1}，范围是 [0, prime - 1]
        for _ in range(t - 1):
            a_k = gmpy2.mpz_random(rs, prime)
            coeffs.append(a_k)

        # 4. 计算 shares: f(1), f(2), ..., f(n) mod prime
        shares = {}
        for j in range(1, n + 1):
            x = gmpy2.mpz(j)
            y = gmpy2.mpz(0)
            x_pow = gmpy2.mpz(1)  # x^0

            for coeff in coeffs:
                # term = (coeff * x_pow) mod prime
                term = gmpy2.f_mod(gmpy2.mul(coeff, x_pow), prime)
                # y = (y + term) mod prime
                y = gmpy2.f_mod(gmpy2.add(y, term), prime)
                # x_pow = (x_pow * x) mod prime
                x_pow = gmpy2.f_mod(gmpy2.mul(x_pow, x), prime)

            shares[j] = y

        return shares