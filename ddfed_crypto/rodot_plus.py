# DDFed/DDfed_crypto/rodot_plus.py

import gmpy2
import math
import random
from . import config
from .math_utils import generate_safe_prime, hash_to_zn2_star
from .shamir_ss import ShamirSS


class RodotPlus:
    def __init__(self):
        self.pp = None

    def setup(self, lam=config.LAMBDA_SEC, n=config.N_DECRYPTORS, t=config.T_THRESHOLD):
        """
        Setup(\lambda): 初始化系统参数
        """
        lam_mpz = gmpy2.mpz(lam)

        # 1. 生成安全素数 p, q，使得 p = 2p'+1, q = 2q'+1 且 p', q' > 2^\lambda
        print("    -> 正在生成安全素数 p (这可能需要一些时间)...")
        p, p_prime = generate_safe_prime(lam)
        print("    -> 正在生成安全素数 q (这可能需要一些时间)...")
        q, q_prime = generate_safe_prime(lam)

        # N = pq
        N = gmpy2.mul(p, q)
        N2 = gmpy2.mul(N, N)

        # 2. 计算标准差 \sigma > \sqrt{\lambda} * N^{5/2}
        # 为了避免浮点精度问题，我们将不等式转换为: \sigma > \sqrt{\lambda * N^5}
        # 使用 gmpy2.isqrt_rem 计算 \lfloor\sqrt{\lambda * N^5}\rfloor 然后 + 1 保证严格大于
        print("    -> 正在计算高斯分布标准差 sigma...")
        N5 = gmpy2.mul(gmpy2.mul(N2, N2), N)  # N^5
        lambda_N5 = gmpy2.mul(lam_mpz, N5)
        isqrt_val, _ = gmpy2.isqrt_rem(lambda_N5)
        sigma = gmpy2.add(isqrt_val, 1)

        # 3. 计算边界 I > \sqrt{2(\lambda + 1)\ln 2} * \sigma
        print("    -> 正在计算边界 I...")
        const_factor = math.sqrt(2 * (lam + 1) * math.log(2))
        const_factor_ceil = gmpy2.mpz(math.ceil(const_factor))

        # 计算 I 的下界，并加 1 保证严格大于
        I = gmpy2.add(gmpy2.mul(const_factor_ceil, sigma), 1)

        # 预计算 \Delta = n!
        delta = gmpy2.fac(n)

        # 4. 组装公开参数 pp
        # H(.) 通过 math_utils.hash_to_zn2_star 隐式提供，不需要作为变量存入字典
        self.pp = {
            'N': N,
            'N2': N2,  # 额外保存 N^2 以便后续加密和解密运算加速
            'sigma': sigma,
            'I': I,
            'n': n,
            't': t,
            'delta': delta
        }

        return self.pp

    def kgen(self, i):
        """
        KGen(i): 为加密节点 i 生成私钥 sk_i
        输入: 节点编号 i (虽然本逻辑中 i 目前仅作标识，并未直接参与数学运算)
        """
        if self.pp is None:
            raise ValueError("请先运行 setup() 初始化公开参数 pp")

        sigma = self.pp['sigma']
        I = self.pp['I']

        while True:
            # 1. 应对超大 sigma 的高斯分布采样策略
            # 先从标准正态分布 N(0, 1) 采样出一个浮点数 z
            z = random.gauss(0, 1)

            # 提取高精度浮点数并转换为大整数乘法，避免 float 溢出
            # 2**53 是双精度浮点数 float 的尾数精度
            z_scaled = int(z * (1 << 53))

            # s_i = (z_scaled * sigma) // (2**53)，等价于 s_i = z * sigma 的大数版本
            s_i = gmpy2.f_div(gmpy2.mul(gmpy2.mpz(z_scaled), sigma), 1 << 53)

            # 2. 拒绝采样条件: |s_i| <= (q0 - 1) / 2
            if abs(s_i) <= I:
                sk_i = s_i
                break

        return sk_i

    def dkshare(self, sk_i, k_i):
        """
        DKShare(sk_i, k_i): 为加密节点 i 生成发送给解密节点的密钥份额
        """
        if self.pp is None:
            raise ValueError("请先运行 setup() 初始化公开参数 pp")

        n = self.pp['n']
        t = self.pp['t']
        I = self.pp['I']
        lam = config.LAMBDA_SEC
        delta = self.pp['delta']

        # 1. 调用 ShamirSS 生成中间份额 dk'_{ij}
        # 传入 I 和 lam 供底层生成统计隐藏的随机系数边界
        dk_prime_shares = ShamirSS.share_int(sk_i, n, t, I, lam, delta)

        # 2. 乘以标量 k_i 得到最终份额: dk_{ij} = k_i * dk'_{ij}
        # k_i \in Z，因此直接进行常规的大整数乘法
        k_i_mpz = gmpy2.mpz(k_i)
        dk_shares = {}
        for j, dk_prime_ij in dk_prime_shares.items():
            dk_shares[j] = gmpy2.mul(k_i_mpz, dk_prime_ij)

        return dk_shares

    def enc(self, sk_i, x_i, l):
        """
        Enc(sk_i, x_i, l): 加密节点生成密文
        """
        if self.pp is None:
            raise ValueError("请先运行 setup() 初始化公开参数 pp")

        N = self.pp['N']
        N2 = self.pp['N2']
        delta = self.pp['delta']

        # 确保标签 l 转换为字节流，以便进行哈希
        if isinstance(l, str):
            l_bytes = l.encode('utf-8')
        elif isinstance(l, int):
            l_bytes = str(l).encode('utf-8')
        else:
            l_bytes = bytes(l)

        # 1. 计算 H(l)
        H_l = hash_to_zn2_star(l_bytes, N2)

        # 2. 计算 (1+N)^{x_i} mod N^2
        # 利用二项式展开优化: 1 + x_i * N mod N^2
        term1 = gmpy2.f_mod(gmpy2.add(1, gmpy2.mul(gmpy2.mpz(x_i), N)), N2)

        # 3. 计算 H(l)^{\Delta * sk_i} mod N^2
        # sk_i 可能为负数，gmpy2.powmod 会自动计算逆元
        delta_sk_i = gmpy2.mul(delta, gmpy2.mpz(sk_i))
        term2 = gmpy2.powmod(H_l, delta_sk_i, N2)

        # 4. 组装密文 ct_i
        ct_i = gmpy2.f_mod(gmpy2.mul(term1, term2), N2)

        return ct_i

    def dkcom(self, dk_shares_for_j, k_dict_U_K):
        """
        DKCom((dk_{ij})_{i \in U_K}): 解密节点 j 处理收集到的份额，减去初始偏移量。
        - dk_shares_for_j: dict, { i: dk_ij } 包含了 U_K 中所有存活节点发给节点 j 的份额
        - k_dict_U_K: dict, { i: k_i } 包含了 U_K 中所有存活节点的 k_i 值
        """
        if self.pp is None:
            raise ValueError("请先运行 setup() 初始化公开参数 pp")

        # 存放处理后的份额 dk_j，论文重载了 dk_ij 符号，这里用字典的 key-value 来体现集合
        dk_j = {}

        for i, dk_ij in dk_shares_for_j.items():
            dk_j[i] = dk_ij

        return dk_j

    def pardec(self, dk_j, k_dict_U_K, ct_dict_U_M, l):
        """
        ParDec(dk_j, (k_i)_{i \in U_K}, (ct_i)_{i \in U_M}, l):
        解密节点 j 使用自己的调整后份额和接收到的密文，计算部分解密结果 y_j。
        """
        if self.pp is None:
            raise ValueError("请先运行 setup() 初始化公开参数 pp")

        N2 = self.pp['N2']

        # 将标签转换为 bytes
        if isinstance(l, str):
            l_bytes = l.encode('utf-8')
        elif isinstance(l, int):
            l_bytes = str(l).encode('utf-8')
        else:
            l_bytes = bytes(l)

        H_l = hash_to_zn2_star(l_bytes, N2)

        # 获取集合交集 U_M \cap U_K
        U_K = set(dk_j.keys())
        U_M = set(ct_dict_U_M.keys())
        U_intersect = U_M.intersection(U_K)

        if not U_intersect:
            raise ValueError("交集 U_M \cap U_K 为空，无法进行部分解密")

        # 1. 计算 \prod_{i \in 交集} (ct_i)^{k_i} mod N^2
        part_A = gmpy2.mpz(1)
        for i in U_intersect:
            ct_i = gmpy2.mpz(ct_dict_U_M[i])
            k_i = gmpy2.mpz(k_dict_U_K[i])
            term = gmpy2.powmod(ct_i, k_i, N2)
            part_A = gmpy2.f_mod(gmpy2.mul(part_A, term), N2)

        # 2. 计算 \sum_{i \in 交集} dk_ij
        sum_dk_ij = gmpy2.mpz(0)
        for i in U_intersect:
            sum_dk_ij = gmpy2.add(sum_dk_ij, dk_j[i])

        # 3. 计算 H(l)^{-\sum dk_ij} mod N^2
        # powmod 直接支持负指数求逆元
        part_B = gmpy2.powmod(H_l, -sum_dk_ij, N2)

        # 4. 计算最终的 y_j = part_A * part_B mod N^2
        y_j = gmpy2.f_mod(gmpy2.mul(part_A, part_B), N2)

        return y_j

    def comdec(self, y_dict, k_dict_U_K, u_m_intersect_keys):
        """
        ComDec((y_j)_{j \in U'_D}, (k_i)_{i \in U_K}):
        解密节点共同重构最终的聚合结果 y。
        """
        if self.pp is None:
            raise ValueError("请先运行 setup() 初始化公开参数 pp")

        N = self.pp['N']
        N2 = self.pp['N2']
        delta = self.pp['delta']  # 直接提取预计算的 delta

        # 1. 计算整数拉格朗日系数 L'_j = \Delta * \prod_{m \neq j} (-m) / (j - m)
        S = list(y_dict.keys())
        L_prime = {}
        for j in S:
            num = delta
            den = gmpy2.mpz(1)
            for m in S:
                if m != j:
                    num = gmpy2.mul(num, gmpy2.mpz(-m))
                    den = gmpy2.mul(den, gmpy2.mpz(j - m))
            # 在整数域上，这个除法一定能整除
            L_prime[j] = gmpy2.divexact(num, den)

        # 2. 计算 C = \prod y_j^{L'_j} mod N^2
        C = gmpy2.mpz(1)
        for j, y_j in y_dict.items():
            # L'_j 可能是负数，gmpy2.powmod 会自动在 N^2 下求逆
            term = gmpy2.powmod(y_j, L_prime[j], N2)
            C = gmpy2.f_mod(gmpy2.mul(C, term), N2)

        # 3. 提取明文: y'_delta = (C - 1 mod N^2) / N
        # 此时得到的是 \Delta * \sum k_i x_i
        y_prime_delta = gmpy2.divexact(gmpy2.sub(C, 1), N)

        # 乘以 \Delta^{-1} mod N 还原出真正的 y' = \sum k_i x_i
        delta_inv = gmpy2.invert(delta, N)
        y_prime = gmpy2.f_mod(gmpy2.mul(y_prime_delta, delta_inv), N)

        # 4. 计算权重调整
        # \sum_{i \in U_K} k_i
        sum_k_uk = gmpy2.mpz(0)
        for val in k_dict_U_K.values():
            sum_k_uk = gmpy2.add(sum_k_uk, gmpy2.mpz(val))

        # \sum_{i \in U_M \cap U_K} k_i
        sum_k_intersect = gmpy2.mpz(0)
        for i in u_m_intersect_keys:
            if i in k_dict_U_K:
                sum_k_intersect = gmpy2.add(sum_k_intersect, gmpy2.mpz(k_dict_U_K[i]))

        # 5. 直接在明文整数域上计算真正的整数结果
        numerator = gmpy2.mul(y_prime, sum_k_uk)
        # f_div 是 gmpy2 提供的向下取整的整数除法，等价于 Python 的 //
        y = gmpy2.f_div(numerator, sum_k_intersect)

        return y
