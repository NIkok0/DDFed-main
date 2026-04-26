# DDFed/ddfed_crypto/math_utils.py

import os
import hashlib
import gmpy2


def get_random_prime_over_bound(bound_bits):
    """
    生成一个大于 2^bound_bits 的随机素数。
    """
    # 使用系统真随机数初始化 gmpy2 随机状态
    seed = int.from_bytes(os.urandom(16), 'big')
    rs = gmpy2.random_state(seed)

    # 设定基线 2^bound_bits
    base = gmpy2.mpz(2) ** bound_bits
    # 生成一个与 base 同等量级的随机数加上去，确保严格大于 base
    rand_val = gmpy2.mpz_random(rs, base)

    # 寻找下一个素数
    return gmpy2.next_prime(base + rand_val)


def generate_safe_prime(bound_bits):
    """
    生成安全素数 p = 2p' + 1，其中 p' > 2^bound_bits 且 p' 也是素数。
    返回 (p, p')
    """
    while True:
        p_prime = get_random_prime_over_bound(bound_bits)
        p = 2 * p_prime + 1
        if gmpy2.is_prime(p):
            return p, p_prime


def hash_to_zn2_star(data_bytes, N2):
    """
    哈希函数 H: {0,1}* -> Z_{N^2}^*
    """
    h = hashlib.sha512(data_bytes).digest()
    val = gmpy2.mpz(int.from_bytes(h, 'big'))

    # 映射到 Z_{N^2}
    val = val % N2

    # 确保在 Z_{N^2}^* 中 (即 gcd(val, N^2) == 1)
    # 对于实际生成的大 N，gcd != 1 的概率可以忽略不计，但为了严谨性加上检查
    while gmpy2.gcd(val, N2) != 1:
        val = (val + 1) % N2

    return val