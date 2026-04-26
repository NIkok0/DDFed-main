# DDFed/ddfed_crypto/config.py

# 安全参数 \lambda
LAMBDA_SEC = 128

# 解密节点总数 n
N_DECRYPTORS = 10

# 阈值 t
T_THRESHOLD = 5

# 加密节点总数
N_ENCRYPTORS = 50

# ================= 掉线模拟配置 =================

# DKShare 阶段掉线的加密节点数 (未进入 U_K 集合)
N_DROPPED_K = 0

# Enc 加密阶段掉线的加密节点数 (未进入 U_M 集合)
N_DROPPED_M = 0

# ParDec 部分解密阶段掉线的解密节点数 (未进入 U_D' 集合)
# 注意: (N_DECRYPTORS - N_DROPPED_D) 必须 >= T_THRESHOLD 才能成功解密
N_DROPPED_D = 2
