#!/usr/bin/env bash
# ============================================================
# FedEraser 冒烟测试
# 最小配置：MNIST, 5 total clients, 2 per round, 2 global epoch
# 跳过 MIA（不需要 xgboost）
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  FedEraser 冒烟测试"
echo "  data=MNIST  total_clients=5  n_per_round=2"
echo "  global_epoch=2  local_epoch=1  --skip_mia"
echo "=============================================="
echo ""

python FedEraser-Code/Fed_Unlearn_main.py \
    --data_name mnist \
    --n_clients 2 \
    --n_total_clients 5 \
    --global_epoch 2 \
    --local_epoch 1 \
    --local_batch_size 32 \
    --seed 42 \
    --skip_mia

echo ""
echo "✅ FedEraser 冒烟测试完成"
echo "🗑️  测试通过，销毁冒烟脚本..."
rm -- "$SELF"
