#!/usr/bin/env bash
# ============================================================
# QuickDrop 冒烟测试
# 1. 生成 DiLICHET 环境（MNIST, 10 clients）
# 2. 运行 QuickDrop unlearning（2 轮通信, 1 轮 local epoch, CPU）
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
QD_ROOT="$SCRIPT_DIR/quickdrop-main"

echo "=============================================="
echo "  QuickDrop 冒烟测试"
echo "  Step 1: 生成 DiLICHET FL 环境"
echo "=============================================="
echo ""

# Step 1: 生成 dilichlet 分配环境
python "$QD_ROOT/env_generator/dilichlet_allocator/dilichlet_allocator.py" \
    --dataset_name MNIST \
    --num_clients 10 \
    --alpha 0.1 \
    --seed 42

echo ""
echo "  DiLICHET 环境生成完成"
echo ""

echo "=============================================="
echo "  Step 2: 运行 QuickDrop Unlearning"
echo "  dataset=MNIST  env=mnist-seed42-u10-alpha0.1"
echo "  communication_round=2  local_epoch=1  device=cpu"
echo "=============================================="
echo ""

cd "$QD_ROOT/reproduce/fig2_with_recovering/code"

python reproduce_single_unlearning.py \
    --dataset MNIST \
    --model ConvNet \
    --env mnist-seed42-u10-alpha0.1 \
    --env_path ../../../../env \
    --data_path ../../../../data \
    --strategy quickdrop-affine \
    --device cpu \
    --communication_round 2 \
    --local_epoch 1 \
    --learning_rate 0.02 \
    --scale 0.01 \
    --forgetting_epoch 1 \
    --forgetting_rate 0.004 \
    --recovering_round 1 \
    --seed 42 \
    --with_affine_dataset False \
    --weight_decay 0.001 \
    --momentum 0.0 \
    --batch_size 64 \
    --num_workers 0

echo ""
echo "✅ QuickDrop 冒烟测试完成"
echo "🗑️  测试通过，销毁冒烟脚本..."
rm -- "$SELF"
