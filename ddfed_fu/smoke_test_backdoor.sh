#!/usr/bin/env bash
# ============================================================
# Backdoor Unlearning 冒烟测试
# 最小配置：CIFAR-10, 5 rounds, poison+unlearn
# 毒化：第1轮开始, 持续2轮, 遗忘1轮
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR/federated_backdoor_unlearning-main"

echo "=============================================="
echo "  Backdoor Unlearning 冒烟测试"
echo "  data=CIFAR-10  rounds=5  poison+unlearn"
echo "  poison_start=1  poison_dur=2  unlearn_dur=1"
echo "=============================================="
echo ""

python main.py \
    --num_rounds 5 \
    --data_name cifar10 \
    --n_clients 3 \
    --poison \
    --unlearn \
    --poison_strategy normal \
    --poison_start_round 1 \
    --poison_duration 2 \
    --unlearn_duration 1

echo ""
echo "✅ Backdoor Unlearning 冒烟测试完成"
echo "🗑️  测试通过，销毁冒烟脚本..."
rm -- "$SELF"
