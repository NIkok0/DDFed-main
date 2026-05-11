#!/usr/bin/env bash
# ============================================================
# DDFed-FU 全集冒烟测试聚合脚本（基石）
# 动态创建并运行 FedEraser / Backdoor / QuickDrop 三个冒烟测试
# 每个子脚本执行完毕后自毁
# 用法: bash smoke_test_all.sh
# ============================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# -----------------------------------------------------------
# 颜色输出
# -----------------------------------------------------------
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
NC="\033[0m"

PASS_COUNT=0
FAIL_COUNT=0
FAILED_TESTS=""

# -----------------------------------------------------------
# 子脚本内容定义（heredoc 用于动态创建自毁脚本）
# -----------------------------------------------------------
FEDERASER_CONTENT='
#!/usr/bin/env bash
# FedEraser 冒烟测试（自毁）
set -e
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$0")"

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
    --forget_client_idx 1 \
    --skip_mia

echo ""
echo "✅ FedEraser 冒烟测试完成"
echo "🗑️  自毁脚本..."
rm -- "$SELF"
'

BACKDOOR_CONTENT='
#!/usr/bin/env bash
# Backdoor Unlearning (Neurotoxin) 冒烟测试（自毁）
set -e
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$0")"

echo "=============================================="
echo "  Backdoor Unlearning (Neurotoxin) 冒烟测试"
echo "  data=fashion-mnist  n_selected_clients=3"
echo "  rounds=5  local_epoch=1  batch_size=32"
echo "=============================================="
echo ""

python federated_backdoor_unlearning-main/main.py \
    --dataset fashion-mnist \
    --n_selected_clients 3 \
    --rounds 5 \
    --local_epoch 1 \
    --batch_size 32 \
    --seed 42

echo ""
echo "✅ Backdoor Unlearning 冒烟测试完成"
echo "🗑️  自毁脚本..."
rm -- "$SELF"
'

QUICKDROP_CONTENT='
#!/usr/bin/env bash
# QuickDrop 冒烟测试（自毁）
set -e
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QD_ROOT="$SCRIPT_DIR/Env"

export PYTHONPATH="$QD_ROOT:$PYTHONPATH"

echo "=============================================="
echo "  QuickDrop 冒烟测试"
echo "  Step 1: 生成 DiLICHET FL 环境"
echo "=============================================="
echo ""

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
echo "🗑️  自毁脚本..."
rm -- "$SELF"
'

# -----------------------------------------------------------
# 动态创建自毁子脚本并运行
# 参数: $1=测试名称  $2=脚本文件名  $3=子脚本内容
# -----------------------------------------------------------
run_smoke() {
    local name="$1"
    local script="$SCRIPT_DIR/$2"
    local content="$3"

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  开始: ${name}${NC}"
    echo -e "${YELLOW}  脚本: ${script}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # 动态创建子脚本
    echo "$content" > "$script"
    chmod +x "$script"

    if ! bash "$script"; then
        echo -e "${RED}❌ ${name} 失败${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_TESTS="${FAILED_TESTS}  ${name}\n"
    else
        echo -e "${GREEN}✅ ${name} 通过${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
    fi
}

# -----------------------------------------------------------
# 运行三个冒烟测试
# -----------------------------------------------------------
echo "============================================================"
echo "  DDFed-FU 冒烟测试全集"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

run_smoke "FedEraser"           "smoke_test_federaser.sh" "$FEDERASER_CONTENT"
run_smoke "Backdoor Unlearning" "smoke_test_backdoor.sh"  "$BACKDOOR_CONTENT"
run_smoke "QuickDrop"           "smoke_test_quickdrop.sh" "$QUICKDROP_CONTENT"

# -----------------------------------------------------------
# 汇总结果
# -----------------------------------------------------------
echo ""
echo "============================================================"
echo "  DDFed-FU 冒烟测试结果汇总"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo -e "  ${GREEN}通过: ${PASS_COUNT}${NC}"
echo -e "  ${RED}失败: ${FAIL_COUNT}${NC}"

if [ -n "$FAILED_TESTS" ]; then
    echo ""
    echo -e "${RED}  失败的测试:${NC}"
    echo -e "${RED}${FAILED_TESTS}${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}  ✅ 全部冒烟测试通过!${NC}"
fi