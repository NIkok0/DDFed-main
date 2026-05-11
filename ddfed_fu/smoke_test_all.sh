#!/usr/bin/env bash
# ============================================================
# DDFed-FU 全集冒烟测试聚合脚本
# 依次运行 FedEraser / Backdoor / QuickDrop 三个冒烟测试
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
# 声明一个通用的运行函数
# -----------------------------------------------------------
run_smoke() {
    local name="$1"
    local script="$2"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  开始: ${name}${NC}"
    echo -e "${YELLOW}  脚本: ${script}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

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

run_smoke "FedEraser"          "$SCRIPT_DIR/smoke_test_federaser.sh"
run_smoke "Backdoor Unlearning" "$SCRIPT_DIR/smoke_test_backdoor.sh"
run_smoke "QuickDrop"           "$SCRIPT_DIR/smoke_test_quickdrop.sh"

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