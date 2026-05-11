#!/usr/bin/env bash
# ============================================================
# DDFed-FU 环境完整性检查脚本
# 检查三个联邦遗忘实验所需的所有依赖和环境条件
# ============================================================
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

PASS=0
FAIL=0
WARN=0

echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║        DDFed-FU 环境完整性检查                          ║${NC}"
echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ---------- 1. Python ----------
echo -e "${BOLD}[1/7] Python 环境${NC}"
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 8 ]; then
        echo -e "  ${GREEN}✅${NC} Python $PY_VER (>= 3.8)"
        PASS=$((PASS+1))
    else
        echo -e "  ${RED}❌${NC} Python $PY_VER (需要 >= 3.8)"
        FAIL=$((FAIL+1))
    fi
else
    echo -e "  ${RED}❌${NC} 未找到 python3"
    FAIL=$((FAIL+1))
fi
echo ""

# ---------- 2. PyTorch ----------
echo -e "${BOLD}[2/7] PyTorch 生态${NC}"
TORCH_OK=true

if python3 -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo -e "  ${GREEN}✅${NC} torch $TORCH_VER"
else
    echo -e "  ${RED}❌${NC} torch 未安装"
    FAIL=$((FAIL+1))
    TORCH_OK=false
fi

if python3 -c "import torchvision" 2>/dev/null; then
    TV_VER=$(python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null)
    echo -e "  ${GREEN}✅${NC} torchvision $TV_VER"
else
    echo -e "  ${RED}❌${NC} torchvision 未安装"
    FAIL=$((FAIL+1))
    TORCH_OK=false
fi

# CUDA 检测
if $TORCH_OK; then
    CUDA_AVAIL=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAIL" = "True" ]; then
        CUDA_DEV_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        CUDA_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo -e "  ${GREEN}✅${NC} CUDA 可用 — $CUDA_DEV_COUNT 设备: $CUDA_NAME"
        PASS=$((PASS+1))
    else
        echo -e "  ${YELLOW}⚠️${NC}  CUDA 不可用，将使用 CPU（训练会较慢）"
        WARN=$((WARN+1))
    fi
fi
echo ""

# ---------- 3. 公共依赖 ----------
echo -e "${BOLD}[3/7] 公共 Python 依赖${NC}"

check_pkg() {
    local pkg=$1
    local min_ver=$2
    if python3 -c "import $pkg" 2>/dev/null; then
        local ver=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
        echo -e "  ${GREEN}✅${NC} $pkg $ver"
        return 0
    else
        echo -e "  ${RED}❌${NC} $pkg 未安装"
        return 1
    fi
}

COMMON_OK=true
check_pkg numpy || { FAIL=$((FAIL+1)); COMMON_OK=false; }
check_pkg scipy || { FAIL=$((FAIL+1)); COMMON_OK=false; }
check_pkg sklearn || { FAIL=$((FAIL+1)); COMMON_OK=false; }
check_pkg tqdm || { FAIL=$((FAIL+1)); COMMON_OK=false; }
check_pkg matplotlib || { FAIL=$((FAIL+1)); COMMON_OK=false; }
check_pkg PIL || { FAIL=$((FAIL+1)); COMMON_OK=false; }

if $COMMON_OK; then
    PASS=$((PASS+1))
fi
echo ""

# ---------- 4. FedEraser 特有：xgboost ----------
echo -e "${BOLD}[4/7] FedEraser 需求（MIA 攻击模型）${NC}"
if python3 -c "import xgboost" 2>/dev/null; then
    XGB_VER=$(python3 -c "import xgboost; print(xgboost.__version__)" 2>/dev/null)
    echo -e "  ${GREEN}✅${NC} xgboost $XGB_VER"
    PASS=$((PASS+1))
else
    echo -e "  ${YELLOW}⚠️${NC}  xgboost 未安装（MIA 攻击需 --skip_mia 跳过）"
    WARN=$((WARN+1))
fi
echo ""

# ---------- 5. Backdoor Unlearning 特有 ----------
echo -e "${BOLD}[5/7] Backdoor Unlearning 需求${NC}"
# 依赖与公共依赖基本一致，无额外特有包
echo -e "  ${GREEN}✅${NC} 依赖与公共依赖一致，无需额外检查"
PASS=$((PASS+1))
echo ""

# ---------- 6. QuickDrop 特有：seaborn ----------
echo -e "${BOLD}[6/7] QuickDrop 需求${NC}"
if python3 -c "import seaborn" 2>/dev/null; then
    SNS_VER=$(python3 -c "import seaborn; print(seaborn.__version__)" 2>/dev/null)
    echo -e "  ${GREEN}✅${NC} seaborn $SNS_VER"
    PASS=$((PASS+1))
else
    echo -e "  ${YELLOW}⚠️${NC}  seaborn 未安装（QuickDrop 画图需要）"
    WARN=$((WARN+1))
fi
echo ""

# ---------- 7. 数据目录 ----------
echo -e "${BOLD}[7/7] 数据目录${NC}"
DATA_DIR="$(cd "$(dirname "$0")" && pwd)/../data"
if [ -d "$DATA_DIR" ]; then
    echo -e "  ${GREEN}✅${NC} data/ 目录存在: $DATA_DIR"
    PASS=$((PASS+1))
else
    echo -e "  ${YELLOW}⚠️${NC}  data/ 目录不存在，PyTorch 将自动下载数据集"
    WARN=$((WARN+1))
fi
echo ""

# ---------- 汇总 ----------
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}检查结果:${NC}"
echo -e "  ${GREEN}通过: $PASS${NC}"
echo -e "  ${YELLOW}警告: $WARN${NC}"
echo -e "  ${RED}失败: $FAIL${NC}"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}${BOLD}❌ 环境不完整！请安装缺失的依赖后再运行实验。${NC}"
    echo ""
    echo "安装建议："
    echo "  pip install torch torchvision numpy scipy scikit-learn tqdm matplotlib Pillow"
    echo "  pip install xgboost    # FedEraser MIA 需要（可选）"
    echo "  pip install seaborn    # QuickDrop 绘图需要（可选）"
    exit 1
else
    echo -e "${GREEN}${BOLD}✅ 环境检查全部通过！可以运行实验。${NC}"
    exit 0
fi