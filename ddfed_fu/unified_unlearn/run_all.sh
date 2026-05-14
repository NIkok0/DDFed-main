#!/usr/bin/env bash
# ============================================================
#  Federated Unlearning — 20-experiment batch runner
#
#  Algorithms (5):  Baseline_FedSGA, Baseline_Backdoor/Neurotoxin,
#                   Baseline_FedQuickDrop, Proposed_ddfu_client,
#                   Proposed_ddfu_sample
#  Datasets  (2):  FashionMNIST, CIFAR10
#  Alphas    (2):  100 (mild non-IID), 0.1 (strong non-IID)
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

ALGOS=(
  "Baseline_FedSGA"
  "Baseline_Neurotoxin"
  "Baseline_FedQuickDrop"
  "Proposed_ddfu_client"
  "Proposed_ddfu_sample"
)
DATASETS=("FashionMNIST" "CIFAR10")
ALPHAS=("100.0" "0.1")

TOTAL=$(( ${#ALGOS[@]} * ${#DATASETS[@]} * ${#ALPHAS[@]} ))
echo "============================================================"
echo "  Total experiments: ${TOTAL}"
echo "  Started at: $(date)"
echo "============================================================"

COUNT=0
FAIL=0

for DS in "${DATASETS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for ALGO in "${ALGOS[@]}"; do
      COUNT=$((COUNT + 1))
      RUN_ID="[${COUNT}/${TOTAL}] ${ALGO} | ${DS} α=${ALPHA}"
      echo ""
      echo ">>>> ${RUN_ID}"

      START_TS="$(date +%s)"
      if ${PYTHON} -m ddfed_fu.unified_unlearn.main \
          --dataset "${DS}" \
          --alpha "${ALPHA}" \
          --algo "${ALGO}" \
          --pretrain 30 \
          --unlearn 20 2>&1; then
        ELAPSED=$(( $(date +%s) - START_TS ))
        echo "<<<< PASS  ${RUN_ID}  (${ELAPSED}s)"
      else
        FAIL=$((FAIL + 1))
        echo "<<<< FAIL  ${RUN_ID}"
      fi
    done
  done
done

echo ""
echo "============================================================"
echo "  Completed at: $(date)"
echo "  Total: ${TOTAL}   Pass: $(( TOTAL - FAIL ))   Fail: ${FAIL}"
echo "============================================================"
exit ${FAIL}