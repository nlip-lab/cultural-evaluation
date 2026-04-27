set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="${PROJECT_ROOT}/04_experiments/logs/reverse_adapt.log"
PID_FILE="$SCRIPT_DIR/run_reverse.pid"
OUTPUT_DIR="${PROJECT_ROOT}/03_data/processed/reverse_adapted_data"
echo $$ > "$PID_FILE"

echo "============================================================"
echo "  Reverse Cultural Adaptation -- GPU Detection"
echo "============================================================"

TARGET_GPUS=(0 2)
FREE_GPUS=()
while IFS=',' read -r GPU_IDX FREE_MIB; do
    GPU_IDX="${GPU_IDX// /}"
    FREE_MIB="${FREE_MIB// /}"
    is_target=0
    for tg in "${TARGET_GPUS[@]}"; do
        [ "$GPU_IDX" = "$tg" ] && is_target=1 && break
    done
    [ "$is_target" -eq 0 ] && continue
    if [ "$FREE_MIB" -ge 20480 ] 2>/dev/null; then
        FREE_GPUS+=("$GPU_IDX")
        echo "  GPU $GPU_IDX: ${FREE_MIB} MiB free (>=20 GB -- usable)"
    else
        echo "  GPU $GPU_IDX: ${FREE_MIB} MiB free (< 20 GB -- busy, skipping)"
    fi
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "")

if [ ${#FREE_GPUS[@]} -eq 0 ]; then
    echo ""
    echo "  ERROR: No GPUs with >=20 GB free VRAM found."
    exit 1
fi

NUM_GPUS=${#FREE_GPUS[@]}
GPU_LIST=$(IFS=','; echo "${FREE_GPUS[*]}")
echo ""
echo "  Using GPUs: $GPU_LIST  (${NUM_GPUS} GPU(s))"

if [ "$NUM_GPUS" -ge 2 ]; then
    MAX_TP=2
    SKIP_TP2=""
    MODELS="all"
    echo "  Strategy: 2 GPUs -- TP=2 for large models, TP=1 for small models."
else
    MAX_TP=1
    SKIP_TP2="--skip_tp2"
    MODELS="all"
    echo "  Strategy: 1 GPU -- skipping TP=2 models."
fi

echo ""
echo "  Output dir: $OUTPUT_DIR"
echo "  Start time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export VLLM_LOGGING_LEVEL="WARNING"
export VLLM_NO_USAGE_STATS="1"
export TOKENIZERS_PARALLELISM="false"

python3 -u reverse_adapt.py \
    --output_dir  "$OUTPUT_DIR"  \
    --model       "$MODELS"      \
    --prompt_level all           \
    --domain      all            \
    --temperature 0.0            \
    --gpus        "$GPU_LIST"    \
    --max_tp      "$MAX_TP"      \
    $SKIP_TP2

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  Reverse adaptation finished -- exit code: $EXIT_CODE"
echo "  End time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"

rm -f "$PID_FILE"
exit $EXIT_CODE
