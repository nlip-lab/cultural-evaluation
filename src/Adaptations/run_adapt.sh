set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INPUT="${PROJECT_ROOT}/03_data/processed/benchmark.jsonl"
LOG_DIR="${PROJECT_ROOT}/04_experiments/logs"
LOG_FILE="${LOG_DIR}/adapt_cot_run.log"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1,2"

if [[ ! -f "${INPUT}" ]]; then
    echo "ERROR: benchmark.jsonl not found at: ${INPUT}" >&2
    exit 1
fi

LINE_COUNT=$(wc -l < "${INPUT}")
if [[ "${LINE_COUNT}" -lt 6000 ]]; then
    echo "WARNING: benchmark.jsonl has only ${LINE_COUNT} lines (expected >=6000)." >&2
fi

mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "  Cultural Adaptation -- cot prompt mode"
echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  GPUs:     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  Input:    ${INPUT} (${LINE_COUNT} lines)"
echo "  Log:      ${LOG_FILE}"
echo "  PID:      $$"
echo "============================================================"

python -u "${SCRIPT_DIR}/adapt.py" \
    --input    "${INPUT}"  \
    --model    all         \
    --prompt_level cot     \
    --temperature 0.0      \
    --gpus "1,2"

EXIT_CODE=$?

echo "============================================================"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Exit code: ${EXIT_CODE}"
echo "============================================================"

exit "${EXIT_CODE}"
