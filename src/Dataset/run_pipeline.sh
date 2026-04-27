set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../../../04_experiments/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.pid"

echo "================================================================"
echo "  CSI Detection Pipeline -- $(date)"
echo "  Datasets: gsm8k -> dailydialog -> rocstories -> recipenlg"
echo "  Log: ${LOG_FILE}"
echo "================================================================"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

nohup python3 -u "${SCRIPT_DIR}/03_process.py" \
    --datasets gsm8k dailydialog rocstories recipenlg \
    --device cuda:0 \
    --ner_gpu 0 \
    --batch_size 8 \
    --checkpoint_freq 100 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "  PID: ${PID}"
echo "  PID file: ${PID_FILE}"
echo "  Monitor: tail -f ${LOG_FILE}"
echo "  Stop: kill ${PID}"
echo "================================================================"
