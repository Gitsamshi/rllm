#!/bin/bash
set -x

# Environment setup
export TOKENIZERS_PARALLELISM=true

# Use the shared FSx path
RLLM_DIR=/fsx/zzsamshi/rllm

# Change to the RLLM directory
cd "$RLLM_DIR"

# Add RLLM_DIR and verl to PYTHONPATH
export PYTHONPATH="${RLLM_DIR}:${RLLM_DIR}/verl:${PYTHONPATH}"

# ============== Configuration ==============
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-4B"}
MODEL_PORT=${MODEL_PORT:-30000}
BASE_URL="http://localhost:${MODEL_PORT}/v1"
N_PARALLEL=${N_PARALLEL:-64}
N_SAMPLES=${N_SAMPLES:-8}
OUTPUT_DIR=${OUTPUT_DIR:-"/fsx/zzsamshi/rllm/eval_results"}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

# Datasets to evaluate
DATASETS=${DATASETS:-"aime2024 aime2025 math500 olympiad amc23"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for this evaluation run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "============================================"
echo "Evaluation Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Model Port: $MODEL_PORT"
echo "  Base URL: $BASE_URL"
echo "  Tensor Parallel: $TENSOR_PARALLEL"
echo "  GPU Memory Util: $GPU_MEMORY_UTILIZATION"
echo "  Parallel Agents: $N_PARALLEL"
echo "  Samples per problem: $N_SAMPLES"
echo "  Datasets: $DATASETS"
echo "  Output Directory: $RUN_DIR"
echo "============================================"

# ============================================
# Step 1: Start vLLM Server
# ============================================
echo ""
echo "============================================"
echo "Starting vLLM server for model: $MODEL_NAME"
echo "============================================"

# Start vLLM server in background
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$MODEL_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --disable-log-requests \
    2>&1 | tee "${RUN_DIR}/vllm_server.log" &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Function to cleanup vLLM server
cleanup() {
    echo ""
    echo "============================================"
    echo "Cleaning up..."
    echo "============================================"
    
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null
        echo "vLLM server stopped."
    fi
    
    # Kill any remaining vLLM processes
    pkill -f "vllm.entrypoints.openai.api_server.*--port $MODEL_PORT" 2>/dev/null || true
    
    echo "Cleanup complete."
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=300  # 5 minutes timeout
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${MODEL_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    
    # Check if vLLM process is still running
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died unexpectedly!"
        echo "Check logs at: ${RUN_DIR}/vllm_server.log"
        exit 1
    fi
    
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Still waiting... ($WAITED/$MAX_WAIT seconds)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
    exit 1
fi

# ============================================
# Step 2: Run Evaluations
# ============================================
echo ""
echo "============================================"
echo "Starting evaluations..."
echo "============================================"

EVAL_SUCCESS=true

for DATASET in $DATASETS; do
    echo ""
    echo "============================================"
    echo "Evaluating dataset: $DATASET"
    echo "============================================"
    
    python3 examples/math_tool/eval_math_with_tool.py \
        --model_name "$MODEL_NAME" \
        --base_url "$BASE_URL" \
        --dataset "$DATASET" \
        --n_parallel "$N_PARALLEL" \
        --n_samples "$N_SAMPLES" \
        --output_dir "$RUN_DIR" \
        2>&1 | tee "${RUN_DIR}/${DATASET}_log.txt"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "WARNING: Evaluation for $DATASET failed!"
        EVAL_SUCCESS=false
    fi
done

# ============================================
# Step 3: Generate Summary
# ============================================
echo ""
echo "============================================"
echo "Generating evaluation summary..."
echo "============================================"

python3 -c "
import json
import os
from pathlib import Path

run_dir = '$RUN_DIR'
datasets = '$DATASETS'.split()

print('\n' + '='*60)
print('EVALUATION SUMMARY')
print('='*60)
print(f'Model: $MODEL_NAME')
print(f'Run Directory: {run_dir}')
print('='*60)

results = []
for dataset in datasets:
    score_file = Path(run_dir) / f'{dataset}_scores.json'
    if score_file.exists():
        with open(score_file) as f:
            data = json.load(f)
        results.append({
            'dataset': dataset,
            'pass_at_1': data.get('pass_at_1', 0.0),
            'pass_at_k': data.get('pass_at_k', 0.0),
            'total_problems': data.get('total_problems', 0),
            'duration_seconds': data.get('duration_seconds', 0)
        })
        print(f'{dataset:12} | Pass@1: {data.get(\"pass_at_1\", 0.0):.4f} | Pass@k: {data.get(\"pass_at_k\", 0.0):.4f} | Problems: {data.get(\"total_problems\", 0):4} | Time: {data.get(\"duration_seconds\", 0):.1f}s')
    else:
        print(f'{dataset:12} | Results not found')
        results.append({'dataset': dataset, 'error': 'Results not found'})

print('='*60)

# Calculate averages
valid_results = [r for r in results if 'pass_at_1' in r]
if valid_results:
    avg_pass_1 = sum(r['pass_at_1'] for r in valid_results) / len(valid_results)
    avg_pass_k = sum(r['pass_at_k'] for r in valid_results) / len(valid_results)
    print(f'{'AVERAGE':12} | Pass@1: {avg_pass_1:.4f} | Pass@k: {avg_pass_k:.4f}')
    print('='*60)

# Save summary
summary = {
    'model': '$MODEL_NAME',
    'run_directory': run_dir,
    'datasets_evaluated': len(valid_results),
    'datasets_failed': len(results) - len(valid_results),
    'average_pass_at_1': avg_pass_1 if valid_results else None,
    'average_pass_at_k': avg_pass_k if valid_results else None,
    'results': results
}

summary_file = Path(run_dir) / 'summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSummary saved to: {summary_file}')
"

echo ""
echo "============================================"
echo "All evaluations completed!"
echo "Results saved to: $RUN_DIR"
echo "============================================"

# The cleanup trap will automatically stop vLLM server on exit
