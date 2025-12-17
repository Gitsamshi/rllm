#!/bin/bash

# Ray Job Submission for Math Tool Training - Rollout.n Sweep
# This script submits multiple training jobs with different rollout.n values: 4, 8, 16, 32
# Batch size is fixed at 128, max_steps=2

set -x

# Get the current head pod name
HEAD_POD=$(kubectl get pods -l ray.io/cluster=rayml-efa-2workers-verl,ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

echo "Using Ray head pod: $HEAD_POD"

# Array of rollout.n values to sweep
ROLLOUT_N_VALUES=(16 24 32)

# Fixed batch size for all jobs
BATCH_SIZE=256

echo "Submitting Math Tool GRPO training jobs with rollout.n values: ${ROLLOUT_N_VALUES[*]}"
echo "Batch size is fixed at $BATCH_SIZE"
echo "Submitting with 5-minute intervals to avoid HuggingFace rate limiting"
echo ""

# Submit a job for each rollout.n value with 5-minute delay between submissions
for rollout_n in "${ROLLOUT_N_VALUES[@]}"; do
    echo "=============================================="
    echo "Submitting job with rollout.n=$rollout_n, batch_size=$BATCH_SIZE"
    echo "=============================================="
    
    kubectl exec $HEAD_POD -- ray job submit \
      --runtime-env-json '{
        "env_vars": {
          "WANDB_API_KEY": "0bf200379ea117dd7541973d7025d5e7b93ed93a",
          "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
          "PYTHONPATH": "/fsx/zzsamshi/rllm:/fsx/zzsamshi/rllm/verl"
        }
      }' \
      --no-wait \
      -- bash /fsx/zzsamshi/rllm/examples/math_tool/train_math_with_tool_ray_batchsize.sh $BATCH_SIZE $rollout_n
    
    echo "Job with rollout.n=$rollout_n submitted!"
    
    # Wait 5 minutes (300 seconds) before submitting the next job
    # Skip wait after the last job
    if [ "$rollout_n" != "${ROLLOUT_N_VALUES[-1]}" ]; then
        echo ""
        echo "Waiting 5 minutes before submitting next job..."
        sleep 300
        echo ""
    fi
done

echo "=============================================="
echo "All jobs submitted successfully!"
echo "=============================================="
echo ""
echo "To monitor jobs:"
echo "1. Check job status: kubectl exec $HEAD_POD -- ray job list"
echo "2. Check job logs:   kubectl exec $HEAD_POD -- ray job logs <job_id>"
echo "3. Stop job:         kubectl exec $HEAD_POD -- ray job stop <job_id>"
echo "4. Ray dashboard:    kubectl port-forward svc/rayml-efa-2workers-verl-head-svc 8265:8265"

