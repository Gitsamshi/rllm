#!/bin/bash
set -x

# Wandb API Key
export WANDB_API_KEY=0bf200379ea117dd7541973d7025d5e7b93ed93a

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# ============== Configuration Variables ==============
policy_path=Qwen/Qwen2.5-Math-1.5B

# Accept batch size as first argument, default to 128
rollout_batch_size=${1:-128}
# Accept rollout.n as second argument, default to 8
n_samples_per_prompts=${2:-8}
total_epochs=5
temperature=1.0
ppo_mini_batch_size=64  # Fixed at 64 as requested
lr=1e-6
kl_loss_coef=0.0
kl_coef=0.0
entropy_coeff=0
max_prompt_length=1200
max_gen_length=2800
max_steps=2
max_ckpt_to_keep=5

dataset_name=torl_math
# Extract model name for run_name (e.g., Qwen/Qwen2.5-Math-1.5B -> qwen2.5_math_1.5b)
model_name=$(echo $policy_path | sed 's|Qwen/||' | sed 's|-|_|g' | tr '[:upper:]' '[:lower:]')
run_name=rl.grpo_${model_name}_${dataset_name}_bs${rollout_batch_size}_n${n_samples_per_prompts}_kl${kl_loss_coef}_maxsteps${max_steps}

# Output directory for checkpoints (on shared FSx filesystem)
output_dir=/fsx/zzsamshi/rllm/checkpoints/torl/${run_name}

# =====================================================

# Use the shared FSx path directly (not the container's /workspace/rllm)
RLLM_DIR=/fsx/zzsamshi/rllm

# Change to the RLLM directory so Python can find the examples module
cd "$RLLM_DIR"

# Add RLLM_DIR and verl to PYTHONPATH so examples and verl modules can be found
export PYTHONPATH="${RLLM_DIR}:${RLLM_DIR}/verl:${PYTHONPATH}"

python3 -m examples.math_tool.train_math_with_tool \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$rollout_batch_size \
    data.val_batch_size=500 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_gen_length \
    actor_rollout_ref.model.path=$policy_path \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_samples_per_prompts \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='torl' \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=$max_ckpt_to_keep \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$output_dir \
    rllm.agent.max_steps=$max_steps \
    rllm.stepwise_advantage.enable=False \
    trainer.total_epochs=$total_epochs \
    trainer.log_episodes=True

