#!/usr/bin/env bash
set -euxo pipefail

# usage: ./run_dapo_qwen2.5_7b.sh <group_shuffle>

if [ $# -ne 2 ]; then
    echo "Usage: ./run_dapo_qwen2.5_7b.sh <group_shuffle> <partial_rollout>"
    exit 1
fi

group_shuffle=$1
partial_rollout=$2

adv_estimator=ppo

kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 16))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=256
n_groups=4
n_resp_per_prompt=1
train_prompt_mini_bsz=256
train_micro_bsz_per_gpu=4
infer_micro_bsz_per_gpu=8

project_name='DAPO-PPO'
exp_name=Qwen2.5-7B
if [ "${group_shuffle}" = "True" ]; then
    exp_name="${exp_name}-GS"
    if [ "${partial_rollout}" = "True" ]; then
        exp_name="${exp_name}-PR"
    fi
fi

# Ray
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B"}
CKPTS_DIR=${CKPTS_DIR:-"/mnt/blob/ckpts_bugfix/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${HOME}/data/aime-2024.parquet"}

mkdir -p "${CKPTS_DIR}"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4


python3 -m verl.trainer.main_ppo \
    --config-path=./config --config-name='ppo_trainer' \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    +actor_rollout_ref.rollout.group_shuffle=${group_shuffle} \
    +actor_rollout_ref.rollout.n_groups=${n_groups} \
    +actor_rollout_ref.rollout.partial_rollout=${partial_rollout} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=True \
    critic.use_dynamic_bsz=${use_dynamic_bsz} \
    critic.forward_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    critic.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    critic.forward_micro_batch_size_per_gpu=${train_micro_bsz_per_gpu} \
    critic.ppo_micro_batch_size_per_gpu=${train_micro_bsz_per_gpu} \
    critic.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    critic.model.fsdp_config.param_offload=${offload} \
    critic.model.fsdp_config.optimizer_offload=${offload} \
    critic.model.use_remove_padding=True \
    critic.ulysses_sequence_parallel_size=${sp_size} \
    +custom_reward_function.overlong_buffer.enable=${enable_overlong_buffer} \
    +custom_reward_function.overlong_buffer.len=${overlong_buffer_len} \
    +custom_reward_function.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.total_epochs=100 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto