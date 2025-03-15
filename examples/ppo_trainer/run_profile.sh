#!/bin/bash

# Print help message if requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [TP_SIZE] [PP_SIZE] [GEN_LEN] [BSZ_PER_NODE] [ROLLOUT_N]"
    echo ""
    echo "Defaults:"
    echo "  TP_SIZE     = 8"
    echo "  PP_SIZE     = 8"
    echo "  GEN_LEN     = 128"
    echo "  BSZ_PER_NODE= 32"
    echo "  ROLLOUT_N   = 8"
    exit 0
fi

DATA_DIR=$HOME/data/gsm8k

# Use command line args with default values if not provided
TP_SIZE=${1:-8}
PP_SIZE=${2:-8}
DP_SIZE=${3:-1}
GEN_LEN=${4:-128}
BSZ=${5:-128}
ROLLOUT_N=${6:-8}
MICRO_BSZ_PER_GPU=1
GROUP_SHUFFLE=False
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct

echo "TP_SIZE=$TP_SIZE"
echo "PP_SIZE=$PP_SIZE"
echo "GEN_LEN=$GEN_LEN"
echo "BSZ=$BSZ"
echo "ROLLOUT_N=$ROLLOUT_N"

ray job submit --address="http://localhost:8265" \
  --runtime-env-json='{"working_dir": "./"}' \
  --no-wait \
  -- python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    actor_rollout_ref.rollout.name=sglang \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$BSZ \
    data.val_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=$GEN_LEN \
    +data.dummy=False \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.sequence_parallel=False \
    actor_rollout_ref.ref.megatron.sequence_parallel=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.rollout.ignore_eos=True \
    +actor_rollout_ref.rollout.min_new_tokens=$((GEN_LEN - 2)) \
    +actor_rollout_ref.rollout.group_shuffle=$GROUP_SHUFFLE \
    +actor_rollout_ref.rollout.n_groups=4 \
    +actor_rollout_ref.rollout.oversubscribe=False \
    +actor_rollout_ref.rollout.n_over=4 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_sglang_Profile_new' \
    trainer.experiment_name=pp-$PP_SIZE-tp-$TP_SIZE-dp-$DP_SIZE-gen-$GEN_LEN-bsz-$BSZ-rol-$ROLLOUT_N \
    +trainer.val_before_train=False \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$((PP_SIZE * TP_SIZE * DP_SIZE / 8)) \
    trainer.save_freq=-1 \
    trainer.test_freq=26 \
    trainer.total_training_steps=21 2>&1 | tee verl_demo.log
