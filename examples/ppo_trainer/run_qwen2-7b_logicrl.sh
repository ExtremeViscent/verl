# Discliamer: the model used in the script is only for academic purpose.
set -x

DATA_DIR=$HOME/data/kk

TP_SIZE=8
PP_SIZE=2
MICRO_BSZ_PER_GPU=1
GROUP_SHUFFLE=False
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct

export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

ray job submit --address="http://localhost:8265" \
  --runtime-env-json='{"working_dir": "./"}' \
  --no-wait \
  -- python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.do_sample=True \
    +actor_rollout_ref.rollout.group_shuffle=$GROUP_SHUFFLE \
    +actor_rollout_ref.rollout.n_groups=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_example' \
    +trainer.val_before_train=False \
    trainer.experiment_name='Qwen2.5-7B-logicrl' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
