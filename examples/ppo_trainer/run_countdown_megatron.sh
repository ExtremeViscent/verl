DATA_DIR=$HOME/data/countdown

TP_SIZE=8
PP_SIZE=2
MICRO_BSZ_PER_GPU=1
GROUP_SHUFFLE=False
BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct

ray job submit --address="http://localhost:8265" \
  --runtime-env-json='{"working_dir": "./"}' \
  -- python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    actor_rollout_ref.rollout.name=sglang \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    +actor_rollout_ref.rollout.group_shuffle=$GROUP_SHUFFLE \
    +actor_rollout_ref.rollout.n_groups=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    critic.optim.lr=1e-5 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=$MICRO_BSZ_PER_GPU \
    critic.megatron.tensor_model_parallel_size=$TP_SIZE \
    critic.megatron.pipeline_model_parallel_size=$PP_SIZE \
    critic.model.enable_gradient_checkpointing=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_sglang_tinyzero' \
    trainer.experiment_name=llama-3b-GroupShuffle-$GROUP_SHUFFLE-8k \
    +trainer.val_before_train=True \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=26 \
    trainer.total_epochs=100 2>&1 | tee verl_demo.log