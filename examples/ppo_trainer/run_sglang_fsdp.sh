DATA_DIR=$HOME/data/gsm8k

TP_SIZE=8
PP_SIZE=2

ray job submit --address="http://localhost:8265" \
  --runtime-env-json='{"working_dir": "./"}' \
  -- python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.name=sglang \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 2>&1 | tee verl_demo.log