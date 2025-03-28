set -x

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet


train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

pdsh pkill -f sglang

ray job submit --address="http://localhost:8265" \
  --runtime-env-json='{"working_dir": "./", "env_vars": {"VERL_PPO_LOGGING_LEVEL": "DEBUG"}}' \
  -- python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_trainer'\
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.do_sample=False \
    +actor_rollout_ref.model.use_liger=True \
    +actor_rollout_ref.rollout.group_shuffle=True \
    +actor_rollout_ref.rollout.n_groups=4 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-7B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.param_offload=True \
    critic.ulysses_sequence_parallel_size=4 \
    critic.model.use_remove_padding=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_sglang_gsm8k' \
    trainer.experiment_name='qwen2_5_7b_GS' \
    trainer.n_gpus_per_node=8 \
    +trainer.val_before_train=True \
    trainer.nnodes=2 \
    trainer.save_freq=50 \
    trainer.test_freq=26 \
    trainer.total_epochs=100 $@
