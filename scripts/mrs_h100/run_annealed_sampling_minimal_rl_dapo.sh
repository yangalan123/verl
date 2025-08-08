#!/bin/bash

#SBATCH --chdir=/fsx/zhuokai/verl/
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=annealed_sampling_minimal_rl_dapo_temp_1_2_llama
#SBATCH --output=/fsx/zhuokai/verl/slurm/annealed_sampling_minimal_rl_dapo_temp_1_2_llama.stdout
#SBATCH --error=/fsx/zhuokai/verl/slurm/annealed_sampling_minimal_rl_dapo_temp_1_2_llama.stderr


data=numina_math
project_name="minimal_rl_numina_math"
algorithm=grpo
# model=Qwen2.5-Math-1.5B
# model_name_or_path=Qwen/$model
model=Llama-3.2-1B-Instruct
model_name_or_path=meta-llama/$model
# for grpo rollout
rollout_n=4
# rollout_n=16 (to conform better with DAPO)
# for mean@K computation
k_max=16
# config for annealed sampling
decay_freq=250
start_temp=1.2
end_temp=0.1
warmup_period=10
# config for cluster
num_gpu_per_node=8
save_freq=10
test_freq=10
temperature=1.2
experiment_name="dapo_baseline_without_dynamic_sampling_temperature_${temperature}_${model}_zzk"
# where you run minimal_rl_step0_data_creation.sh -- fix ROOT_DIR, math_train_path, math_test_path below
ROOT_DIR=/fsx/zhuokai/verl/

math_train_path=$ROOT_DIR/data/$data/train.parquet
math_test_path=$ROOT_DIR/data/math500/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"

log_dir=$ROOT_DIR/logs/${project_name}/${experiment_name}
mkdir -p $log_dir
# TODO: add the following if needed
# copied from https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_wo_ds_qwen2.5_32b.sh
    # for optimizer warmup
    # actor_rollout_ref.actor.optim.lr_warm_up_steps=10 \
    # default: 0.01
    # actor_rollout_ref.actor.optim.weight_decay=0.1 \
    # for performance/efficiency
    # actor_rollout_ref.actor.use_dynamic_bsz=True \
    # actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    # actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \

PYTHONUNBUFFERED=1 VLLM_USE_V1=0 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_name_or_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${num_gpu_per_node} \
    trainer.rollout_data_dir=${log_dir}/rollout_data \
    trainer.validation_data_dir=${log_dir}/validation_data \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=1 2>&1 | tee logs/${project_name}/${experiment_name}.log
