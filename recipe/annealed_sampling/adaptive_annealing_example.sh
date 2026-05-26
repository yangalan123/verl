#!/bin/bash
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=ALL
#SBATCH --output=/path/to/logs/%A_%a.%N.stdout
#SBATCH --error=/path/to/logs/%A_%a.%N.stderr
#SBATCH --chdir=/path/to/working/dir
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=600gb
#SBATCH --job-name=adaptive_annealing_example
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=11:59:00
#SBATCH --signal=SIGUSR1@120

# Setup environment
unset ROCR_VISIBLE_DEVICES
cd /path/to/working/dir
source ~/miniconda3/etc/profile.d/conda.sh
conda activate your_env_name
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Configuration
data=gsm8k
project_name="adaptive_annealing_example"
algorithm=grpo
model=Qwen2.5-1.5B-Instruct
model_name_or_path=Qwen/$model

# Rollout configuration
rollout_n=4
k_max=16

# Adaptive annealing configuration
exploration_temp=1.2
stability_temp=0.1
decay_freq=50
warmup_period=10
decay_mode="adaptive"  # New adaptive mode
adaptive_decay=True     # Enable adaptive decay

# Cluster configuration
num_gpu_per_node=4
save_freq=10
test_freq=10

experiment_name="adaptive_annealing_${decay_mode}_explore_${exploration_temp}_stable_${stability_temp}_warmup_${warmup_period}"

# Data paths
ROOT_DIR=/path/to/your/data
train_path=$ROOT_DIR/data/$data/train.parquet
test_path=$ROOT_DIR/data/$data/test.parquet

train_files="['$train_path']"
test_files="['$test_path']"

log_dir=$ROOT_DIR/logs/${project_name}/${experiment_name}
mkdir -p $log_dir

# Cache directory for historical data
cache_dir=$ROOT_DIR/cache/annealing_history

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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.annealed_sampling.enable=True \
    actor_rollout_ref.rollout.annealed_sampling.decay_mode=${decay_mode} \
    actor_rollout_ref.rollout.annealed_sampling.exploration_temp=${exploration_temp} \
    actor_rollout_ref.rollout.annealed_sampling.stability_temp=${stability_temp} \
    actor_rollout_ref.rollout.annealed_sampling.decay_freq=${decay_freq} \
    actor_rollout_ref.rollout.annealed_sampling.warmup_period=${warmup_period} \
    actor_rollout_ref.rollout.annealed_sampling.adaptive_decay=${adaptive_decay} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.annealing_cache_dir=$cache_dir \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.n_gpus_per_node=$num_gpu_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=15 2>&1 