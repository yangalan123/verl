#!/bin/bash
#SBATCH --mail-user=chenghao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch2/chenghao/annealing_sampling/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch2/chenghao/annealing_sampling/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch2/chenghao/annealing_sampling/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=600gb
#SBATCH --job-name=run_verl_annealed_sampling
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=11:59:00
#SBATCH --signal=SIGUSR1@120

# slurm cluster requirement, no need to keep these three lines in your local machine
unset ROCR_VISIBLE_DEVICES
cd /net/scratch2/chenghao/annealing_sampling
source ~/miniconda3/etc/profile.d/conda.sh
# where you install the local verl env
conda activate /net/scratch2/chenghao/annealing_sampling/Minimal-RL/env
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
data=numina_math
project_name="minimal_rl_numina_math"
algorithm=grpo
model=Qwen2.5-Math-1.5B
model_name_or_path=Qwen/$model
rollout_n=4
decay_freq=2000
start_temp=1.2
end_temp=0.1
num_gpu_per_node=4
save_freq=10
test_freq=10
experiment_name="annealed_sampling_grpo_explore_${start_temp}_stable_${end_temp}_decay_freq_${decay_freq}"
# where you run minimal_rl_step0_data_creation.sh -- fix ROOT_DIR, math_train_path, math_test_path below
ROOT_DIR=/net/scratch2/chenghao/annealing_sampling/Minimal-RL

math_train_path=$ROOT_DIR/data/$data/train.parquet
math_test_path=$ROOT_DIR/data/math500/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"

mkdir -p logs/${project_name}

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
    actor_rollout_ref.rollout.annealed_sampling.exploration_temp=${start_temp} \
    actor_rollout_ref.rollout.annealed_sampling.stability_temp=${end_temp} \
    actor_rollout_ref.rollout.annealed_sampling.decay_freq=${decay_freq} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${num_gpu_per_node} \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=1 2>&1 | tee logs/${project_name}/${experiment_name}.log
