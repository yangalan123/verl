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

#train_path=./datasets/aime_83_23_train.parquet
#test_path=./datasets/aime_83_23_val.parquet
unset ROCR_VISIBLE_DEVICES
cd /net/scratch2/chenghao/annealing_sampling
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./env
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
data=numina_math
project_name="minimal_rl_numina_math"
algorithm=grpo
model=Qwen2.5-Math-1.5B
model_name_or_path=Qwen/$model
n=4
#experiment_name=${model}-${algorithm}-${data}-n${n}
experiment_name="initial_grpo_baseline"
GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
    trainer.test_freq=10 \
    trainer.total_epochs=1 2>&1 | tee logs/${project_name}/${experiment_name}.log
