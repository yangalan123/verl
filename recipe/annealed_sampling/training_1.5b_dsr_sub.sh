#!/bin/bash
#SBATCH --mail-user=chenghao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch2/chenghao/annealing_sampling/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch2/chenghao/annealing_sampling/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch2/chenghao/annealing_sampling/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=600gb
#SBATCH --job-name=rlvr_oneshot_annealed_sampling
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=11:59:00
#SBATCH --signal=SIGUSR1@120
set -x

# CHECKPOINTS_DIR=... # TODO: change to your own path

#export VLLM_ATTENTION_BACKEND=XFORMERS
ROOT_DIR=/net/scratch2/chenghao/annealing_sampling/One-Shot-RLVR
unset ROCR_VISIBLE_DEVICES
cd /net/scratch2/chenghao/annealing_sampling
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./env
PROJECT_NAME="oneshort_rlvr_dsrsub_recipe_qwen2.5_math_1.5b"
RUN_NAME="initial_grpo_baseline"

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=$ROOT_DIR/data/train/one_shot_rlvr/dsr_sub.parquet \
 data.val_files=$ROOT_DIR/data/test/math500.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=530 \
 data.max_prompt_length=1024 \
 data.max_response_length=3072 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path='Qwen/Qwen2.5-Math-1.5B' \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=0.6 \
 actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name=$PROJECT_NAME \
 trainer.experiment_name=$RUN_NAME \
 trainer.val_before_train=True \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=20 \
 trainer.test_freq=20 \
 trainer.default_hdfs_dir=null \
 trainer.total_epochs=2000 2>&1 | tee verl_demo.log
