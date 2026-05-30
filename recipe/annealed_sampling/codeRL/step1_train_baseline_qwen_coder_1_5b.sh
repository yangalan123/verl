#!/bin/bash
# Fixed-temperature GRPO baseline on Qwen2.5-Coder-1.5B-Instruct for code RL.
# Usage:
#   bash recipe/annealed_sampling/codeRL/step1_train_baseline_qwen_coder_1_5b.sh [TEMP]
# Default TEMP is 1.0. Common sweep: 0.7, 1.0, 1.2.
#
# Requires the parquets produced by step0_data_preprocess.sh.
# No Docker / firejail -- the prime_code & humanevalplus rewards both fork
# Python subprocesses with SIGALRM timeouts.

set -euxo pipefail

# -- Cluster scaffolding (edit for your slurm setup) --
# unset ROCR_VISIBLE_DEVICES
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <your_env>

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

# -- Run-level config --
TEMP="${1:-1.0}"
ROOT_DIR="${ROOT_DIR:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-4}"

project_name="codeRL_eurus2_qwen_coder_1_5b"
experiment_name="baseline_fixed_temp_${TEMP}"
algorithm="grpo"
rollout_n=4
k_max=8
save_freq=10
test_freq=10
# Quick-and-dirty rebuttal runs: cap at TOTAL_TRAINING_STEPS steps (default 100)
# to reveal the EAD-vs-baseline trend without a full epoch. Set -1 for a full epoch.
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"

train_files="['${DATA_ROOT}/eurus2_code/train.parquet']"
val_files="['${DATA_ROOT}/livecodebench/release_v2_test.parquet','${DATA_ROOT}/humanevalplus/test.parquet']"

log_dir="${ROOT_DIR}/logs/${project_name}/${experiment_name}"
mkdir -p "${log_dir}"

PYTHONUNBUFFERED=1 VLLM_USE_V1=0 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${algorithm} \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=512 \
    data.max_prompt_length=1536 \
    data.max_response_length=2560 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_NAME_OR_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.temperature=${TEMP} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${TEMP} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NUM_GPU_PER_NODE} \
    trainer.rollout_data_dir=${log_dir}/rollout_data \
    trainer.validation_data_dir=${log_dir}/validation_data \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
    trainer.test_freq=${test_freq} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.total_epochs=1 2>&1 | tee ${log_dir}/run.log
