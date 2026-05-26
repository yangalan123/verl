#!/bin/bash
# Ablation: sweep d_max (decay_freq_cap_large) for EAD on Qwen2.5-Math-1.5B.
# Addresses the COLM reviewer concern about overlooked sensitivity to d_max.
#
# Usage:
#   bash recipe/annealed_sampling/ablation_d_max_sweep_qwen_math_1_5b.sh
#
# Runs five values of d_max: 200, 1000, 5000, 40000 (paper default), 200000.
# Each run launches sequentially. Total cost ~10 h x 5 on 4xA100.

set -euxo pipefail

# -- Cluster scaffolding (edit for your slurm setup) --
# unset ROCR_VISIBLE_DEVICES
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <your_env>

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Math-1.5B}"
NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-4}"

project_name="ablation_d_max_qwen_math_1_5b"
algorithm="grpo"
rollout_n=4
k_max=16
save_freq=10
test_freq=10
warmup_period=10
start_temp=1.2
end_temp=0.1
decay_freq=200
strategy="negexp"
decay_freq_increase_factor=5

train_files="['${DATA_ROOT}/numina_math/train.parquet']"
val_files="['${DATA_ROOT}/math500/test.parquet']"

for D_MAX in 200 1000 5000 40000 200000; do
    experiment_name="ead_negexp_explore_${start_temp}_stable_${end_temp}_d0_${decay_freq}_alpha_${decay_freq_increase_factor}_dmax_${D_MAX}"
    log_dir="${ROOT_DIR}/logs/${project_name}/${experiment_name}"
    mkdir -p "${log_dir}"

    PYTHONUNBUFFERED=1 VLLM_USE_V1=0 python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=${algorithm} \
        data.train_files="${train_files}" \
        data.val_files="${val_files}" \
        data.train_batch_size=1024 \
        data.max_prompt_length=1024 \
        data.max_response_length=3072 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=${MODEL_NAME_OR_PATH} \
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
        actor_rollout_ref.rollout.annealed_sampling.enable=True \
        actor_rollout_ref.rollout.annealed_sampling.decay_mode=${strategy} \
        actor_rollout_ref.rollout.annealed_sampling.exploration_temp=${start_temp} \
        actor_rollout_ref.rollout.annealed_sampling.stability_temp=${end_temp} \
        actor_rollout_ref.rollout.annealed_sampling.decay_freq=${decay_freq} \
        actor_rollout_ref.rollout.annealed_sampling.warmup_period=${warmup_period} \
        actor_rollout_ref.rollout.annealed_sampling.decay_freq_increase_factor=${decay_freq_increase_factor} \
        actor_rollout_ref.rollout.annealed_sampling.decay_freq_cap_large=${D_MAX} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=${rollout_n} \
        actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.0 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=${project_name} \
        trainer.experiment_name=${experiment_name} \
        trainer.n_gpus_per_node=${NUM_GPU_PER_NODE} \
        trainer.rollout_data_dir=${log_dir}/rollout_data \
        trainer.validation_data_dir=${log_dir}/validation_data \
        trainer.max_actor_ckpt_to_keep=3 \
        trainer.max_critic_ckpt_to_keep=3 \
        trainer.val_before_train=True \
        trainer.nnodes=1 \
        trainer.save_freq=${save_freq} \
        trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
        trainer.test_freq=${test_freq} \
        trainer.total_epochs=1 2>&1 | tee ${log_dir}/run.log
done
