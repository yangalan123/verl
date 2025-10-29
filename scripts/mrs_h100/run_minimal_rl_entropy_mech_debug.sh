#!/bin/bash

#SBATCH --chdir=/fsx/zhuokai/verl/
#SBATCH --gres=gpu:8
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=debug_entropy_mech_no_filter_1_2_qwen_2_5_math_1_5b
#SBATCH --output=/fsx/zhuokai/verl/slurm/debug_entropy_mech_no_filter_1_2_qwen_2_5_math_1_5b.stdout
#SBATCH --error=/fsx/zhuokai/verl/slurm/debug_entropy_mech_no_filter_1_2_qwen_2_5_math_1_5b.stderr


# export VLLM_ATTENTION_BACKEND=XFORMERS
data=numina_math
project_name="minimal_rl_numina_math"
algorithm=grpo
model=Qwen2.5-Math-1.5B
model_name_or_path=Qwen/$model
# model=Llama-3.2-1B-Instruct
# model_name_or_path=meta-llama/$model
# model=OctoThinker-1B-Hybrid-Base
# model_name_or_path=OctoThinker/$model
tokenizer=meta-llama/Llama-3.2-1B-Instruct
rollout_n=4
# for mean@K computation
k_max=16
#experiment_name=${model}-${algorithm}-${data}-n${n}
# [TODO for Zhuokai]: change the temperature to other values
temperature=1.2
experiment_name="debug_entropy_mech_no_filter_${temperature}_${model}_zzk"
GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}
ROOT_DIR=/fsx/zhuokai/verl/

math_train_path=$ROOT_DIR/data/$data/train.parquet
math_test_path=$ROOT_DIR/data/math500/test.parquet
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=1
clip_ratio_high=1
clip_cov_ratio=0.0002
clip_cov_lb=1.0
clip_cov_ub=5.0
top_p=1.0
ppo_kl_coef=1
kl_cov_ratio=0.2
loss_agg_mode="token-mean"
# or "kl_cov"
loss_mode="clip_cov"

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
    data.tokenizer=$tokenizer \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.policy_loss.clip_cov_ratio=${clip_cov_ratio} \
    actor_rollout_ref.actor.policy_loss.clip_cov_lb=${clip_cov_lb} \
    actor_rollout_ref.actor.policy_loss.clip_cov_ub=${clip_cov_ub} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
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
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
    trainer.test_freq=10 \
    trainer.total_epochs=1 2>&1 | tee logs/${project_name}/${experiment_name}.log
