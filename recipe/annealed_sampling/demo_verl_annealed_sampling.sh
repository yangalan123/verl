#!/bin/bash

# make sure to run verl/examples/data_preprocess/gsm8k.py to get the gsm8k dataset first
decays=("25" "50" "100")
start_temps=("1.0" "1.2" "0.6")
end_temps=("0.1" "0.0")

# Calculate total number of combinations
total_decays=${#decays[@]}
total_start_temps=${#start_temps[@]}
total_end_temps=${#end_temps[@]}
total_combinations=$((total_decays * total_start_temps * total_end_temps))

# Use SLURM_ARRAY_TASK_ID to index the specific combination
task_id=${SLURM_ARRAY_TASK_ID:-0}

# Calculate indices for each parameter
decay_idx=$((task_id % total_decays))
start_temp_idx=$(((task_id / total_decays) % total_start_temps))
end_temp_idx=$((task_id / (total_decays * total_start_temps)))

# Extract the specific values
decay_freq=${decays[$decay_idx]}
start_temp=${start_temps[$start_temp_idx]}
end_temp=${end_temps[$end_temp_idx]}

echo "Task ID: $task_id"
echo "Combination: decay_freq=$decay_freq, start_temp=$start_temp, end_temp=$end_temp"

unset ROCR_VISIBLE_DEVICES
PROJECT_NAME="demo_verl_gsm8k_qwen2.5_1.5b_instruct"
RUN_NAME="annealed_sampling_grpo_lower_lr_explore_${start_temp}_stable_${end_temp}_decay_freq_${decay_freq}_zzk"

PYTHONUNBUFFERED=1 VLLM_USE_V1=0 python3 -m verl.trainer.main_ppo \
 data.train_files=./dataset/gsm8k/train.parquet \
 data.val_files=./dataset/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.annealed_sampling.exploration_temp=${start_temp} \
 actor_rollout_ref.rollout.annealed_sampling.stability_temp=${end_temp} \
 actor_rollout_ref.rollout.annealed_sampling.decay_freq=${decay_freq} \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.actor.clip_ratio=0.2 \
 actor_rollout_ref.rollout.n=16 \
 algorithm.adv_estimator="grpo" \
 critic.optim.lr=1.2e-6 \
 critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console','wandb'] \
 trainer.project_name=$PROJECT_NAME \
 trainer.experiment_name=$RUN_NAME \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1
