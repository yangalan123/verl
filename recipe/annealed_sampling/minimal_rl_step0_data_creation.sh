#!/bin/bash

# credit to Xiong Wei at UIUC: https://github.com/RLHFlow/Minimal-RL/tree/main

conda activate [path_to_local_verl_env]

python minimal_rl_data_preprocess/math_dataset.py --local_dir [path_to_local_math_data]
python minimal_rl_data_preprocess/numina_math.py --local_dir [path_to_local_numina_math_data]

# then, checkout both minimal_rl_grpo.sh (for GRPO baseline) and annealed_sampling_minimal_rl.sh (for annealed sampling)
