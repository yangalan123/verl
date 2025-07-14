import numpy as np
import torch
from typing import Union


def annealed_sampling_processor(token_ids: Union[list[int], tuple[int]], logits: torch.Tensor, 
                               exploration_temp: float = 1.0, stability_temp: float = 0.1, 
                               decay_freq: int = 50, global_step: int = 0,
                               decay_mode: str = 'both') -> torch.Tensor:
    """
    Annealed sampling logits processor for vLLM.
    
    Args:
        token_ids: List of token IDs generated so far
        logits: Logits tensor from the model
        exploration_temp: Exploration temperature (higher = more exploration)
        stability_temp: Stability temperature (lower = more focused)
        decay_freq: Decay frequency for temperature annealing
        global_step: Current global optimization step
        decay_mode: Which annealing mode to use. Options: 'global_step', 'token_length', 'both', 'none'.
        
    Returns:
        Modified logits tensor
    """
    # Calculate the current temperature based on the selected decay mode
    if decay_mode == 'global_step':
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-global_step / decay_freq)
    elif decay_mode == 'token_length':
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * decay_freq))
    elif decay_mode == 'both':
        _exploration_temp = exploration_temp * np.exp(-global_step / decay_freq)
        current_temp = stability_temp + (_exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * decay_freq))
    elif decay_mode == 'none':
        current_temp = exploration_temp
    else:
        raise ValueError(f"Unknown decay_mode: {decay_mode}")

    # Apply temperature scaling to logits
    logits = logits / current_temp
    
    return logits