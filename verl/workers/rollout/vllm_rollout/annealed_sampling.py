import numpy as np
import torch
from typing import Union


def annealed_sampling_processor(token_ids: Union[list[int], tuple[int]], logits: torch.Tensor, 
                               exploration_temp: float = 1.0, stability_temp: float = 0.1, 
                               decay_freq: int = 50, global_step: int = 0) -> torch.Tensor:
    """
    Annealed sampling logits processor for vLLM.
    
    Args:
        token_ids: List of token IDs generated so far
        logits: Logits tensor from the model
        exploration_temp: Exploration temperature (higher = more exploration)
        stability_temp: Stability temperature (lower = more focused)
        decay_freq: Decay frequency for temperature annealing
        global_step: Current global optimization step
        
    Returns:
        Modified logits tensor
    """
    # Calculate the current temperature based on global step instead of sequence length
    _exploration_temp = exploration_temp * np.exp(-global_step / decay_freq)
    current_temp = stability_temp + (_exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * decay_freq))
    
    # Apply temperature scaling to logits
    logits = logits / current_temp
    
    return logits