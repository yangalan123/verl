import numpy as np
import torch
from typing import Union, Dict, Optional
import pickle
import os
from collections import defaultdict
try:
    from vllm.v1.sample.logits_processor import (
        AdapterLogitsProcessor, # Wrapper base-class
        RequestLogitsProcessor, # Request-level logitsproc type annotation
    )
except ImportError as e:
    print(f"Warning: vllm.v1.sample.logits_processor is not installed, using v0 API instead")
    AdapterLogitsProcessor = None
    RequestLogitsProcessor = None
    if "VLLM_USE_V1" not in os.environ:
        os.environ["VLLM_USE_V1"] = "0"
from vllm import SamplingParams


class HistoricalDataManager:
    """
    Manages historical data for adaptive annealing based on previous rollout performance.
    
    Stores:
    - Last-round rollout token lengths L(x) for each prompt
    - Last-round advantages for each trajectory  
    - Current round log probabilities (for potential future use)
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size: int = 10000):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.history_cache: Dict[str, Dict] = defaultdict(dict)
        
        # Load existing cache if available
        if cache_dir and os.path.exists(cache_dir):
            self._load_cache()
    
    def _get_cache_file(self) -> str:
        """Get the cache file path."""
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, "annealing_history.pkl")
    
    def _load_cache(self):
        """Load historical data from disk."""
        cache_file = self._get_cache_file()
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.history_cache = pickle.load(f)
                print(f"Loaded {len(self.history_cache)} historical entries from {cache_file}")
            except Exception as e:
                print(f"Warning: Could not load historical cache: {e}")
    
    def _save_cache(self):
        """Save historical data to disk."""
        cache_file = self._get_cache_file()
        if cache_file:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.history_cache, f)
            except Exception as e:
                print(f"Warning: Could not save historical cache: {e}")
    
    def update_history(self, uids: np.ndarray, token_lengths: np.ndarray, 
                      advantages: Optional[np.ndarray] = None, 
                      log_probs: Optional[np.ndarray] = None):
        """
        Update historical data with new rollout information.
        
        Args:
            uids: Array of unique identifiers for each trajectory
            token_lengths: Array of response token lengths for each trajectory
            advantages: Optional array of advantages for each trajectory
            log_probs: Optional array of log probabilities for each trajectory
        """
        for i, uid in enumerate(uids):
            uid_str = str(uid)
            
            # Store token length (this is the key metric for annealing)
            self.history_cache[uid_str]['last_token_length'] = int(token_lengths[i])
            
            # Store additional data if provided
            if advantages is not None:
                self.history_cache[uid_str]['last_advantage'] = float(advantages[i])
            if log_probs is not None:
                # Store mean log prob for the sequence
                self.history_cache[uid_str]['last_log_prob'] = float(log_probs[i])
            
            # Add timestamp for potential cleanup
            self.history_cache[uid_str]['last_updated'] = len(self.history_cache)
        
        # Cleanup old entries if cache is too large
        if len(self.history_cache) > self.max_cache_size:
            self._cleanup_cache()
        
        # Save to disk periodically
        if len(self.history_cache) % 1000 == 0:
            self._save_cache()
    
    def get_last_token_length(self, uid: str, default_length: int = 50) -> int:
        """
        Get the last rollout token length for a given UID.
        
        Args:
            uid: Unique identifier for the trajectory
            default_length: Default length to return if no history exists
            
        Returns:
            The last token length, or default_length if no history exists
        """
        uid_str = str(uid)
        if uid_str in self.history_cache:
            return self.history_cache[uid_str].get('last_token_length', default_length)
        return default_length
    
    def get_last_advantage(self, uid: str) -> Optional[float]:
        """Get the last advantage for a given UID."""
        uid_str = str(uid)
        if uid_str in self.history_cache:
            return self.history_cache[uid_str].get('last_advantage', None)
        return None
    
    def _cleanup_cache(self):
        """Remove oldest entries to keep cache size manageable."""
        if len(self.history_cache) <= self.max_cache_size:
            return
        
        # Sort by last_updated and keep only the most recent entries
        sorted_items = sorted(self.history_cache.items(), 
                            key=lambda x: x[1].get('last_updated', 0))
        
        # Keep the most recent entries
        items_to_keep = sorted_items[-self.max_cache_size:]
        self.history_cache = dict(items_to_keep)


# Global instance for easy access
_historical_manager = None

def get_historical_manager(cache_dir: Optional[str] = None) -> HistoricalDataManager:
    """Get or create the global historical data manager."""
    global _historical_manager
    if _historical_manager is None:
        _historical_manager = HistoricalDataManager(cache_dir=cache_dir)
    return _historical_manager

# Example Usage for LogitsProcessor in V1 API (waiting to check whether this implementation is correct in InferenceTime Scaling Experiment):
# llm = LLM(
            #     model=args.model_name_or_path,
            #     tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            #     pipeline_parallel_size=args.pipeline_parallel_size,
            #     trust_remote_code=True,
            #     gpu_memory_utilization=args.gpu_memory_utilization,
            #     max_logprobs=100,
            #     logits_processors=[WrapperAdapterLogitsProcessor]
            # )
# sampling_param = SamplingParams(
        #     temperature=args.temperature,
        #     top_p=args.top_p,
        #     max_tokens=args.max_tokens_per_call,
        #     logprobs=50,
        #     n=1,
        #     stop=stop_words,
        #     stop_token_ids=(
        #         [151645, 151643] if "qwen2" in args.model_name_or_path.lower() else None
        #     ),
        #     extra_args={
        #         "exploration_temp": args.annealed_sampling_exploration_temp,
        #         "stability_temp": args.annealed_sampling_stability_temp,
        #         "decay_freq": args.annealed_sampling_decay_freq,
        #         "global_step": args.annealed_sampling_global_step,
        #         "decay_mode": args.annealed_sampling_decay_mode,
        #         "warmup_period": args.annealed_sampling_warmup_period,
        #         "decay_freq_increase_factor": args.annealed_sampling_decay_freq_increase_factor,
        #     },
        # )
# then, use normal llm.generate(..., sampling_param=sampling_param)

class AnnealedSamplingProcessor:
    def __init__(self, exploration_temp: float = 1.0, stability_temp: float = 0.1, decay_freq: int = 50, global_step: int = 0, decay_mode: str = "both", warmup_period: int = 10, decay_freq_increase_factor: int = 5, decay_freq_cap_small: int = 2000, decay_freq_cap_large: int = 40000):
        self.exploration_temp = exploration_temp
        self.stability_temp = stability_temp
        self.decay_freq = decay_freq
        self.global_step = global_step
        self.decay_mode = decay_mode
        self.warmup_period = warmup_period
        self.decay_freq_increase_factor = decay_freq_increase_factor
        self.decay_freq_cap_small = decay_freq_cap_small
        self.decay_freq_cap_large = decay_freq_cap_large
    
    def __call__(self, token_ids: Union[list[int], tuple[int]], logits: torch.Tensor) -> torch.Tensor:
        return annealed_sampling_processor(token_ids, logits, 
            exploration_temp=self.exploration_temp,
            stability_temp=self.stability_temp,
            decay_freq=self.decay_freq,
            global_step=self.global_step,
            decay_mode=self.decay_mode,
            warmup_period=self.warmup_period,
            decay_freq_increase_factor=self.decay_freq_increase_factor,
            decay_freq_cap_small=self.decay_freq_cap_small,
            decay_freq_cap_large=self.decay_freq_cap_large
        )

if AdapterLogitsProcessor is not None and RequestLogitsProcessor is not None:
    class WrapperAdapterLogitsProcessor(AdapterLogitsProcessor):
        def is_argmax_invariant(self) -> bool:
            return True
        
        def new_req_logits_processor(self, params: SamplingParams) -> Optional[RequestLogitsProcessor]:
            exploration_temp = params.extra_args and params.extra_args.get("exploration_temp")
            if exploration_temp is not None:
                stability_temp = params.extra_args and params.extra_args.get("stability_temp")
                decay_freq = params.extra_args and params.extra_args.get("decay_freq")
                global_step = params.extra_args and params.extra_args.get("global_step")
                decay_mode = params.extra_args and params.extra_args.get("decay_mode")
                warmup_period = params.extra_args and params.extra_args.get("warmup_period")
                decay_freq_increase_factor = params.extra_args and params.extra_args.get("decay_freq_increase_factor")
                decay_freq_cap_small = params.extra_args and params.extra_args.get("decay_freq_cap_small", 2000)
                decay_freq_cap_large = params.extra_args and params.extra_args.get("decay_freq_cap_large", 40000)
                # historical_manager is not supported in v1 API -- at least we do not want to make such a giant object as an extra_arg
                # [TODO] we should find a way to support it, so we can move to EAD v2, one way is to override __init__ of WrapperAdapterLogitsProcessor
                return AnnealedSamplingProcessor(
                    exploration_temp=exploration_temp,
                    stability_temp=stability_temp,
                    decay_freq=decay_freq,
                    global_step=global_step,
                    decay_mode=decay_mode,
                    warmup_period=warmup_period,
                    decay_freq_increase_factor=decay_freq_increase_factor,
                    decay_freq_cap_small=decay_freq_cap_small,
                    decay_freq_cap_large=decay_freq_cap_large
                )
            return None



def annealed_sampling_processor(token_ids: Union[list[int], tuple[int]], logits: torch.Tensor, 
                               exploration_temp: float = 1.0, stability_temp: float = 0.1, 
                               decay_freq: int = 50, global_step: int = 0,
                               decay_mode: str = 'both', warmup_period: int = 10,
                               adaptive_decay: bool = False, uid: Optional[str] = None,
                               historical_manager: Optional[HistoricalDataManager] = None,
                               decay_freq_increase_factor: int = 5,
                               decay_freq_cap_small: int = 2000, decay_freq_cap_large: int = 40000) -> torch.Tensor:
    """
    Annealed sampling logits processor for vLLM.
    
    Args:
        token_ids: List of token IDs generated so far
        logits: Logits tensor from the model
        exploration_temp: Exploration temperature (higher = more exploration)
        stability_temp: Stability temperature (lower = more focused)
        decay_freq: Decay frequency for temperature annealing
        global_step: Current global optimization step
        decay_mode: Which annealing mode to use. Options: 'global_step', 'token_length', 'both', 'none', 'adaptive'.
        warmup_period: If len(token_ids) < warmup_period, do not apply temperature scaling (default: 10)
        adaptive_decay: Whether to use adaptive decay based on historical performance
        uid: Unique identifier for the current trajectory (required for adaptive decay)
        historical_manager: Historical data manager instance (optional, will use global if None)
        decay_freq_increase_factor: Factor by which decay_freq increases with global_step (default: 5)
        decay_freq_cap_small: Cap value for decay_freq in 'both_v_1_5' and 'both_v_1_5_rev' modes (default: 2000)
        decay_freq_cap_large: Cap value for decay_freq in 'negexp' and 'negexp_rev' modes (default: 40000)
    Returns:
        Modified logits tensor
    """
    # If in warmup period, do not apply temperature scaling
    if len(token_ids) < warmup_period:
        return logits
    
    # Calculate the current temperature based on the selected decay mode
    if decay_mode == 'global_step':
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-global_step / decay_freq)
    elif decay_mode == 'token_length':
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * decay_freq))
    elif decay_mode == 'both':
        _exploration_temp = exploration_temp * np.exp(-global_step / decay_freq)
        current_temp = stability_temp + (_exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * decay_freq))
    elif decay_mode == "both_v_1_5":
        _decay_freq = min(decay_freq + decay_freq_increase_factor * global_step, decay_freq_cap_small)
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * _decay_freq))
    elif decay_mode == "both_v_1_5_rev":
        _decay_freq = min(decay_freq + decay_freq_increase_factor * global_step, decay_freq_cap_small)
        current_temp = exploration_temp + (stability_temp - exploration_temp) * np.exp(-len(token_ids) / (20 * _decay_freq))
    elif decay_mode == "negexp":
        # as we use -exp(x/d), we need to use a larger decay_freq to get a smaller temperature and to keep the temperature >= 0
        _decay_freq = min(decay_freq + decay_freq_increase_factor * global_step, decay_freq_cap_large)
        current_temp = 1 + exploration_temp - np.exp(len(token_ids) / (20 * _decay_freq))
        # avoid temperature < stability_temp
        current_temp = max(current_temp, stability_temp)
    elif decay_mode == "negexp_rev":
        _decay_freq = min(decay_freq + decay_freq_increase_factor * global_step, decay_freq_cap_large)
        if len(token_ids) / (20 * _decay_freq) < np.log(exploration_temp - stability_temp):
            current_temp = stability_temp + np.exp(len(token_ids) / (20 * _decay_freq))
        else:
            current_temp = exploration_temp
        # current_temp = min(current_temp, exploration_temp)
    elif decay_mode == "steps_variant":
        # use the same temperature at all positions, no matter how long token_ids is
        # the temperature gradually increases from stability_temp to exploration_temp
        current_temp = exploration_temp + (stability_temp - exploration_temp) * np.exp(-global_step / decay_freq)
        current_temp = min(current_temp, 1.0)
    elif decay_mode == "steps_variant_rev":
        # use the same temperature at all positions, no matter how long token_ids is
        # the temperature gradually increases from stability_temp to exploration_temp
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-global_step / decay_freq)
        current_temp = max(current_temp, 0.1)
    elif decay_mode == 'adaptive':
        # New adaptive decay mode based on historical performance
        if adaptive_decay and uid is not None:
            # Get historical manager
            if historical_manager is None:
                historical_manager = get_historical_manager()
            
            # Get last token length for this UID
            last_token_length = historical_manager.get_last_token_length(uid, default_length=50)
            
            # Use adaptive decay: exp(-len(token_ids) / (0.1 * L(x)))
            adaptive_decay_rate = len(token_ids) / (0.1 * last_token_length)
            current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-adaptive_decay_rate)
        else:
            # Fallback to standard token_length mode
            current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * decay_freq))
    elif decay_mode == 'none':
        current_temp = exploration_temp
    else:
        raise ValueError(f"Unknown decay_mode: {decay_mode}")

    # Apply temperature scaling to logits
    logits = logits / current_temp
    
    return logits