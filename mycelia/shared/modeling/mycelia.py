from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from mycelia.shared.app_logging import structlog
from mycelia.shared.config import MinerConfig, ValidatorConfig
from mycelia.shared.expert_manager import ExpertManager
from mycelia.shared.helper import *
from mycelia.shared.modeling.custom_qwen3_next import (
    CustomQwen3NextModel,
    get_moe_model_config,
)


logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
def get_base_model(
    config: MinerConfig | ValidatorConfig,
    expert_manager: ExpertManager,
    group_ids: List | None = None, 
    state_dicts: List = [],
    partial = False
) -> Optional[nn.Module]:
    """
    Load a base Causal LM by `config.model.model_path` and optionally convert to MoE.

    Returns
    -------
    Optional[nn.Module]
        A Hugging Face causal LM (LLaMA or OLMo), possibly converted to OLMoE.
    """
    topk = config.moe.partial_topk if partial else config.moe.full_topk
    moe_config = get_moe_model_config(config, topk, group_ids, expert_manager)
    model = CustomQwen3NextModel(moe_config)

    if model is not None and get_nested_attr(config, "model.torch_compile", False):
        model = torch.compile(model)

    if len(state_dicts) > 0: 
        merged_stated_dict, missing = merge_state_dicts_with_priority(state_dicts, model)
        assert len(missing) == 0 
        model.load_state_dict(merged_stated_dict, strict=True)  # partial by design
    
    return model


def get_base_tokenizer(config: MinerConfig | ValidatorConfig):
    """
    Load the tokenizer for `config.model.model_path`.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, use_fast=True)
    # tokenizer.pad_token = "</s>"
    return tokenizer


from collections import OrderedDict
import torch
from typing import List, Dict, Tuple, Optional

def merge_state_dicts_with_priority(
    state_dicts: List[Dict[str, torch.Tensor]],
    model: Optional[torch.nn.Module] = None,
) -> Tuple[OrderedDict, Optional[List[str]]]:
    """
    Merge a list of state_dicts where earlier dicts have *higher* priority.
    Unexpected keys (not present in the model) are removed automatically.

    Args:
        state_dicts: list of state dicts, in priority order.
                     state_dicts[0] has highest priority, state_dicts[-1] lowest.
        model: optional model, used to filter out unexpected keys 
               and check for missing keys.

    Returns:
        merged_state_dict: OrderedDict with cleaned + merged parameters.
        missing_keys: keys that the model expects but are not in merged
    """
    if not state_dicts:
        raise ValueError("state_dicts must be a non-empty list")

    merged = OrderedDict()

    # Build merged dict: earlier dicts override later ones.
    for sd in reversed(state_dicts):
        for k, v in sd.items():
            if k not in merged:
                merged[k] = v

    # If no model provided, return as is
    if model is None:
        return merged, None

    # Filter out unexpected keys
    model_keys = set(model.state_dict().keys())
    cleaned = OrderedDict((k, v) for k, v in merged.items() if k in model_keys)

    # Compute missing keys
    missing_keys = sorted(model_keys - set(cleaned.keys()))

    return cleaned, missing_keys
