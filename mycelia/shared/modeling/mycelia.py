from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from mycelia.shared.app_logging import structlog
from mycelia.shared.config import MinerConfig, ValidatorConfig
from mycelia.shared.expert_manager import ExpertManager
from mycelia.shared.helper import *
from mycelia.shared.modeling.custom_qwen3_next import (
    CustomQwen3NextForCausalLM,
    get_moe_model_config,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
def get_base_model(
    config: MinerConfig | ValidatorConfig,
    expert_manager: ExpertManager,
    group_ids: list | None = None,
    state_dicts: list = [],
    partial=False,
) -> nn.Module | None:
    """
    Load base model with role-specific optimizations.

    Validators: Load with 4-bit quantization + Unsloth for memory efficiency
    Miners: Load standard model for training
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    topk = config.moe.partial_topk if partial else config.moe.full_topk
    moe_config = get_moe_model_config(config, topk, group_ids, expert_manager)

    is_validator = config.role == "validator"
    use_quantization = get_nested_attr(config, "model.use_quantization", False) and is_validator
    use_unsloth = get_nested_attr(config, "model.use_unsloth", False) and is_validator

    # === QUANTIZED PATH (Validators only) ===
    if use_quantization:
        logger.info("Loading with 4-bit quantization for validator")

        # Try Unsloth first (fastest)
        if use_unsloth:
            try:
                from unsloth import FastLanguageModel

                model, _ = FastLanguageModel.from_pretrained(
                    model_name=config.model.model_path,
                    max_seq_length=moe_config.max_position_embeddings,
                    dtype=torch.float16,
                    load_in_4bit=True,
                    device_map="auto",
                )
                FastLanguageModel.for_inference(model)
                logger.info("✓ Loaded with Unsloth optimizations")
                return model
            except Exception as e:
                logger.warning(f"Unsloth failed, falling back to BitsAndBytes: {e}")

        # Fallback to BitsAndBytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        max_memory = get_nested_attr(config, "model.max_memory", None)
        if max_memory is None:
            max_memory = {0: "46GB", "cpu": "100GB"}

        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_path,
            config=moe_config,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        logger.info("✓ Loaded with BitsAndBytes quantization")
        return model

    # === STANDARD PATH (Miners) ===
    model = CustomQwen3NextForCausalLM(moe_config)

    if len(state_dicts) > 0:
        merged_stated_dict, missing = merge_state_dicts_with_priority(state_dicts, model)
        assert len(missing) == 0
        model.load_state_dict(merged_stated_dict, strict=True)

    if model is not None and get_nested_attr(config, "model.torch_compile", False):
        model = torch.compile(model)

    return model


def get_base_tokenizer(config: MinerConfig | ValidatorConfig):
    """
    Load the tokenizer for `config.model.model_path`.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, use_fast=True)
    # tokenizer.pad_token = "</s>"
    return tokenizer


def merge_state_dicts_with_priority(
    state_dicts: list[dict[str, torch.Tensor]],
    model: torch.nn.Module | None = None,
) -> tuple[OrderedDict, list[str] | None]:
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
