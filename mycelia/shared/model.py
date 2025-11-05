# llm_weightnet/shared/model.py
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional, Tuple

import requests
import torch
from torch import nn

from mycelia.shared.config import MinerConfig
from mycelia.shared.app_logging import structlog  
from mycelia.shared.expert_manager import ExpertManager, create_expert_groups 
from mycelia.shared.modeling.modeling_mycelia import get_base_model  , partial_moe
from mycelia.shared import chain
from mycelia.shared.checkpoint import (
    get_resume_info,
    load_checkpoint,
)
from mycelia.shared.helper import *

logger = structlog.get_logger(__name__)

def _default_model(rank: int, config: MinerConfig) -> nn.Module:
    resume = False   
    start_step = 0
    latest_checkpoint_path = None
    if get_nested_attr(config,"ckpt.resume_from_ckpt", False):
        resume, start_step, latest_checkpoint_path = get_resume_info(rank, config)

    em = ExpertManager(
        model=get_base_model(config, noise= True),
        num_experts=config.moe.num_experts,
        num_worker_groups=config.moe.num_worker_groups,
    )
    em.compute_group_assignments(seed=start_step if config.moe.rotate_expert else 0)

    model = get_base_model(config, noise=start_step == 0, expert_group_assignment=em.expert_group_assignment).to(config.model.device)

    if get_nested_attr(config,"moe.partial_moe", False):
        model = partial_moe(config, model, config.moe.my_expert_group_id, em.expert_group_assignment)

    if get_nested_attr(config,"ckpt.resume_from_ckpt", False) and resume and latest_checkpoint_path:
        logger.info("rank %s setup training: resuming from %s (start_step=%s)", rank, latest_checkpoint_path, start_step)
        load_checkpoint(config=config, checkpoint_path=latest_checkpoint_path, model=model, rank=rank, device=config.model.device)

    model = model.to(config.model.device)

    model.gradient_checkpointing_enable()
    
    return model, em


# TODO: fill function
def _fetch_validator_endpoint_from_chain(round_hint: Optional[str] = None) -> Optional[str]:
    """
    Ask chain for the *current* validator node endpoint (e.g., https://validator-1:8080).
    You implement this inside shared/blockchain.py
    """
    try:
        info = chain.get_active_validator_info()  # must return {"api_base": "...", ...}
        endpoint = (info or {}).get("api_base")
        return endpoint
    except Exception as e:
        logger.warning("model.chain_lookup_failed", error=str(e))
        return None


def load_base_model(rank: int, config: Optional[Config] = None, round_hint: Optional[str] = None) -> Tuple[nn.Module, ExpertManager]:
    """
    Main entry point used by miners (and potentially validator itself).
    1) Ask the chain for an active validator endpoint.
    2) If available, ping and fetch current model.
    3) Else, initialize a default model.
    """
    config = config or MinerConfig()
    api_base = _fetch_validator_endpoint_from_chain(round_hint)

    if api_base:
        pass

    return _default_model(rank = rank, config = config)

# --- Helper: export/save current model to artifact (used by validator) ---
# TODO: fill function
def export_model_artifact(model: nn.Module) -> Tuple[bytes, str]:
    """
    Serialize the model to bytes + return format string.
    Default: state_dict via torch.save
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read(), "state_dict"
