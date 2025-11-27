from __future__ import annotations

from typing import Optional, Tuple

from torch import nn

from mycelia.shared import chain
from mycelia.shared.app_logging import structlog
from mycelia.shared.checkpoint import load_checkpoint, start_model_from
from mycelia.shared.config import MinerConfig, ValidatorConfig
from mycelia.shared.expert_manager import ExpertManager
from mycelia.shared.helper import *
from mycelia.shared.modeling.mycelia import get_base_model

logger = structlog.get_logger(__name__)


def get_model_from_checkpoint(rank: int, config: MinerConfig | ValidatorConfig, expert_manager: ExpertManager) -> Tuple[nn.Module, dict]:
    resume = False
    miner_version = 0 
    validator_version = 0
    latest_checkpoint_path = None
    if get_nested_attr(config, "ckpt.resume_from_ckpt", False):
        resume, model_version, latest_checkpoint_path = start_model_from(rank, config)
        
    model = get_base_model(
        config, 
        expert_manager=expert_manager, 
        group_ids = [config.moe.my_expert_group_id] if config.role == 'miner' else None,
        partial=(config.role == 'miner')
    ).to(config.model.device)

    if get_nested_attr(config, "ckpt.resume_from_ckpt", False) and resume and latest_checkpoint_path:
        load_checkpoint(
            config=config, checkpoint_path=latest_checkpoint_path, model=model, rank=rank, device=config.model.device
        )

    model = model.to(config.model.device)

    model.gradient_checkpointing_enable()

    return model, model_version


# TODO: fill function
def fetch_validator_endpoint_from_chain(round_hint: Optional[str] = None) -> Optional[str]:
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


def load_model(
    rank: int, config: MinerConfig | ValidatorConfig, expert_manager: ExpertManager, round_hint: Optional[str] = None, 
) -> Tuple[nn.Module, dict]:
    """
    Main entry point used by miners (and potentially validator itself).
    1) Ask the chain for an active validator endpoint.
    2) If available, ping and fetch current model.
    3) Else, initialize a default model.
    """
    config = config or MinerConfig()
    api_base = fetch_validator_endpoint_from_chain(round_hint)

    if api_base:
        pass

    return get_model_from_checkpoint(rank=rank, config=config, expert_manager = expert_manager)
