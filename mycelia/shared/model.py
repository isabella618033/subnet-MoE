from __future__ import annotations

import bittensor
from torch import nn

from mycelia.shared.app_logging import structlog
from mycelia.shared.chain import fetch_model_from_chain
from mycelia.shared.checkpoint import ModelMeta, load_checkpoint, start_model_from
from mycelia.shared.config import MinerConfig, ValidatorConfig
from mycelia.shared.expert_manager import ExpertManager
from mycelia.shared.helper import get_nested_attr
from mycelia.shared.modeling.mycelia import get_base_model

logger = structlog.get_logger(__name__)


def get_model_from_checkpoint(
    rank: int, config: MinerConfig | ValidatorConfig, expert_manager: ExpertManager
) -> tuple[nn.Module, ModelMeta]:
    resume = False
    latest_checkpoint_path = None

    logger.info(
        "Get base model for checkpoint",
        group_ids=[config.task.expert_group_id] if config.role == "miner" else None,
        partial=(config.role == "miner"),
    )
    # get base model
    model = get_base_model(
        config,
        expert_manager=expert_manager,
        group_ids=[config.task.expert_group_id] if config.role == "miner" else None,
        partial=(config.role == "miner"),
    ).to(config.model.device)

    # load from checkpoint
    if get_nested_attr(config, "ckpt.resume_from_ckpt", False):
        resume, model_version, latest_checkpoint_path = start_model_from(
            rank,
            config,
            primary_ckpt_path=config.ckpt.validator_checkpoint_path,
            secondary_ckpt_path=config.ckpt.checkpoint_path,
        )

        if resume and latest_checkpoint_path:
            load_checkpoint(
                config=config,
                checkpoint_path=latest_checkpoint_path,
                model=model,
                rank=rank,
                device=config.model.device,
            )
        else:
            logger.info("Tried to resume from checkpoint, but no checkpoint found.")

    model = model.to(config.model.device)
    model.gradient_checkpointing_enable()
    return model, model_version


def load_model(
    rank: int,
    config: MinerConfig | ValidatorConfig,
    expert_manager: ExpertManager,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    current_model_meta: ModelMeta | None = None,
) -> tuple[nn.Module, dict]:
    """
    Main entry point used by miners (and potentially validator itself).
    1) Ask the chain for an active validator endpoint.
    2) If available, ping and fetch current model.
    3) Else, initialize a default model.
    """
    # download new model from chain into file
    fetch_model_from_chain(current_model_meta=current_model_meta, config=config, subtensor=subtensor, wallet=wallet)
    return get_model_from_checkpoint(rank=rank, config=config, expert_manager=expert_manager)
