# llm_weightnet/shared/model.py
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional, Tuple

import requests
import torch
from torch import nn

from mycelia.config import Config
from mycelia.miner.checkpoint import get_resume_info
from mycelia.shared.logging import structlog  
from mycelia.shared.expert_manager import ExpertManager, create_expert_groups 
from mycelia.shared.modeling_moe import get_base_model  , partial_moe
from mycelia.shared import blockchain

logger = structlog.get_logger(__name__)

def _default_model(config: Config) -> nn.Module:
    resume = False   
    start_step = 0
    latest_checkpoint_path = None
    if getattr(config, "resume_from_ckpt", False):
        resume, start_step, latest_checkpoint_path = get_resume_info(config)

    em = ExpertManager(
        model=get_base_model(config, noise= True),
        rank=rank,
        num_experts=config.moe.num_experts,
        num_worker_groups=config.moe.num_worker_groups,
    )
    em.compute_group_assignments(seed=start_step if config.moe.rotate_expert else 0)

    model = get_base_model(config, noise=start_step == 0, expert_group_assignment=em.expert_group_assignment).to(config.model.device)

    # model has to be loaded before partitioning into moe
    if getattr(config, "resume_from_ckpt", False) and resume and latest_checkpoint_path:
        logger.info(
            "rank %s setup training: resuming from %s (start_step=%s)", rank, latest_checkpoint_path, start_step
        )
        _ = load_checkpoint(
            config=config, checkpoint_path=latest_checkpoint_path, model=model, rank=rank, device=config.model.device
        )

    if getattr(config, "partial_moe", False):
        model = partial_moe(config, model, config.moe.my_expert_group_id, em.expert_group_assignment)

    model = model.to(config.model.device)

    return model, em


# TODO
def _fetch_validator_endpoint_from_chain(round_hint: Optional[str] = None) -> Optional[str]:
    """
    Ask chain for the *current* validator node endpoint (e.g., https://validator-1:8080).
    You implement this inside shared/blockchain.py
    """
    try:
        info = blockchain.get_active_validator_info(round_hint)  # must return {"api_base": "...", ...}
        endpoint = (info or {}).get("api_base")
        return endpoint
    except Exception as e:
        logger.warning("model.chain_lookup_failed", error=str(e))
        return None

# TODO
def _ping_validator_and_get_manifest(api_base: str, timeout: float = 5.0) -> Optional[dict]:
    """
    Ping validator and retrieve a model manifest:
      { "artifact_uri": "s3://bucket/path/model.bin", "format": "state_dict", "sha256": "..." }
    Or returns None if not ready.
    """
    try:
        r = requests.get(f"{api_base}/v1/health", timeout=timeout)
        r.raise_for_status()
        # Optionally check response content (e.g., {"status":"ok"})
        r = requests.get(f"{api_base}/v1/model/manifest", timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        logger.info("model.validator_unreachable", api_base=api_base, error=str(e))
        return None

# TODO
def _download_artifact(manifest: dict, settings: Settings) -> bytes:
    """
    Download the model artifact. Supports:
      - Direct bytes via validator
      - Indirect via artifact_uri (S3/IPFS/local) using storage.py
    """
    if "artifact_bytes_url" in manifest:
        url = manifest["artifact_bytes_url"]
        logger.info("model.downloading_direct", url=url)
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content

    uri = manifest.get("artifact_uri")
    if not uri:
        raise ValueError("Manifest missing 'artifact_uri' or 'artifact_bytes_url'")
    logger.info("model.downloading_via_storage", uri=uri)
    return get_bytes(uri)  # your storage backend should handle s3://, file://, ipfs://, etc.

# TODO
def _load_from_blob(blob: bytes, fmt: str, device: str) -> nn.Module:
    """
    Convert blob -> model instance.
    Default assumes a Torch state_dict was saved via torch.save(state_dict).
    """
    buffer = io.BytesIO(blob)
    state = torch.load(buffer, map_location="cpu")
    model = _default_model()  # must match architecture used by validator
    model.load_state_dict(state, strict=False)
    dev = torch.device(device if torch.cuda.is_available() or "cuda" in device else "cpu")
    model.to(dev)
    model.eval()
    return model


def load_base_model(config: Optional[Config] = None, round_hint: Optional[str] = None) -> Tuple[nn.Module, ExpertManager]:
    """
    Main entry point used by miners (and potentially validator itself).
    1) Ask the chain for an active validator endpoint.
    2) If available, ping and fetch current model.
    3) Else, initialize a default model.
    """
    config = config or Config()
    api_base = _fetch_validator_endpoint_from_chain(round_hint)

    if api_base:
        manifest = _ping_validator_and_get_manifest(api_base)
        if manifest:
            try:
                blob = _download_artifact(manifest, config)
                model = _load_from_blob(blob, manifest.get("format", "state_dict"), config.model.device)
                logger.info("model.loaded_from_validator", api_base=api_base)
                return model
            except Exception as e:
                logger.warning("model.validator_load_failed", error=str(e))

    # Fallback
    return _default_model(rank = config.run.miner_uid, config = config)

# --- Helper: export/save current model to artifact (used by validator) ---
# TODO
def export_model_artifact(model: nn.Module) -> Tuple[bytes, str]:
    """
    Serialize the model to bytes + return format string.
    Default: state_dict via torch.save
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read(), "state_dict"
