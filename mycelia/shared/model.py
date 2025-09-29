# llm_weightnet/shared/model.py
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional, Tuple

import requests
import torch
from torch import nn

from ..settings import Settings
from .storage import get_bytes, put_bytes  # optional, if you store artifacts remotely
from .logging import structlog  # or standard logging if you prefer
from . import blockchain

log = structlog.get_logger(__name__)

# ---- Default tiny model (stand-in) ----
class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: int = 512, out_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _default_model(settings: Settings) -> nn.Module:
    """
    Provide a small, dependency-light model that always works in CPU/GPU.
    Swap for HF AutoModel if you prefer (commented example below).
    """
    model = TinyMLP(in_dim=1024, hidden=512, out_dim=1024)
    device = torch.device(settings.device if torch.cuda.is_available() or "cuda" in settings.device else "cpu")
    model.to(device)
    log.info("model.default_initialized", kind="TinyMLP", device=str(device))
    return model

# If you want a HF-based default instead, uncomment:
# from transformers import AutoModel
# def _default_model(settings: Settings) -> nn.Module:
#     model = AutoModel.from_pretrained(settings.model_name)
#     device = torch.device(settings.device if torch.cuda.is_available() or "cuda" in settings.device else "cpu")
#     model.to(device)
#     log.info("model.default_initialized", kind="HF", name=settings.model_name, device=str(device))
#     return model


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
        log.warning("model.chain_lookup_failed", error=str(e))
        return None


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
        log.info("model.validator_unreachable", api_base=api_base, error=str(e))
        return None


def _download_artifact(manifest: dict, settings: Settings) -> bytes:
    """
    Download the model artifact. Supports:
      - Direct bytes via validator
      - Indirect via artifact_uri (S3/IPFS/local) using storage.py
    """
    if "artifact_bytes_url" in manifest:
        url = manifest["artifact_bytes_url"]
        log.info("model.downloading_direct", url=url)
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content

    uri = manifest.get("artifact_uri")
    if not uri:
        raise ValueError("Manifest missing 'artifact_uri' or 'artifact_bytes_url'")
    log.info("model.downloading_via_storage", uri=uri)
    return get_bytes(uri)  # your storage backend should handle s3://, file://, ipfs://, etc.


def _load_from_blob(blob: bytes, fmt: str, device: str) -> nn.Module:
    """
    Convert blob -> model instance.
    Default assumes a Torch state_dict was saved via torch.save(state_dict).
    """
    buffer = io.BytesIO(blob)
    state = torch.load(buffer, map_location="cpu")
    model = TinyMLP()  # must match architecture used by validator
    model.load_state_dict(state, strict=False)
    dev = torch.device(device if torch.cuda.is_available() or "cuda" in device else "cpu")
    model.to(dev)
    model.eval()
    return model


def load_base(settings: Optional[Settings] = None, round_hint: Optional[str] = None) -> nn.Module:
    """
    Main entry point used by miners (and potentially validator itself).
    1) Ask the chain for an active validator endpoint.
    2) If available, ping and fetch current model.
    3) Else, initialize a default model.
    """
    cfg = settings or Settings()
    api_base = _fetch_validator_endpoint_from_chain(round_hint)

    if api_base:
        manifest = _ping_validator_and_get_manifest(api_base)
        if manifest:
            try:
                blob = _download_artifact(manifest, cfg)
                model = _load_from_blob(blob, manifest.get("format", "state_dict"), cfg.device)
                log.info("model.loaded_from_validator", api_base=api_base)
                return model
            except Exception as e:
                log.warning("model.validator_load_failed", error=str(e))

    # Fallback
    return _default_model(cfg)


# --- Helper: export/save current model to artifact (used by validator) ---

def export_model_artifact(model: nn.Module) -> Tuple[bytes, str]:
    """
    Serialize the model to bytes + return format string.
    Default: state_dict via torch.save
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read(), "state_dict"
