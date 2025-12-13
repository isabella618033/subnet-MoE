import hashlib
import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def route_tokens_to_experts(router_logits):
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, 10, dim=-1)
    if True:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(router_logits.dtype)
    return selected_experts, routing_weights


def convert_to_str(obj):
    """
    Recursively convert Path/PosixPath objects to strings
    inside any dict, list, or tuple.
    """

    if isinstance(obj, dict):
        return {k: convert_to_str(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convert_to_str(i) for i in obj]

    if isinstance(obj, tuple):
        return tuple(convert_to_str(i) for i in obj)

    if not isinstance(obj, int) and not isinstance(obj, float) and obj is not None:
        return str(obj)

    return obj


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def import_from_string(path: str) -> type:
    """
    Import a class from a string like 'package.module:ClassName'.
    """
    module_path, class_name = path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_nested_attr(obj, attr_chain, default=None):
    for attr in attr_chain.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return default
    return obj


def parse_dynamic_filename(filename: str) -> dict:
    """
    Parse filenames like key_val_key_val... into a dictionary.
    Example:
        uid_13_hotkey_5FnRrH_block_5759026.pt
    → {"uid": 13, "hotkey": "5FnRrH", "block": 5759026}
    """
    # Remove .pt extension
    name = Path(filename).stem

    parts = name.split("_")
    meta = {}
    i = 0
    while i < len(parts) - 1:
        key = parts[i]
        value = parts[i + 1]

        # Handle potential composite keys (non-even splits)
        # Example: if filename has uneven underscores
        if key in meta:  # duplicate key, skip
            i += 1
            continue

        # Try to cast numeric values to int
        try:
            value = int(value)
        except ValueError:
            pass

        meta[key] = value
        i += 2

    meta["filename"] = Path(filename)

    return meta


def h256_int(*parts: Any) -> int:
    """Deterministic 256-bit hash -> int."""
    m = hashlib.sha256()
    for p in parts:
        m.update(str(p).encode("utf-8"))
        m.update(b"\x00")  # separator
    return int.from_bytes(m.digest(), "big")


def serialize_torch_model_path(state) -> bytes:
    """
    Load a torch model from disk and serialize its state_dict
    deterministically into raw bytes.
    """
    # If it's a full model, extract state_dict
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    elif not isinstance(state, dict):
        raise ValueError("Model file must contain a state_dict or nn.Module")

    buffer = []
    for key, tensor in state.items():
        buffer.append(key.encode())
        buffer.append(tensor.cpu().numpy().tobytes())

    return b"".join(buffer)


def hash_model_bytes(model_bytes: bytes) -> bytes:
    """
    Blake2b-256 hash (32 bytes) of the model.
    """
    return hashlib.blake2b(model_bytes, digest_size=32).digest()


def get_model_hash(state):
    """
    Create a model hash from model mocated at specified path.
    """
    # 1. Serialize model → bytes
    model_bytes = serialize_torch_model_path(state)

    # 2. Hash model to 32 bytes
    model_hash = hash_model_bytes(model_bytes)
    return model_hash
