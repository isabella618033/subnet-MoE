import fnmatch
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

import hivemind
from hivemind.averaging import DecentralizedAverager

from mycelia.shared.app_logging import structlog
from mycelia.shared.expert_manager import get_layer_expert_id

logger = structlog.get_logger(__name__)


def get_init_peer_id():
    return ['/ip4/127.0.0.1/tcp/41001/p2p/12D3KooWJbtD23NdFUF7wFCFx6Jz2QTW7C6jM9LmGFgpe4cW4s4Y']


def connect_with_peers():
    initial_peer_ids: List[str] = get_init_peer_id()
    dht = hivemind.DHT(start=True, initial_peers=initial_peer_ids)
    return dht


# --- expert group selection helpers ---
def names_for_expert(
    model: nn.Module, eid, expert_name_fmt: str, include_buffers: bool
) -> List[Tuple[str, torch.Tensor]]:
    """Collect all tensors whose names start with the expert module prefix."""
    prefix = expert_name_fmt.format(eid=eid)
    out = []
    for name, tensor in iter_named_grads(model):
        if name.startswith(prefix + ".") or name == prefix:
            out.append((name, tensor))
    return out


def iter_named_grads(model: nn.Module):
    """
    Yield (name, grad_tensor) for all model parameters that have gradients.
    """
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            p.grad = torch.zeros_like(p)

        yield n, p.grad


def name_selected(name, include_globs, exclude_globs):
    inc_ok = (not include_globs) or any(fnmatch.fnmatch(name, pat) for pat in include_globs)
    exc_ok = not any(fnmatch.fnmatch(name, pat) for pat in exclude_globs)
    return inc_ok and exc_ok


def select_tensors(model, include_globs=(), exclude_globs=()):
    # deterministic order across peers: sort by name!
    chosen = []
    for name, tensor in sorted(iter_named_grads(model), key=lambda kv: kv[0]):
        if name_selected(name, include_globs, exclude_globs):
            chosen.append(tensor)
    return chosen


# --- packaging gradient buff ---
def build_buff_from_params(params):
    numels = [p.numel() for p in params]
    offsets = [0]
    for n in numels[:-1]:
        offsets.append(offsets[-1] + n)
    total = sum(numels)
    flat_grad = torch.zeros(total, device="cpu")  # or cuda

    return {"params": params, "numels": numels, "offsets": offsets, "buff": flat_grad}


def pack_grads(buff_meta):
    with torch.no_grad():
        for p, off, n in zip(buff_meta["params"], buff_meta["offsets"], buff_meta["numels"]):
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            buff_meta["buff"][off : off + n].copy_(g.view(-1))


def unpack_to_grads(buff_meta):
    with torch.no_grad():
        for p, off, n in zip(buff_meta["params"], buff_meta["offsets"], buff_meta["numels"]):
            view = buff_meta["buff"][off : off + n].view_as(p).to(p.device)
            if p.grad is None:
                p.grad = view.clone()
            else:
                p.grad.copy_(view)


# --- getting averager ---

from hivemind.averaging.allreduce import AveragingMode

def build_grad_buff_from_model(
    model: nn.Module,
    expert_group_assignment: Dict[int, Dict[int, List[int]]],
) -> Dict[str | int, Dict]:
    """
    Returns:
      - group_averagers: dict[group_id] -> DecentralizedAverager averaging *all* experts in that group
      - non_expert_averager: DecentralizedAverager averaging all non-expert tensors
    Notes:
      - All peers that should meet in the same group MUST use the same prefix (we derive it from group_id).
      - We sort tensor names to keep a deterministic order across peers.
    """
    # 1) Index tensors by name and prepare expert buckets
    all_named = list(iter_named_grads(model))
    all_named.sort(key=lambda kv: kv[0])  # deterministic order
    name_to_tensor = dict(all_named)
    expert_group_to_names = {group_id: [] for group_id, _ in list(expert_group_assignment.values())[0].items()}

    for name, p in name_to_tensor.items():
        layer_id, expert_id = get_layer_expert_id(name)
        if layer_id and expert_id is not None:
            for group_id, expert_ids in expert_group_assignment[layer_id].items():
                if expert_id in expert_ids:
                    expert_group_to_names[group_id].append(name)

    # 2) Build gradient buffer per expert group
    group_buff_metas: Dict[str | int, Any] = {}
    for group_id in expert_group_to_names.keys():
        tensors_for_group = [name_to_tensor[name] for name in expert_group_to_names[group_id]]
        group_buff_metas[group_id] = build_buff_from_params(params=tensors_for_group)

    expert_owned_names = [name for names in expert_group_to_names.values() for name in names]
    non_expert_names = [n for n, _t in all_named if n not in expert_owned_names]
    non_expert_tensors = [name_to_tensor[n] for n in non_expert_names]
    group_buff_metas["shared"] = build_buff_from_params(non_expert_tensors)

    return group_buff_metas


def build_averagers_from_buff(
    group_buff_metas: Dict[int | str, Dict[str, torch.Tensor]],
    dht: hivemind.DHT,
    prefix_base: str = "expert_averaging",
    target_group_size: int = 4,
    min_group_size: int = 2,
    averaging_alpha: float = 1.0,
) -> Dict[str | int, DecentralizedAverager]:
    """
    Returns:
      - group_averagers: dict[group_id] -> DecentralizedAverager averaging *all* experts in that group
      - non_expert_averager: DecentralizedAverager averaging all non-expert tensors
    Notes:
      - All peers that should meet in the same group MUST use the same prefix (we derive it from group_id).
      - We sort tensor names to keep a deterministic order across peers.
    """

    group_averagers: Dict[str | int, DecentralizedAverager] = {}
    for group_id, buff_meta in group_buff_metas.items():
        prefix = f"{prefix_base}/group{group_id}"
        group_averagers[group_id] = DecentralizedAverager(
            averaged_tensors=[buff_meta["buff"]],
            dht=dht,
            start=True,
            prefix=prefix,
            target_group_size=target_group_size,
            # min_group_size=min_group_size,
            # averaging_alpha=averaging_alpha,
            allreduce_timeout=120,
            client_mode=False,
        )
        logger.info("build hivemind averager - shared", prefix=prefix, mode=group_averagers[group_id].mode, client_mode = group_averagers[group_id].client_mode)

    return group_averagers
