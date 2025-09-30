"""
Expert (MoE) group management and distributed weight synchronization utilities.

This module provides:
  * `ExpertManager` to discover expert-bearing layers and compute rank/expert groupings.
  * Helpers to deterministically split ranks/experts into groups.
  * Utilities to create torch.distributed process groups per expert group.
  * Synchronization primitives to average either shared weights (global) or expert
    weights (within each expert group), plus broadcasting helpers.

Assumptions
-----------
* `torch.distributed` has been initialized (e.g., `init_process_group`) before calling
  any collective ops here.
* "Expert" parameters are identified by the substring `"expert"` in their names. Adjust
  `is_expert_param` if your naming differs (e.g., `"experts."`).
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from mycelia.shared.modeling_moe import get_layer_expert_id

logger = logging.getLogger("diloco.expert_manager")


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def is_expert_param(name: str) -> bool:
    """Heuristic to detect MoE expert parameters by name."""
    return "expert" in name  # customize if needed (e.g., "experts.")


def split_into_groups(lst: List[int], num_groups: int, seed: int | None = 123) -> Dict[int, List[int]]:
    """
    Deterministically split a list of items into `num_groups` interleaved buckets.

    Parameters
    ----------
    lst : List[int]
        Items to split (e.g., ranks or expert IDs).
    num_groups : int
        Number of buckets to produce.
    seed : Optional[int]
        Seed for reproducible shuffling. If None, keeps original order.

    Returns
    -------
    Dict[int, List[int]]
        Mapping: group_id -> sublist of items.

    Notes
    -----
    Uses a local RNG so global randomness is unaffected.
    """
    if num_groups <= 0:
        raise ValueError("num_groups must be >= 1")

    shuffled = lst[:]
    if seed is not None:
        rnd = random.Random(seed)
        rnd.shuffle(shuffled)

    return {i: shuffled[i::num_groups] for i in range(num_groups)}


def create_expert_groups(
    my_rank: int, rank_group_assignment: Mapping[int, Iterable[int]]
) -> Tuple[int, Dict[int, dist.ProcessGroup]]:
    """
    Create torch.distributed process groups for each expert group.

    Parameters
    ----------
    my_rank : int
        This process' global rank.
    rank_group_assignment : Mapping[int, Iterable[int]]
        Mapping of group_id -> ranks in that group.

    Returns
    -------
    Tuple[int, Dict[int, ProcessGroup]]
        (my_group_id, groups_by_id)

    Notes
    -----
    * Requires `dist.is_initialized()` to be True.
    * Each call will create new groups; reuse the returned dict across calls in your job.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before creating groups")

    expert_groups: Dict[int, dist.ProcessGroup] = {}
    my_group_id: int | None = None

    for group_id, ranks in rank_group_assignment.items():
        group = dist.new_group(ranks=ranks)
        expert_groups[group_id] = group
        if my_rank in ranks:
            my_group_id = group_id

    if my_group_id is None:
        raise ValueError(f"Rank {my_rank} not present in any provided group assignment")

    return my_group_id, expert_groups


# ------------------------------------------------------------
# Expert Manager
# ------------------------------------------------------------
class ExpertManager:
    """
    Manages expert-aware grouping for distributed Mixture-of-Experts training.

    Responsibilities
    ----------------
    * Discover which transformer layers contain experts (by scanning state_dict keys).
    * Split ranks into `num_worker_groups` groups.
    * Split experts (per expert layer) into the same number of groups.

    Attributes
    ----------
    expert_layers : List[int]
        Indices of layers that contain experts (best-effort heuristic).
    rank_group_assignment : Dict[int, List[int]]
        Mapping of group_id -> ranks in that group.
    expert_group_assignment : Dict[int, Dict[int, List[int]]]
        Mapping of layer -> (group_id -> expert IDs in that group).
    """

    def __init__(
        self, rank: int, num_experts: int, num_worker_groups: int, model: nn.Module | None = None
    ):
        self.rank = rank
        self.num_experts = int(num_experts)
        self.num_worker_groups = int(num_worker_groups)
        if model is not None:
            self.expert_layers = self._discover_expert_layers(model)
        else:
            self.expert_layers = None

    def set_expert_layers(self, model: nn.Module):
        self.expert_layers = self._discover_expert_layers(model)

    # ---- discovery ----
    def _discover_expert_layers(self, model: nn.Module) -> List[int]:
        """
        Inspect model state_dict keys to locate layers that include experts.

        Heuristic: looks for keys like "{layer_idx}.mlp.experts..." (tweak for your arch).
        """
        sd_keys = model.state_dict().keys()
        num_layers = getattr(getattr(model, "config", object()), "num_hidden_layers", None)
        if num_layers is None:
            logger.warning("Model has no config.num_hidden_layers; scanning keys without layer bounds.")

        expert_layers: List[int] = []
        # If num_layers is unknown, fall back to a generous range (0..255).
        layer_range = range(num_layers if isinstance(num_layers, int) else 256)
        for l in layer_range:
            pattern = f"{l}.mlp.experts"
            if any(pattern in k for k in sd_keys):
                expert_layers.append(l)

        if not expert_layers:
            logger.info("No expert-bearing layers found by heuristic; check naming or discovery logic.")
        else:
            logger.info(f"Detected expert layerss: {expert_layers}")

        return expert_layers

    # ---- grouping ----
    def compute_group_assignments(self, seed=0) -> None:
        """Compute rank groups and expert groups (per expert layer)."""
        expert_groups: Dict[int, Dict[int, List[int]]] = {}  # layer -> (group_id -> [expert_ids])
        for layer in self.expert_layers:
            experts = list(range(self.num_experts))
            expert_groups[layer] = split_into_groups(experts, self.num_worker_groups, seed=123 + layer**2 + seed)

        self.expert_group_assignment = expert_groups

        for layer, groups in self.expert_group_assignment.items():
            if layer == 0 or layer == max(self.expert_group_assignment):
                logger.info(f"rank {self.rank} seed {seed} Expert group assignment for layer {layer}: {groups}")

    # ---- loading ----
    def _load_expert_assignments(self, my_group_id, state_dict) -> None:
        """Compute expert groups fom model (per expert layer)."""

        if hasattr(self, "expert_group_assignment"):
            expert_groups = self.expert_group_assignment
        else:
            expert_groups: Dict[int, Dict[int, List[int]]] = {}  # layer -> (group_id -> [expert_ids])

        for k in state_dict.keys():
            layer_id, expert_id = get_layer_expert_id(k)

            if layer_id is None or expert_id is None:
                continue

            if layer_id not in expert_groups:
                expert_groups[layer_id] = {}

            if my_group_id not in expert_groups[layer_id]:
                expert_groups[layer_id][my_group_id] = []

            if expert_id not in expert_groups[layer_id][my_group_id]:
                expert_groups[layer_id][my_group_id].append(expert_id)

        self.expert_group_assignment = expert_groups


# ------------------------------------------------------------
# Synchronization primitives
# ------------------------------------------------------------
def _named_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    """Return a dict name -> parameter for stable matching across models."""
    return dict(model.named_parameters())


def sync_weights(rank: int, global_model: nn.Module, model: nn.Module, shared_only: bool = False) -> None:
    """
    Average the differences for *shared* (non-expert) parameters across all ranks.

    Workflow
    --------
    * For each shared param `p` in `model` and corresponding `g` in `global_model`:
        grad_g = (g.data - p.data)
        all_reduce(grad_g, AVG)  # average difference across workers
      (You typically apply these "gradients" via an optimizer on `global_model` later.)

    Notes
    -----
    * We avoid relying on parameter iteration order by matching by name.
    * Uses `.data` to avoid autograd tracking (intentional, as these are sync ops).
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before sync")

    local_named = _named_params(model)
    global_named = _named_params(global_model)

    for name, p in local_named.items():
        if shared_only and is_expert_param(name):
            continue
        g = global_named.get(name)
        if g is None:
            logger.warning(f"[rank {rank}] Shared param '{name}' not found in global model; skipping.")
            continue

        # Compute difference and average it across all ranks into g.grad
        diff = g.data - p.data
        g.grad = diff  # store in .grad for a later optimizer step on global_model
        dist.all_reduce(g.grad, op=dist.ReduceOp.AVG)


def sync_expert_weights(
    rank: int,
    global_model: nn.Module,
    model: nn.Module,
    my_group_id: int,
    expert_groups: Mapping[int, dist.ProcessGroup],
) -> None:
    """
    Average the differences for *expert* parameters within this expert group only.

    Parameters
    ----------
    my_group_id : int
        ID of the group this rank belongs to.
    expert_groups : Mapping[int, ProcessGroup]
        Mapping from group ID to its ProcessGroup.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before sync")

    group = expert_groups.get(my_group_id)
    if group is None:
        raise KeyError(f"No process group for my_group_id={my_group_id}")

    local_named = _named_params(model)
    global_named = _named_params(global_model)

    for name, p in local_named.items():
        if not is_expert_param(name):
            continue
        g = global_named.get(name)
        if g is None:
            logger.warning(f"[rank {rank}] Expert param '{name}' not found in global model; skipping.")
            continue

        diff = g.data - p.data
        g.grad = diff
        dist.all_reduce(g.grad, op=dist.ReduceOp.AVG, group=group)


def broadcast_weights(
    model: nn.Module,
    my_group_id: int,
    rank_group_assignment: Mapping[int, Iterable[int]],
    expert_groups: Mapping[int, dist.ProcessGroup],
) -> None:
    """
    Broadcast parameters so every rank has a consistent view.

    Behavior
    --------
    * Expert params: broadcast **within** group from the lowest-rank member.
    * Shared params: broadcast **globally** from rank 0.

    Notes
    -----
    Ensure collectives are called by all ranks consistently.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before broadcast")

    src_expert_rank = min(rank_group_assignment[my_group_id])
    expert_group = expert_groups[my_group_id]

    for name, p in model.named_parameters():
        if is_expert_param(name):
            dist.broadcast(p.data, src=src_expert_rank, group=expert_group)
        else:
            dist.broadcast(p.data, src=0)


def get_weight_sum(model: nn.Module, shared: bool = True) -> Tuple[str, torch.Tensor]:
    """
    Return the (name, sum) for the first parameter that matches the filter.

    Parameters
    ----------
    model : nn.Module
        Model to inspect.
    shared : bool
        If True, look at non-expert (shared) params; else look at expert params.

    Returns
    -------
    Optional[Tuple[str, Tensor]]
        (parameter_name, tensor_sum) for the first matching parameter, or None if none match.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if shared and not is_expert_param(name):
                return name, p.data.sum()
            if not shared and is_expert_param(name):
                return name, p.data.sum()
