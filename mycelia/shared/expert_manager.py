from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from mycelia.shared.app_logging import structlog
from mycelia.shared.config import TaskCfg, WorkerConfig


logger = structlog.getLogger(__name__)

# ------------------------------------------------------------
# Expert Manager
# ------------------------------------------------------------
ExpertMapping = Tuple[int, int]                     # (my_expert_idx, org_expert_idx)
LayerAssignments = Dict[int, List[ExpertMapping]]   # layer_id -> list of mappings
ExpertAssignments = Dict[int, LayerAssignments]     # group_id -> layer assignments

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

    def __init__(self, config: WorkerConfig, model: nn.Module | None = None):
        if model is not None:
            self.set_expert_layers(model)
        else:
            self.expert_layers = None

        self.expert_group_assignment = self.load_expert_group_assignment(config)
        self.validate_unique_mycelia_expert_ids()
        self.validate_expert_layers()

    @property
    def num_expert_groups(self) -> int:
        """
        Number of expert groups present in the assignment.
        Directly equal to number of group_id keys.
        """
        return len(self.expert_group_assignment)

    @property
    def num_experts(self) -> int:
        """
        Total count of unique my_expert_idx across all groups/layers.
        """
        expert_ids = set()

        for layers in self.expert_group_assignment.values():
            for mappings in layers.values():
                for my_expert_idx, _ in mappings:
                    expert_ids.add(my_expert_idx)

        return len(expert_ids)
    
    def get_num_experts_in_group(self, group_id) -> int:
        """
        Total count of unique my_expert_idx across all groups/layers.
        """
        expert_ids = set()

        for layer_id, mappings in self.expert_group_assignment[group_id].items():
            for my_expert_idx, _ in mappings:
                expert_ids.add(my_expert_idx)

        return len(expert_ids)
    
    def set_expert_layers(self, model: nn.Module) -> List[int]:
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

        self.expert_layers = expert_layers
        self.validate_expert_layers()

    # ---- loading ----
    def load_expert_group_assignment(self, config) -> ExpertAssignments:
        base_path: Path = config.task.base_path
        task_folders = [d for d in base_path.iterdir() if d.is_dir()]

        expert_assignments: ExpertAssignments = {}

        for task_folder in task_folders:
            logger.info('loading task folder', task_folder)
            # Load per-task config (to get expert_group_id)
            task_config = TaskCfg.from_path(task_folder / "config.yaml")

            # Load raw JSON assignment
            with open(task_folder / "expert_assignment.json", "r", encoding="utf-8") as f:
                raw_assignment = json.load(f)

            # raw_assignment: Dict[str, List[List[int]]]
            # Convert to: LayerAssignments (Dict[int, List[Tuple[int, int]]])
            layer_assignments: LayerAssignments = {}
            for layer_id_str, pair_list in raw_assignment.items():
                layer_id = int(layer_id_str)
                # Ensure we store tuples of ints, not lists
                mappings: List[ExpertMapping] = [tuple(pair) for pair in pair_list]
                layer_assignments[layer_id] = mappings

            # Map this task's expert_group_id -> its layer assignments
            expert_assignments[task_config.expert_group_id] = layer_assignments

        return expert_assignments

    # ---- Check correctness ----
    def validate_unique_mycelia_expert_ids(self) -> None:
        """
        Check that `my_expert_idx` is unique within each (group_id, layer_id).

        In other words, for any fixed (group_id, layer_id) pair, the same
        `my_expert_idx` must not appear more than once in its mapping list.
        Reuse of a `my_expert_idx` across *different* groups or layers is allowed.

        Raises
        ------
        ValueError
            If any `my_expert_idx` appears more than once in the same
            (group_id, layer_id).
        """
        duplicates: List[str] = []

        for group_id, layers in self.expert_group_assignment.items():
            for layer_id, mappings in layers.items():
                seen_in_layer: set[int] = set()
                for my_expert_idx, org_expert_idx in mappings:
                    if my_expert_idx in seen_in_layer:
                        duplicates.append(
                            f"mycelia_expert_idx={my_expert_idx} duplicated "
                            f"within (group={group_id}, layer={layer_id})"
                        )
                    else:
                        seen_in_layer.add(my_expert_idx)

        if duplicates:
            # Show a concise but useful error
            msg = (
                "Duplicate mycelia_expert_idx found within expert groups/layers:\n"
                + "\n".join(duplicates[:10])
            )
            if len(duplicates) > 10:
                msg += f"\n... and {len(duplicates) - 10} more"
            raise ValueError(msg)

    def validate_expert_layers(self) -> None:
        """
        Validate that for every expert group, the set of layer_ids matches
        exactly the required set in expert_layers (inclusive + exclusive).

        Raises
        ------
        ValueError
            If any group is missing layers or has extra layers.
        """
        if self.expert_layers is None:
            return
          
        expected = set(self.expert_layers)
        errors = []

        for group_id, layers in self.expert_group_assignment.items():
            actual = set(layers.keys())

            missing = expected - actual
            extra   = actual - expected

            if missing or extra:
                msg = [f"Group {group_id} layer mismatch:"]

                if missing:
                    msg.append(f"  Missing layers: {sorted(missing)}")
                if extra:
                    msg.append(f"  Extra layers: {sorted(extra)}")

                errors.append("\n".join(msg))

        if errors:
            raise ValueError(
                "Expert layer validation failed:\n\n" + "\n\n".join(errors)
            )
# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def is_expert_param(name: str) -> bool:
    """Heuristic to detect MoE expert parameters by name."""
    return "expert" in name  # customize if needed (e.g., "experts.")

def get_layer_expert_id(layer_name: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract (layer_id, expert_id) from a parameter name.

    Examples
    --------
    "model.layers.3.mlp.experts.7.w1.weight" -> (3, 7)
    "model.layers.5.mlp.gate.weight"         -> (5, None)
    """
    m = re.search(r"model\.layers\.(\d+)(?:\.mlp\.experts\.(\d+))?", layer_name)
    if not m:
        return None, None
    layer_id = int(m.group(1))
    expert_id = int(m.group(2)) if m.group(2) is not None else None
    return layer_id, expert_id


def split_into_groups(lst: List[int], num_groups: int, shuffle: bool = False, seed: int | None = 123) -> Dict[int, List[int]]:
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

    if shuffle:
        shuffled = lst[:]
        if seed is not None:
            rnd = random.Random(seed)
            rnd.shuffle(shuffled)

        return {i: shuffled[i::num_groups] for i in range(num_groups)}

    else:
        return { i: lst[i * (len(lst)//num_groups) : (i + 1) * (len(lst)//num_groups)] for i in range(num_groups) }
   
    
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
        (group_ids, groups_by_id)

    Notes
    -----
    * Requires `dist.is_initialized()` to be True.
    * Each call will create new groups; reuse the returned dict across calls in your job.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before creating groups")

    expert_groups: Dict[int, dist.ProcessGroup] = {}
    group_ids: int | None = None

    for group_id, ranks in rank_group_assignment.items():
        group = dist.new_group(ranks=ranks)
        expert_groups[group_id] = group
        if my_rank in ranks:
            group_ids = group_id

    if group_ids is None:
        raise ValueError(f"Rank {my_rank} not present in any provided group assignment")

    return group_ids, expert_groups

# ------------------------------------------------------------
# Synchronization primitives
# ------------------------------------------------------------
def _named_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    """Return a dict name -> parameter for stable matching across models."""
    return dict(model.named_parameters())


def populate_global_grads_from_local(
    global_model: nn.Module, model: nn.Module, shared_only: bool = False, weight: float = 0.2
) -> None:
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
    local_named = _named_params(model)
    global_named = _named_params(global_model)

    for name, p in local_named.items():
        if shared_only and is_expert_param(name):
            continue

        g = global_named.get(name)
        if g is None:
            logger.warning(f"Shared param '{name}' not found in global model; skipping.")
            continue

        diff = g.data - p.data
        if g.grad is None:
            g.grad = diff * weight
        else:
            g.grad += diff * weight


def sync_weights(rank: int, global_model: nn.Module, shared_only: bool = False) -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before sync")

    global_named = _named_params(global_model)

    for name, g in global_named.items():
        if shared_only and is_expert_param(name):
            continue

        dist.all_reduce(g.grad, op=dist.ReduceOp.AVG)


def sync_expert_weights(
    rank: int,
    global_model: nn.Module,
    model: nn.Module,
    group_ids: int,
    expert_groups: Mapping[int, dist.ProcessGroup],
) -> None:
    """
    Average the differences for *expert* parameters within this expert group only.

    Parameters
    ----------
    group_ids : int
        ID of the group this rank belongs to.
    expert_groups : Mapping[int, ProcessGroup]
        Mapping from group ID to its ProcessGroup.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before sync")

    group = expert_groups.get(group_ids)
    if group is None:
        raise KeyError(f"No process group for group_ids={group_ids}")

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
    group_ids: int,
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

    src_expert_rank = min(rank_group_assignment[group_ids])
    expert_group = expert_groups[group_ids]

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
