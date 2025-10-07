from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional

import fsspec
import torch
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, field_validator, model_validator

from mycelia.shared.logging import structlog  

logger = structlog.get_logger(__name__)

# ---------------------------
# Enums
# ---------------------------

class Precision(str, Enum):
    fp32 = "fp32"
    bf16 = "bf16"
    fp16_mixed = "fp16-mixed"


class AttnImpl(str, Enum):
    sdpa = "sdpa"
    flash = "flash"
    eager = "eager"


# ---------------------------
# Sections
# ---------------------------

class RunCfg(BaseModel):
    run_name: str = "centralised"
    miner_uid: int = 1

class ModelCfg(BaseModel):
    model_path: str = "deepseek-ai/deepseek-moe-16b-base" # although we are calling a large model here, but we would only be training a partial of it for each miner
    torch_compile: bool = True
    attn_implementation: AttnImpl = AttnImpl.sdpa
    precision: Precision = Precision.fp16_mixed
    device: str = "cuda"

class DataCfg(BaseModel):
    dataset_name: str = "allenai/c4"
    data_dir: str = "en"
    batch_size: PositiveInt = 512
    sequence_length: PositiveInt = 1024
    per_device_train_batch_size: PositiveInt = 5
    world_size: int = 10 #TODO
    rank: int = 1 #TODO


class MoECfg(BaseModel):
    my_expert_group_id: int = 1
    dense_to_moe: bool = True
    interleave: bool = True
    noise: bool = False
    noise_std: float = 0.1
    num_experts: PositiveInt = 8
    num_experts_per_tok: PositiveInt = 2
    partial_topk: PositiveInt = 1
    full_topk: PositiveInt = 2
    aux_load_balance: bool = False
    router_aux_loss_coef: float = 1.0
    partial_moe: bool = True
    num_worker_groups: PositiveInt = 2
    rotate_expert: bool = False
    expert_rotate_interval: Optional[PositiveInt] = None


class OptimizerCfg(BaseModel):
    lr: float = 2e-5
    outer_lr: float = 0.7
    outer_momentum: float = 0.9


class ParallelismCfg(BaseModel):# parallelism for local training
    gradient_accumulation_steps: PositiveInt = 100
    global_opt_interval: PositiveInt = 100
    world_size: PositiveInt = 2
    port: PositiveInt = 29500
    ip_address: str = "127.0.0.1" 

    @staticmethod
    def _cuda_device_count_safe() -> int:
        try:
            return int(torch.cuda.device_count())
        except Exception:
            return 0

class ScheduleCfg(BaseModel):
    warmup_steps: PositiveInt = 600
    total_steps: PositiveInt = 88_000
    # Derived defaults for intervals set in Checkpoint/Logging via global_opt_interval


class CheckpointCfg(BaseModel):
    resume_from_ckpt: bool = True
    base_checkpoint_path: Path = Path("./checkpoints")
    checkpoint_path: Optional[Path] = None
    checkpoint_interval: Optional[PositiveInt] = None
    full_validation_interval: Optional[PositiveInt] = None
    checkpoint_topk: PositiveInt = 20


class LoggingCfg(BaseModel):
    log_wandb: bool = True
    wandb_project_name: str = "test-moe"
    wandb_resume: bool = False
    wandb_full_id: str = "oo2vn2v4"
    wandb_partial_id: List[Optional[str]] = ["3q8mckj8"]
    base_metric_path: Path = Path("./mycelia/metrics")
    metric_path: Optional[Path] = None
    metric_interval: Optional[PositiveInt] = None


# ---------------------------
# Top-level config
# ---------------------------

class Config(BaseModel):
    """
    Centralized training/eval configuration for mycelia runs.

    - Inputs grouped into sections
    - Derived fields computed lazily
    - Run-name bumping if checkpoint dir exists
    """
    run: RunCfg = RunCfg()
    model: ModelCfg = ModelCfg()
    data: DataCfg = DataCfg()
    moe: MoECfg = MoECfg()
    opt: OptimizerCfg = OptimizerCfg()
    local_par: ParallelismCfg = ParallelismCfg()
    sched: ScheduleCfg = ScheduleCfg()
    ckpt: CheckpointCfg = CheckpointCfg()
    log: LoggingCfg = LoggingCfg()

    # -----------------------
    # Derivations & hygiene
    # -----------------------

    @model_validator(mode="after")
    def _derive_all(self):
        # 1) Derived parallelism pieces that depend on data cfg
        if self.local_par.gradient_accumulation_steps == 0:
            # ceil(batch_size / per_device_train_batch_size)
            g = math.ceil(self.data.batch_size / self.data.per_device_train_batch_size)
            self.local_par.gradient_accumulation_steps = max(1, int(g))

        # 2) Derived paths from run_name
        if self.ckpt.checkpoint_path is None:
            self.ckpt.checkpoint_path = self.ckpt.base_checkpoint_path / self.run.run_name

        if self.log.metric_path is None:
            self.log.metric_path = self.log.base_metric_path / f"{self.run.run_name}.csv"

        # 3) Interval defaults relative to global_opt_interval
        goi = self.local_par.global_opt_interval
        if self.ckpt.checkpoint_interval is None:
            self.ckpt.checkpoint_interval = max(1, round(goi * 0.2))
        if self.ckpt.full_validation_interval is None:
            self.ckpt.full_validation_interval = max(1, round(goi * 0.2))
        if self.log.metric_interval is None:
            self.log.metric_interval = max(1, round(goi * 0.2))
        if self.moe.expert_rotate_interval is None:
            self.moe.expert_rotate_interval = max(1, round(goi))

        return self

    # ---- String / JSON helpers ----
    def __str__(self) -> str:
        """Pretty JSON representation of the config."""
        # return json.dumps(self.dict(), indent=4)
        return self.to_json()

    def to_dict(self) -> dict:
        return self.model_dump(mode="python")

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs, indent = 4)
    
    # ---- Lifecycle hooks ----
    def __init__(self, **data):
        """
        Initialize, validate derived fields, and auto-bump `run_name` if an on-disk
        config exists and differs.
        """
        super().__init__(**data)

        # Recompute dependent fields that rely on CUDA availability or run_name.
        self._refresh_paths()

        # If an existing config exists, bump run_name when the configs don't match.
        config_path = os.path.join(self.ckpt.checkpoint_path, "config.json")
        while os.path.exists(config_path):
            logger.info(f"found existing config path {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config_dict = json.load(f)
            bumped = self.bump_run_name_if_diff(existing_config_dict)

            if not bumped:  # landed on the same config
                return
            else:
                # bumped so need to check on the config at the new folder
                config_path = os.path.join(self.ckpt.checkpoint_path, "config.json")

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """
        Load a Config from a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON config file.

        Returns
        -------
        Config
            Instantiated Config object.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"loaded config {data}")
        return cls(**data)

    # ---- Internal utilities ----
    def _refresh_paths(self) -> None:
        """Refresh paths that depend on `run_name`."""
        self.ckpt.checkpoint_path = os.path.join(self.ckpt.base_checkpoint_path, self.run.run_name)
        self.log.metric_path = os.path.join(self.log.base_metric_path, f"{self.run.run_name}.csv")

    @staticmethod
    def _bump_run_name(name: str) -> str:
        """
        Increment a run name in a `-vN` style.

        Examples
        --------
        "foo" -> "foo-v2"
        "foo-v3" -> "foo-v4"
        "test-1B" -> "test-1B-v2"
        """
        m = re.match(r"^(.*?)(?:-v(\d+))?$", name)
        if m:
            base, ver = m.group(1), m.group(2)
            return f"{base}-v{int(ver) + 1}" if ver else f"{base}-v2"
        return name + "-v2"

    def _norm(self, v: Any) -> Any:
        # Optional: normalize types that often differ but mean the same thing
        if isinstance(v, Path):
            return v.as_posix()
        return v

    def _deep_compare(self, a: Any, b: Any, path: str = "") -> bool:
        ok = True

        # Dict vs dict: recurse by keys
        if isinstance(a, dict) and isinstance(b, dict):
            a_keys, b_keys = set(a.keys()), set(b.keys())
            missing = a_keys - b_keys
            extra   = b_keys - a_keys
            if missing:
                logger.info(f"Keys present in self but missing in other at '{path}': {missing}")
                ok = False
            if extra:
                logger.info(f"Keys present in other but missing in self at '{path}': {extra}")
                ok = False
            for k in sorted(a_keys & b_keys):
                if not self._deep_compare(a[k], b[k], f"{path}.{k}" if path else k):
                    ok = False
            return ok

        # Sequence vs sequence: compare length then elementwise
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                logger.info(f"Length mismatch at '{path}': existing {len(b)} vs new {len(a)}")
                return False
            for i, (ai, bi) in enumerate(zip(a, b)):
                if not self._deep_compare(ai, bi, f"{path}[{i}]"):
                    ok = False
            return ok

        # Base case: compare normalized scalars/others
        a_n, b_n = self._norm(a), self._norm(b)
        if a_n != b_n:
            logger.info(f"Config mismatch at '{path}': existing {b_n} vs new {a_n}")
            return False
        return True

    # ---- Comparison & versioning ----
    def same_as(self, other: Dict) -> bool:
        """
        Return True if configs are equivalent (deep comparison).
        `other` is a dict (possibly nested) from the same schema.
        """
        self_dict = self.to_dict()
        return self._deep_compare(self_dict, other)

    def bump_run_name_if_diff(self, other: Dict) -> bool:
        """
        If configs differ, bump `run_name` to the next version and refresh paths.

        Returns
        -------
        bool
            True if bumped, False if configs were the same.
        """
        if self.same_as(other):
            return False
        self.run.run_name = self._bump_run_name(self.run.run_name)
        self._refresh_paths()
        logger.info(f"Bumped run_name to {self.run.run_name} due to config differences.")
        return True

    # ---- Persistence ----
    def write(self) -> None:
        """
        Persist this config to `<checkpoint_path>/config.json`.

        Creates the directory if it does not exist.
        """
        os.makedirs(self.ckpt.checkpoint_path, exist_ok=True)
        target = os.path.join(self.ckpt.checkpoint_path, "config.json")
        with fsspec.open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        logger.info(f"Wrote config to {target}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train mycelia with config")
    parser.add_argument(
        "--path",
        type=str,
        help="Path to JSON config file",
    )
    return parser.parse_args()
