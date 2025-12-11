from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any

import bittensor
import fsspec
import torch
import yaml
from pydantic import BaseModel, PositiveInt, model_validator

from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.helper import convert_to_str

configure_logging()
logger = structlog.get_logger(__name__)


def find_project_root() -> Path:
    # Walk up until we see a repo/config marker
    start = Path("./").expanduser().resolve(strict=True)
    for p in [start, *start.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "requirements.txt").exists():
            return p
    return start.parents[1]  # fallback: up one level from mycelia/


class BaseConfig(BaseModel):
    def __str__(self) -> str:
        """Pretty JSON representation of the config."""
        # return json.dumps(self.dict(), indent=4)
        return self.to_json()

    def to_dict(self) -> dict:
        return self.model_dump(mode="python")

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs, indent=4)

    @classmethod
    def from_path(cls, path: str | Path) -> BaseConfig:
        """
        Load a MinerConfig from a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON config file.

        Returns
        -------
        MinerConfig
            Instantiated MinerConfig object.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ---------------------------
# Sections
# ---------------------------
class ChainCfg(BaseConfig):
    netuid: int = 348
    uid: int = 1
    hotkey_ss58: str = ""
    coldkey_ss58: str = ""
    ip: str = "0.0.0.0"
    port: int = 8000
    role: str = "miner"  # ['miner', 'validator']
    coldkey_name: str = "template_coldkey_name"
    hotkey_name: str = "template_hotkey_name"
    network: str = "test"


class CycleCfg(BaseConfig):
    cycle_length: int = 35  # validators run a validation round everytime when sub.block % cycle_length == 0
    distribute_period: int = 10
    train_period: int = 20
    commit_period: int = 5
    submission_period: int = 10
    validate_period: int = 10
    merge_period: int = 20

    owner_url: str = "http://localhost:7000"


class RunCfg(BaseConfig):
    run_name: str = "foundation"
    root_path: Path = find_project_root()


class ModelCfg(BaseConfig):
    # although we are calling a large model here, but we would only be training a partial of it for each miner
    model_path: str = "Qwen/Qwen3-Next-80B-A3B-Thinking"
    foundation: bool = True
    torch_compile: bool = True
    attn_implementation: str = "sdpa"
    precision: str = "fp16-mixed"
    device: str = "cuda"


class DataCfg(BaseConfig):
    dataset_name: str = "allenai/c4"
    data_dir: str | None = "en"
    batch_size: PositiveInt = 512
    sequence_length: PositiveInt = 1024
    per_device_train_batch_size: PositiveInt = 5
    world_size: int = 10  # TODO
    rank: int = 1  # TODO
    dataset_class: str | None = None
    vali_fraction: float = 0.1


class MoECfg(BaseConfig):
    my_expert_group_id: int = 1
    dense_to_moe: bool = True
    interleave: bool = True
    noise: bool = False
    noise_std: float = 0.1
    num_experts: PositiveInt = 8
    num_experts_per_tok: PositiveInt = 2
    partial_topk: PositiveInt = 1
    full_topk: PositiveInt = 2
    aux_load_balance: bool = True
    router_aux_loss_coef: float = 1.0
    partial_moe: bool = True
    num_worker_groups: PositiveInt = 2
    rotate_expert: bool = False
    expert_rotate_interval: PositiveInt | None = None


class OptimizerCfg(BaseConfig):
    lr: float = 2e-5
    outer_lr: float = 0.7
    outer_momentum: float = 0.9


class ParallelismCfg(BaseConfig):  # parallelism for local training
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


class ScheduleCfg(BaseConfig):
    warmup_steps: PositiveInt = 600
    total_steps: PositiveInt = 88_000
    # Derived defaults for intervals set in Checkpoint/Logging via global_opt_interval


class CheckpointCfg(BaseConfig):
    resume_from_ckpt: bool = True
    base_checkpoint_path: Path = Path("checkpoints/miner")
    checkpoint_path: Path | None = None
    checkpoint_interval: PositiveInt | None = None
    full_validation_interval: PositiveInt | None = None
    checkpoint_topk: PositiveInt = 5
    validator_checkpoint_path: Path = Path("validator_checkpoint")


class LoggingCfg(BaseConfig):
    log_wandb: bool = False
    wandb_project_name: str = "test-moe"
    wandb_resume: bool = False
    wandb_full_id: str = "oo2vn2v4"
    wandb_partial_id: list[str | None] = ["3q8mckj8"]
    base_metric_path: Path = Path("metrics")
    metric_path: Path | None = None
    metric_interval: PositiveInt | None = None


class ValidatorCheckpointCfg(CheckpointCfg):
    base_checkpoint_path: Path = Path("checkpoints/validator")
    miner_submission_path: Path = Path("checkpoints/validator/miner_submission")


class OwnerCheckpointCfg(CheckpointCfg):
    base_checkpoint_path: Path = Path("checkpoints/owner")


class OwnerCfg(BaseConfig):
    app_ip: str = "0.0.0.0"
    app_port: int = 7000


class TaskCfg(BaseConfig):
    data: DataCfg = DataCfg()
    expert_group_id: int = 0
    expert_group_name: str = "exp_math"
    base_path: Path = Path("expert_groups")
    path: Path | None = None


# ---------------------------
# Top-level config
# ---------------------------
class WorkerConfig(BaseConfig):
    """
    Centralized training/eval configuration for mycelia runs.
    """

    run: RunCfg = RunCfg()
    chain: ChainCfg = ChainCfg()
    model: ModelCfg = ModelCfg()
    moe: MoECfg = MoECfg()
    ckpt: CheckpointCfg = CheckpointCfg()
    sched: ScheduleCfg = ScheduleCfg()
    log: LoggingCfg = LoggingCfg()
    opt: OptimizerCfg = OptimizerCfg()
    cycle: CycleCfg = CycleCfg()
    task: TaskCfg = TaskCfg()
    # -----------------------
    # Derivations & hygiene
    # -----------------------

    @classmethod
    def create(cls, **data):
        self = cls(**data)  # calls __init__
        return self._post_init()

    def _post_init(self):
        # If an existing config exists, bump run_name when the configs don't match.
        config_path = os.path.join(self.ckpt.checkpoint_path, "config.yaml")

        while os.path.exists(config_path):
            logger.info(f"found existing config path {config_path}")
            with open(config_path, encoding="utf-8") as f:
                existing_config_dict = yaml.safe_load(f)

            new_self = self.bump_run_name_if_diff(existing_config_dict)

            if new_self is self:
                # No overwrite; bump done or same config
                return self

            # Overwrite happened → new instance returned
            self = new_self

            # Update path and continue loop
            config_path = os.path.join(self.ckpt.checkpoint_path, "config.yaml")

        return self

    def __init__(self, **data):
        """
        Initialize, validate derived fields, and auto-bump `run_name` if an on-disk
        config exists and differs.
        """
        super().__init__(**data)
        # check wallet data
        self._fill_wallet_data()

        # Recompute dependent fields that rely on CUDA availability or run_name.
        self._refresh_paths()

        # get task specific config
        self.update_by_task()

        # === create checkpoint directory ===
        os.makedirs(self.task.base_path, exist_ok=True)
        os.makedirs(self.task.path, exist_ok=True)
        os.makedirs(self.ckpt.base_checkpoint_path, exist_ok=True)
        os.makedirs(self.ckpt.checkpoint_path, exist_ok=True)
        os.makedirs(self.log.base_metric_path, exist_ok=True)
        os.makedirs(self.ckpt.validator_checkpoint_path, exist_ok=True)

    def _refresh_paths(self) -> None:
        self.ckpt.base_checkpoint_path = self.run.root_path / self.ckpt.base_checkpoint_path
        self.ckpt.checkpoint_path = (
            self.ckpt.base_checkpoint_path / self.chain.coldkey_name / self.chain.hotkey_name / self.run.run_name
        )
        self.ckpt.validator_checkpoint_path = self.ckpt.base_checkpoint_path / self.ckpt.validator_checkpoint_path

        self.log.base_metric_path = self.run.root_path / self.log.base_metric_path
        self.log.metric_path = self.log.base_metric_path / f"{self.run.run_name}.csv"

        self.task.base_path = self.run.root_path / self.task.base_path
        self.task.path = self.task.base_path / self.task.expert_group_name

        if hasattr(self, "vali"):
            self.ckpt.miner_submission_path = self.run.root_path / self.ckpt.miner_submission_path

    def _fill_wallet_data(self):
        wallet = bittensor.wallet(name=self.chain.coldkey_name, hotkey=self.chain.hotkey_name)
        subtensor = bittensor.subtensor(network=self.chain.network)
        try:
            self.chain.hotkey_ss58 = wallet.hotkey.ss58_address
            self.chain.coldkey_ss58 = wallet.coldkeypub.ss58_address
            self.chain.uid = subtensor.metagraph(netuid=self.chain.netuid).hotkeys.index(self.chain.hotkey_ss58)
        except bittensor.KeyFileError as e:
            logger.warning(
                f"Cannot find the wallet key by name coldkey name: {self.chain.coldkey_name}, hotkey name: {self.chain.hotkey_name}, please make sure it has been set correctly if you are reloading from a  or use --hotkey_name & --coldkey_name when you are creating a config file from a template.",  # noqa: E501
                error=str(e),
            )
            return

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
            extra = b_keys - a_keys
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
        if isinstance(a, list | tuple) and isinstance(b, list | tuple):
            if len(a) != len(b):
                logger.info(f"Length mismatch at '{path}': existing {len(b)} vs new {len(a)}")
                return False
            for i, (ai, bi) in enumerate(zip(a, b, strict=False)):
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
    def same_as(self, other: dict) -> bool:
        """
        Return True if configs are equivalent (deep comparison).
        `other` is a dict (possibly nested) from the same schema.
        """
        self_dict = self.to_dict()
        return self._deep_compare(self_dict, other)

    def bump_run_name_if_diff(self, other: dict) -> bool:
        """
        If configs differ, bump `run_name` to the next version and refresh paths.

        Returns
        -------
        bool
            True if bumped, False if configs were the same.
        """
        if self.same_as(other):
            return self

        choice = (
            input(f"Configurations differ from an existing run. " f"Overwrite '{self.run.run_name}'? (y/n): ")
            .strip()
            .lower()
        )

        if choice in {"y", "yes"}:
            logger.info("Overwriting existing run_name.")
            return WorkerConfig(**other)

        else:
            logger.info("User declined to overwrite existing run_name.")

        self.run.run_name = self._bump_run_name(self.run.run_name)
        self._refresh_paths()
        logger.info(f"Bumped run_name to {self.run.run_name} due to config differences.")
        return self

    # ---- Get config based on task ----
    def update_by_task(self, expert_group_name: str | None = None):
        """
        Start from BaseConfig defaults, then apply YAML overrides.
        """
        if expert_group_name:
            self.task.expert_group_name = expert_group_name
            self._refresh_paths()  # base path may change

        self.task = TaskCfg.from_path(self.task.path / "config.yaml")  # type: ignore
        self._refresh_paths()  # base path may change

    # ---- Persistence ----
    def write(self) -> None:
        """
        Persist this config to `<checkpoint_path>/`.

        Creates the directory if it does not exist.
        """
        data = self.model_dump()
        data = convert_to_str(data)

        os.makedirs(self.ckpt.checkpoint_path, exist_ok=True)
        target = os.path.join(self.ckpt.checkpoint_path, "config.yaml")

        with fsspec.open(target, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False)

        logger.info(f"Wrote config to {target}")


class MinerConfig(WorkerConfig):
    role: str = "miner"
    local_par: ParallelismCfg = ParallelismCfg()

    def __init__(self, **data):
        """
        Initialize, validate derived fields, and auto-bump `run_name` if an on-disk
        config exists and differs.
        """
        super().__init__(**data)
        os.makedirs(self.ckpt.validator_checkpoint_path, exist_ok=True)

    @model_validator(mode="after")
    def _derive_all(self):
        if self.local_par.gradient_accumulation_steps == 0:
            # ceil(batch_size / per_device_train_batch_size)
            g = math.ceil(self.task.data.batch_size / self.task.data.per_device_train_batch_size)
            self.local_par.gradient_accumulation_steps = max(1, int(g))

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


class ValidatorConfig(WorkerConfig):
    role: str = "validator"
    ckpt: ValidatorCheckpointCfg = ValidatorCheckpointCfg()


class OwnerConfig(WorkerConfig):
    role: str = "owner"
    owner: OwnerCfg = OwnerCfg()
    ckpt: OwnerCheckpointCfg = OwnerCheckpointCfg()


def parse_args():
    parser = argparse.ArgumentParser(description="Train mycelia with config")
    parser.add_argument(
        "--path",
        type=str,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--get_template",
        choices=["miner", "validator"],
        help="Get a template config for miner or validator",
    )
    parser.add_argument(
        "--hotkey_name",
        type=str,
        help="Optional, wallet hotkey name for creating the folder path to the template file.",
    )
    parser.add_argument(
        "--coldkey_name",
        type=str,
        help="Optional, wallet coldkey name for creating the folder path to the template file.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Optional, run name for creating the folder path to the template file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.get_template:
        config_dict = {}
        if args.run_name:
            config_dict["run"] = {"run_name": args.run_name}

        if args.hotkey_name:
            config_dict["chain"] = {"hotkey_name": args.hotkey_name}

        if args.coldkey_name:
            if "chain" not in config_dict:
                config_dict["chain"] = {}
            config_dict["chain"]["coldkey_name"] = args.coldkey_name

        if args.get_template == "validator":
            ValidatorConfig(**config_dict).write()

        if args.get_template == "miner":
            MinerConfig(**config_dict).write()
