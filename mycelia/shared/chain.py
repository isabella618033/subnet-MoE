from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import bittensor
from pydantic import BaseModel, ConfigDict, Field

from mycelia.shared.app_logging import structlog
from mycelia.shared.checkpoint import ModelMeta, delete_old_checkpoints
from mycelia.shared.client import download_model
from mycelia.shared.config import WorkerConfig
from mycelia.shared.schema import verify_message

logger = structlog.get_logger(__name__)

# Global lock for subtensor WebSocket access to prevent concurrent recv calls
_subtensor_lock = threading.Lock()


# --- Info gather ---
def get_active_validator_info() -> dict | None:
    raise NotImplementedError


def get_active_miner_info():
    raise NotImplementedError


# --- Status structure and submission (for miner validator communication)---
class WorkerChainCommit(BaseModel):
    ip: str
    port: int
    active: bool
    stake: float
    validator_permit: bool


class ValidatorChainCommit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    model_hash: str | None = Field(default=None, alias="h")
    global_ver: int | None = Field(default=None, alias="v")
    expert_group: int | None = Field(default=None, alias="e")
    miner_seed: int | None = Field(default=None, alias="s")
    block: int | None = Field(default=None, alias="b")


class MinerChainCommit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    block: int = Field(alias="b")
    expert_group: int | None = Field(default=None, alias="e")
    model_hash: str | None = Field(default=None, alias="h")
    global_ver: int | None = Field(default=0, alias="v")
    inner_opt: int | None = Field(default=0, alias="i")


def commit_status(
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    subtensor: bittensor.Subtensor,
    status: ValidatorChainCommit | MinerChainCommit,
) -> None:
    """
    Commit the worker status to chain.

    If encrypted=False:
        - Uses subtensor.set_commitment (plain metadata, immediately visible).

    If encrypted=True:
        - Timelock-encrypts the status JSON using Drand.
        - Stores it via the Commitments pallet so it will be revealed later
          when the target Drand round is reached.

    Assumes:
        - config.chain.netuid: subnet netuid
        - config.chain.timelock_rounds_ahead: how many Drand rounds in the future
          you want the data to be revealed (fallback to 200 if missing).
    """
    # Serialize status first; same input for both plain + encrypted paths
    data_dict = status.model_dump(by_alias=True)

    data = json.dumps(data_dict)

    subtensor.set_commitment(wallet=wallet, netuid=config.chain.netuid, data=data, raise_error=True)

    logger.info("Committed status to chain", status=data_dict)
    return data_dict


def get_chain_commits(
    config: WorkerConfig, subtensor: bittensor.Subtensor, wait_to_decrypt: bool = False
) -> tuple[WorkerChainCommit, bittensor.Neuron]:
    all_commitments = subtensor.get_all_commitments(netuid=config.chain.netuid)
    metagraph = subtensor.metagraph(netuid=config.chain.netuid)

    parsed = []

    for hotkey, commit in all_commitments.items():
        uid = metagraph.hotkeys.index(hotkey)

        try:
            status_dict = json.loads(commit)

            chain_commit = (
                ValidatorChainCommit.model_validate(status_dict)
                if "miner_seed" in status_dict or "s" in status_dict
                else MinerChainCommit.model_validate(status_dict)
            )

        except Exception as e:
            chain_commit = None

        parsed.append((chain_commit, metagraph.neurons[uid]))

    return parsed


# --- setup chain worker ---
def setup_chain_worker(config):
    wallet = bittensor.Wallet(name=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)
    subtensor = bittensor.Subtensor(network=config.chain.network)
    serve_axon(
        config=config,
        wallet=wallet,
        subtensor=subtensor,
    )
    return wallet, subtensor


def serve_axon(config: WorkerConfig, wallet: bittensor.Wallet, subtensor: bittensor.Subtensor):
    axon = bittensor.Axon(wallet=wallet, external_port=config.chain.port, ip=config.chain.ip)
    axon.serve(netuid=348, subtensor=subtensor)


# --- Chain weight submission ---
def submit_weight() -> str:
    raise NotImplementedError


# --- Get model from chain ---
def scan_chain_for_new_model(
    current_model_meta: ModelMeta | None,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
) -> tuple[bool, list[dict]]:
    """
    Returns:
        should_download: True if a newer model (by version) is available and a majority
                         agree on the model_hash among those newer entries.
        download_meta:   list of dicts with fields: uid, ip, port, model_hash, model_version
                         filtered to entries that (a) are newer and (b) match the majority hash.
    """
    commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(config, subtensor)

    max_model_meta = ModelMeta(
        global_ver=max(
            (c.global_ver for c, n in commits if c is not None and getattr(c, "global_ver", None) is not None),
            default=0,
        )
    )

    if current_model_meta is not None:
        logger.info(
            "Scan chain: Max model version on chain",
            max_model_version_on_chain=max_model_meta,
        )
        logger.info(
            "Scan chain: Local model version",
            current_model_version=current_model_meta,
        )
        max_model_meta = max(max_model_meta, current_model_meta)

    # 0) Download only from validator
    # commits = [(c, n) for c, n in commits if n.validator_permit]
    commits = [(c, n) for c, n in commits if getattr(c, "miner_seed", False)]
    commits = [(c, n) for c, n in commits if getattr(c, "model_hash", False)]

    # 1) collect candidates that are newer than the current version
    most_updated_commits = []  # type: ignore
    for c, n in commits:
        raw_ver = getattr(c, "global_ver", None)
        global_ver = raw_ver if isinstance(raw_ver, int) else 0
        if ModelMeta(global_ver=global_ver) >= max_model_meta:
            most_updated_commits.append((c, n))

    if len(most_updated_commits) == 0:
        return False, []

    # 2) majority filter by model_hash among the newer candidates
    hash_counts = Counter([c.model_hash for c, n in most_updated_commits if getattr(c, "model_hash", False)])
    majority_hash, _count = hash_counts.most_common(1)[0]

    filtered_commits = [(c, n) for c, n in most_updated_commits if getattr(c, "model_hash", False) == majority_hash]
    if not filtered_commits:
        return False, []

    # 3) prepare download_meta for each entry (uid, ip, port, model_hash, model_version)
    download_meta = []
    for commit, neuron in filtered_commits:
        # Only include entries with reachable metadata
        try:
            download_meta.append(
                {
                    "uid": neuron.uid,
                    "ip": neuron.axon_info.ip,
                    "port": neuron.axon_info.port,
                    "model_hash": commit.model_hash,
                    "global_ver": commit.global_ver,
                    "target_hotkey_ss58": neuron.hotkey,
                }
            )

        except Exception:
            logger.info("Cannot append commit", commit=commit)

    # should_download if there is at least one newer entry agreeing on a majority hash
    should_download = len(download_meta) > 0

    return should_download, download_meta


def fetch_model_from_chain(
    current_model_meta: ModelMeta | None,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    expert_group_ids: list = [],
) -> dict | None:
    should_download, download_metas = scan_chain_for_new_model(current_model_meta, config, subtensor)

    logger.info("Fetching model from chain", should_download=should_download, download_metas=download_metas)

    if should_download and download_metas:
        download_success = False
        retries = 0
        max_retries = 3
        base_delay_s = 5  # backoff base

        while (not download_success) and (retries < max_retries):
            for download_meta in download_metas:
                logger.info(f"Downloading from chain: uid = {download_meta['uid']}", download_meta=download_meta)

                # Resolve URL if not provided; fall back to ip/port + default route
                url = download_meta.get("url")
                if not url:
                    ip = download_meta.get("ip")
                    port = download_meta.get("port")
                    # Best-effort defaults; customize if your API differs
                    protocol = getattr(getattr(config, "miner", object()), "protocol", "http")
                    if ip and port:
                        url = f"{protocol}://{ip}:{port}/get-checkpoint"
                    else:
                        logger.warning("Skipping meta without URL or ip:port: %s", download_meta)
                        continue

                out_folder = Path(config.ckpt.validator_checkpoint_path) / (
                    f"uid_{download_meta.get('uid')}_hotkey_{download_meta.get('target_hotkey_ss58')}_globalver_{download_meta.get('global_ver')}"
                )

                out_folder.mkdir(parents=True, exist_ok=True)

                if len(expert_group_ids) == 0:
                    expert_group_ids = [config.task.expert_group_id, "shared"]

                for expert_group_id in expert_group_ids:
                    out_file = (
                        f"model_expgroup_{expert_group_id}.pt"
                        if isinstance(expert_group_id, int)
                        else "model_shared.pt"
                    )
                    out_path = out_folder / out_file
                    try:
                        download_model(
                            url=url,
                            my_hotkey=wallet.hotkey,  # type: ignore
                            target_hotkey_ss58=download_meta["target_hotkey_ss58"],
                            block=subtensor.block,
                            expert_group_id=expert_group_id,
                            token=getattr(config.cycle, "token", ""),
                            out_dir=out_path,
                        )
                        # If download_model doesn't raise, consider it a success
                        download_success = True
                        current_model_version = download_meta["global_ver"]
                        current_model_hash = download_meta["model_hash"]
                        logger.info(
                            "✅ Downloaded checkpoint",
                            out_path=out_path,
                            current_model_version=current_model_version,
                            current_model_hash=current_model_hash,
                        )

                        delete_old_checkpoints(
                            checkpoint_path=Path(config.ckpt.validator_checkpoint_path),
                            topk=config.ckpt.checkpoint_topk,
                        )

                        return download_meta
                    except Exception as e:
                        logger.warning("Download failed", url, e)
                        traceback.print_exc()

            if not download_success:
                retries += 1
                if retries < max_retries:
                    delay = base_delay_s * (2 ** (retries - 1))
                    logger.info("Retrying", delay=delay, retries=retries + 1, max_retries=max_retries)
                    time.sleep(delay)

        if not download_success:
            logger.error(f"❌ All download attempts failed after {retries} retries.")

            return None
