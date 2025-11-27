from __future__ import annotations

import json
from typing import Dict, Optional

import bittensor as bt
from pydantic import BaseModel

from mycelia.shared.config import WorkerConfig

# --- Info gather ---
def get_active_validator_info() -> Optional[Dict]:
    raise NotImplemented


def get_active_miner_info():
    raise NotImplemented


# --- Status structure and submission (for miner validator communication)---
class WorkerStatus(BaseModel):
    ip: str
    port: int
    active: bool
    stake: float
    validator_permit: bool


class ValidatorStatus(BaseModel):
    model_hash: str | None = None
    model_version: int | None = None
    expert_group: int | None = None  # block
    miner_seed: int | None = None


class MinerStatus(BaseModel):
    expert_group: int | None = None


def commit_status(
    config: WorkerConfig,
    wallet: bt.Wallet,
    subtensor: bt.Subtensor,
    status: ValidatorStatus | MinerStatus,
):
    subtensor.set_commitment(wallet=wallet, netuid=config.chain.netuid, data=status.model_dump_json())


def get_status(config: WorkerConfig, subtensor: bt.Subtensor):
    all_commitments = subtensor.get_all_commitments(netuid=config.chain.netuid)
    metagraph = subtensor.metagraph(netuid=config.chain.netuid)
    parsed: Dict[str, WorkerStatus] = {}

    for hotkey, commit in all_commitments.items():
        uid = metagraph.hotkeys.index(hotkey)
        status_dict = json.loads(commit)

        try:
            status = (
                ValidatorStatus.model_validate(status_dict)
                if "model_hash" in status_dict
                else MinerStatus.model_validate(status_dict)
            )
        except Exception:
            status = None

        parsed[hotkey] = {"status": status, "neuron": metagraph.neurons[uid]}

    return parsed


def serve_axon(config: WorkerConfig, wallet: bt.Wallet, subtensor: bt.Subtensor):
    axon = bt.Axon(wallet=wallet, external_port=config.chain.port, ip=config.chain.ip)
    axon.serve(netuid=348, subtensor=subtensor)


# --- Chain weight submission ---
def submit_weight() -> str:
    raise NotImplemented
