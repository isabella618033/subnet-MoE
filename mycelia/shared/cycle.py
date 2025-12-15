import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import bittensor
import requests
from pydantic import BaseModel

from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import (
    MinerChainCommit,
    ValidatorChainCommit,
    WorkerChainCommit,
    get_chain_commits,
    serve_axon,
    _subtensor_lock,
)
from mycelia.shared.config import MinerConfig, ValidatorConfig, WorkerConfig
from mycelia.shared.helper import h256_int, parse_dynamic_filename
from mycelia.validator.evaluator import MinerEvalJob

configure_logging()
logger = structlog.get_logger(__name__)


class PhaseResponse(BaseModel):
    block: int
    cycle_length: int  # how long is one cycle
    cycle_index: int  # which cycle are we in
    cycle_block_index: int  # how far in block are we into a cycle
    phase_name: str  # what is the name of the current phase
    phase_index: int  # what is the id of the phase
    phase_start_block: int  # the start block of the phase
    phase_end_block: int  # the end block of the phase
    blocks_into_phase: int  # how far in block are we in the current phase
    blocks_remaining_in_phase: int  # how manuy block left in the phase


@dataclass
class PhaseNames:
    distribute: str = "Distribute"  # miner download from validator
    train: str = "Train"  # miner trian
    commit: str = "Commit"  # miner commit hash and  vlaidators commit seed
    submission: str = "Submission"  # submit model
    validate: str = "Validate"  # validator validate
    merge: str = "Merge"  # validator merge


def wait_till(config: MinerConfig, phase_name: PhaseNames, poll_fallback_seconds: int = 5):
    should_submit = False
    logger.info(f"<{phase_name}> waiting to begin...")
    while not should_submit:
        should_submit, blocks_till, phase_response = should_act(config, phase_name)
        if should_submit is False and blocks_till > 0:
            sleep_sec = min(blocks_till, max(poll_fallback_seconds, blocks_till / 3)) * 12

            check_time = datetime.now() + timedelta(seconds=sleep_sec)
            check_time_str = check_time.strftime("%H:%M:%S")

            logger.info(f"<{phase_name}> to begin in {blocks_till} blocks, check again at {check_time_str}")
            time.sleep(sleep_sec)

    logger.info(f"<{phase_name}> has started, {phase_response.blocks_remaining_in_phase} blocks left in phase.")
    return should_submit, phase_response.phase_end_block


def should_act(config: MinerConfig, phase_name: PhaseNames) -> tuple[bool, int, int]:
    phase_response: PhaseResponse = get_phase(config)
    should_submit = phase_response.phase_name == phase_name
    blocks_till = get_blocks_until_next_phase(config)[phase_name]
    return should_submit, blocks_till, phase_response


def search_model_submission_destination(
    wallet: bittensor.Wallet, config: MinerConfig, subtensor: bittensor.Subtensor
) -> bittensor.Axon:
    validator_miner_assignment = get_validator_miner_assignment(config, subtensor)

    for validator, miners in validator_miner_assignment.items():
        if wallet.hotkey.ss58_address in miners:
            assigned_validator_hotkey = validator
            break


    metagraph = subtensor.metagraph(netuid=config.chain.netuid)
    uid = metagraph.hotkeys.index(assigned_validator_hotkey)
    return metagraph.axons[uid]


def setup_chain_worker(config):
    wallet = bittensor.Wallet(name=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)
    subtensor = bittensor.Subtensor(network=config.chain.network)
    serve_axon(
        config=config,
        wallet=wallet,
        subtensor=subtensor,
    )
    return wallet, subtensor


def assign_miners_to_validators(
    validators: dict[str, Any],  # {validator_id: seed}
    miners: list[str],
) -> dict[str, list[str]]:
    n_v = len(validators)
    n_m = len(miners)

    if n_v == 0:
        raise ValueError("No validators provided")

    # --- 0) Combined seed (hash of all validator seeds)
    combined_seed_str = "".join(str(validators[v]) for v in sorted(validators.keys()))
    combined_seed = hashlib.sha256(combined_seed_str.encode()).hexdigest()

    # --- 1) Balanced capacities
    base = n_m // n_v
    rem = n_m % n_v
    v_ids = list(validators.keys())

    ranked_for_bonus = sorted(
        v_ids,
        key=lambda vid: h256_int("cap_bonus", validators[vid], combined_seed),
        reverse=True,
    )
    capacities = {vid: base for vid in v_ids}
    for vid in ranked_for_bonus[:rem]:
        capacities[vid] += 1

    # --- 2) Deterministic miner order seeded by combined validator seed
    miners_sorted = sorted(miners, key=lambda mid: h256_int("miner_order", mid, combined_seed))

    # --- 3) Preference per miner (based on validator seed + combined seed)
    def validator_prefs(mid: str) -> list[str]:
        return sorted(
            v_ids,
            key=lambda vid: h256_int("preference", mid, validators[vid], combined_seed),
            reverse=True,
        )

    # --- 4) Assign miners evenly, respecting capacities
    assignment: dict[str, list[str]] = {vid: [] for vid in v_ids}
    for mid in miners_sorted:
        prefs = validator_prefs(mid)
        for vid in prefs:
            if capacities[vid] > 0:
                assignment[vid].append(mid)
                capacities[vid] -= 1
                break
        else:
            # Should never happen if capacities sum == len(miners)
            assignment[prefs[-1]].append(mid)

    return assignment


def get_combined_validator_seed(config: WorkerConfig, subtensor: bittensor.Subtensor) -> str:
    """
    Deterministically combine validator seeds into a single hex string.

    We sort validator IDs so the result is independent of dict iteration order.
    """
    commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(config, subtensor)

    validator_seeds = get_validator_seed_from_commit(config, commits)
    if not validator_seeds:
        raise ValueError("No validators provided")

    combined_seed_str = "".join(str(validator_seeds[v]) for v in sorted(validator_seeds.keys()))
    return hashlib.sha256(combined_seed_str.encode()).hexdigest()


def get_validator_miner_assignment(config: WorkerConfig, subtensor: bittensor.Subtensor):
    commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(config, subtensor)
    validator_seeds = get_validator_seed_from_commit(config, commits)
    miners = get_miners_from_commit(config, commits)
    return assign_miners_to_validators(validator_seeds, miners)  # type: ignore


def get_validator_seed_from_commit(config, commits):
    validator_seeds: dict[str, int] = {
        neuron.hotkey: commit.miner_seed
        for commit, neuron in commits
        if isinstance(commit, ValidatorChainCommit)
        and getattr(commit, "expert_group", None) == config.task.expert_group_id
    }
    return validator_seeds


def get_miners_from_commit(config, commits):
    miners: list[str] = [
        neuron.hotkey
        for commit, neuron in commits
        if isinstance(commit, MinerChainCommit) and getattr(commit, "expert_group", None) == config.task.expert_group_id
    ]

    return miners


def get_phase(config: WorkerConfig) -> PhaseResponse:
    """
    Determine current phase based on block schedule.

    Returns:
        str: one of ["training", "submission", "waiting"]
    """
    resp = requests.get(f"{config.cycle.owner_url}/get_phase")
    resp.raise_for_status()
    return PhaseResponse(**resp.json())


def get_blocks_until_next_phase(config: WorkerConfig) -> PhaseResponse:
    """
    Determine current phase based on block schedule.

    Returns:
        str: one of ["training", "submission", "waiting"]
    """
    resp = requests.get(f"{config.cycle.owner_url}/blocks_until_next_phase")
    resp.raise_for_status()
    return resp.json()

def get_blocks_from_previous_phase(config: WorkerConfig) -> PhaseResponse:
    """
    Determine current phase based on block schedule.

    Returns:
        str: one of ["training", "submission", "waiting"]
    """
    resp = requests.get(f"{config.cycle.owner_url}/previous_phase_blocks")
    resp.raise_for_status()
    return resp.json()


def load_submission_files(folder: str = "miner_submission"):
    """
    Scans a folder for .pt files and returns:
        { filename: {parsed key/values} }
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

    files_dict = {}
    for file_name in folder_path.glob("*.pt"):
        meta = parse_dynamic_filename(file_name.name)
        files_dict[file_name.name] = meta

    return files_dict

def gather_validation_job(config: ValidatorConfig, subtensor: bittensor.Subtensor, step: int) -> list[MinerEvalJob]:
    validator_miner_assignment = get_validator_miner_assignment(config, subtensor)
    miner_assignment = validator_miner_assignment.get(config.chain.hotkey_ss58, [])
    miner_submission_files = load_submission_files(str(config.ckpt.miner_submission_path))
    previous_phase_range = get_blocks_from_previous_phase(config)[PhaseNames.submission]

    hotkeys = subtensor.metagraph(netuid=config.chain.netuid).hotkeys
    miner_jobs = []
    for file_name, submission_meta in miner_submission_files.items():
        if submission_meta["hotkey"] in miner_assignment and submission_meta["block"] >= previous_phase_range[0] and submission_meta["block"] <= previous_phase_range[1] :
            miner_jobs.append(
                MinerEvalJob(
                    uid = hotkeys.index(submission_meta["hotkey"]),
                    hotkey=submission_meta["hotkey"],
                    model_path=config.ckpt.miner_submission_path / file_name,
                    step=step,
                )
            )

    return miner_jobs
