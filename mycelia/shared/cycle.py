import os
import re
import sys
import time
import json
import hashlib
import logging
import signal
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

from collections import Counter
import bittensor 
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError as ReqConnectionError

import bittensor 
from mycelia.shared.chain import serve_axon, get_status, commit_status, MinerStatus
from mycelia.shared.checkpoint import get_resume_info, delete_old_checkpoints
from mycelia.shared.config import MinerConfig, parse_args, BaseConfig, ValidatorConfig  
from mycelia.miner.client import submit_model, download_model           
from mycelia.shared.app_logging import structlog, configure_logging
from mycelia.validator.evaluator import MinerEvalJob

configure_logging()
logger = structlog.get_logger(__name__)

def should_submit_model(
    config: MinerConfig,
    subtensor: bittensor.Subtensor,
    last_submission_block: int,
) -> Tuple[bool, int]:     
    schedule = get_validation_schedule(config, subtensor)
    phase_status = get_phase_status(schedule, subtensor.block)
    should_submit = phase_status == 'submission' and (subtensor.block - last_submission_block > config.cycle.submission_rate_limit) 
    block_till = schedule['submission_start_block'] - subtensor.block 
    if block_till < 0 and should_submit == False:
        block_till += config.cycle.validation_period 
    logger.info("should_submit_model", should_start = should_submit, block_till = block_till)
    return should_submit, block_till

def should_start_validation(
    config: ValidatorConfig,
    subtensor: bittensor.Subtensor
) -> Tuple[bool, int]:
    
    schedule = get_validation_schedule(config, subtensor)
    phase_status = get_phase_status(schedule, subtensor.block)
    should_start = phase_status == 'validating'
    block_till = schedule['interval_start_block'] - subtensor.block  
    if block_till < 0 and should_start == False:
        block_till += config.cycle.validation_period
        
    logger.info("should_start_validation", should_start = should_start, block_till = block_till)
    return phase_status == 'validating', block_till
    
def search_model_submission_destination(
    wallet: bittensor.wallet,
    config: MinerConfig, 
    subtensor: bittensor.Subtensor
) -> bittensor.Axon:
    
    validator_miner_assignment = get_validator_miner_assignment(config, subtensor)
    
    for validator, miners in validator_miner_assignment.items():
        if wallet.hotkey.ss58_address in miners:
            assigned_validator_hotkey = validator
            break

    metagraph = subtensor.metagraph(netuid = config.chain.netuid)
    uid = metagraph.hotkeys.index(assigned_validator_hotkey)
    return metagraph.axons[uid]

def scan_for_new_model(
    current_model_version: int,
    current_model_hash: str,
    config: MinerConfig,
    subtensor: bittensor.subtensor,
) -> Tuple[bool, List[dict]]:
    """
    Returns:
        should_download: True if a newer model (by version) is available and a majority
                         agree on the model_hash among those newer entries.
        download_meta:   list of dicts with fields: uid, ip, port, model_hash, model_version
                         filtered to entries that (a) are newer and (b) match the majority hash.
    """
    status_map = get_status(config, subtensor)

    # ---- helpers ------------------------------------------------------------
    def extract_entry_fields(entry: Any):
        """Normalize entry into (status_obj, neuron_obj)."""
        if isinstance(entry, dict) and ("status" in entry or "neuron" in entry):
            return entry.get("status", None), entry.get("neuron", None)
        # Otherwise assume the value itself is a status object (older code paths)
        return entry, None

    def get_version_and_hash(status_obj) -> Tuple[Optional[int], Optional[str]]:
        mv = getattr(status_obj, "model_version", None)
        mh = getattr(status_obj, "model_hash", None)
        return mv, mh

    def get_uid(neuron_obj) -> Optional[int]:
        return getattr(neuron_obj, "uid", None) if neuron_obj is not None else None

    # ------------------------------------------------------------------------

    # 1) collect candidates that are newer than the current version
    newer_candidates = []  # (model_version, model_hash, uid, ip, port)
    for _hotkey, entry in status_map.items():
        status_obj, neuron_obj = extract_entry_fields(entry)
        if status_obj is None:
            continue
        mv, mh = get_version_and_hash(status_obj)
        if mv is None or mh is None:
            continue
        if mv > current_model_version:
            uid = get_uid(neuron_obj)
            ip = neuron_obj.axon_info.ip
            port = neuron_obj.axon_info.port
            newer_candidates.append((mv, mh, uid, ip, port))

    if not newer_candidates:
        return False, []

    # 2) majority filter by model_hash among the newer candidates
    hash_counts = Counter(mh for (_mv, mh, _uid, _ip, _port) in newer_candidates)
    majority_hash, _count = hash_counts.most_common(1)[0]

    filtered = [
        (mv, mh, uid, ip, port)
        for (mv, mh, uid, ip, port) in newer_candidates
        if mh == majority_hash
    ]
    if not filtered:
        return False, []

    # 3) prepare download_meta for each entry (uid, ip, port, model_hash, model_version)
    download_meta = []
    for mv, mh, uid, ip, port in filtered:
        # Only include entries with reachable metadata
        download_meta.append(
            {
                "uid": uid,
                "ip": ip,
                "port": port,
                "model_hash": mh,
                "model_version": mv,
            }
        )

    # should_download if there is at least one newer entry agreeing on a majority hash
    should_download = len(download_meta) > 0

    return should_download, download_meta

def setup_chain_worker(
    config
):
    wallet = bittensor.wallet(name = config.chain.coldkey_name, hotkey = config.chain.hotkey_name)
    subtensor = bittensor.subtensor(network = config.chain.network) 
    serve_axon(
        config = config,
        wallet = wallet,
        subtensor = subtensor,
    )
    return wallet, subtensor

def h256_int(*parts: Any) -> int:
    """Deterministic 256-bit hash -> int."""
    m = hashlib.sha256()
    for p in parts:
        m.update(str(p).encode("utf-8"))
        m.update(b"\x00")  # separator
    return int.from_bytes(m.digest(), "big")

def assign_miners_to_validators(
    validators: Dict[str, Any],  # {validator_id: seed}
    miners: List[str],
) -> Dict[str, List[str]]:
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
    def validator_prefs(mid: str) -> List[str]:
        return sorted(
            v_ids,
            key=lambda vid: h256_int("preference", mid, validators[vid], combined_seed),
            reverse=True,
        )

    # --- 4) Assign miners evenly, respecting capacities
    assignment: Dict[str, List[str]] = {vid: [] for vid in v_ids}
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

def get_validator_miner_assignment(config: BaseConfig, subtensor: bittensor.Subtensor):
    status = get_status(config, subtensor)

    validator_seeds: Dict[str, int] = {
        hotkey: entry["status"].miner_seed
        for hotkey, entry in status.items()
        if entry.get("status")
        and getattr(entry["status"], "expert_group", None) == config.moe.my_expert_group_id
        and getattr(entry["status"], "miner_seed", None) is not None
    }

    miners: List[str] = [
        hk for hk, e in status.items()
        if isinstance(e.get("status"), MinerStatus)
        and getattr(e["status"], "expert_group", None) == config.moe.my_expert_group_id
    ]

    return assign_miners_to_validators(validator_seeds, miners) # type: ignore

def get_validation_schedule(config: BaseConfig, subtensor: bittensor.Subtensor, block = None, last = False) -> Dict[str, int]:

    if block == None:
        block = subtensor.block

    if last: 
        block = block - config.cycle.validation_period

    interval_start_block = (block // config.cycle.validation_period) * config.cycle.validation_period + 1
    interval_end_block = (block // config.cycle.validation_period + 1) * config.cycle.validation_period
    submission_start_block = interval_end_block - config.cycle.submission_offset
    validation_end_block = interval_start_block + config.cycle.validation_offset

    return {
        'interval_start_block': interval_start_block,
        'interval_end_block': interval_end_block,
        'submission_start_block': submission_start_block,
        'validation_end_block': validation_end_block
    }  

def get_phase_status(schedule: dict, block: int) -> str:
    """
    Determine current phase based on block schedule.
    
    Returns:
        str: one of ["training", "submission", "waiting"]
    """
    start = schedule["interval_start_block"]
    validation = schedule["validation_end_block"]
    submission = schedule["submission_start_block"]
    end = schedule["interval_end_block"]

    if start <= block < validation:
        return "validating"
    elif validation <= block < submission:
        return "training"
    elif submission <= block <= end:
        return "submission"
    else:
        return "not_regcognised"

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

    return meta

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

def gather_validation_job(
        config: ValidatorConfig,
        subtensor: bittensor.Subtensor,
        step: int
) -> List[MinerEvalJob]:

    validation_schedule = get_validation_schedule(config, subtensor, last = True)
    validator_miner_assignment = get_validator_miner_assignment(config, subtensor)
    miner_assignment = validator_miner_assignment[config.chain.hotkey_ss58] 
    miner_submission_files = load_submission_files(str(config.vali.miner_submission_path))

    miner_jobs = []
    for file_name, submission_meta in miner_submission_files.items():
        if (
            submission_meta['hotkey'] in miner_assignment 
            and get_phase_status(validation_schedule, submission_meta['block']) == 'submission'
        ):
            miner_jobs.append(
                MinerEvalJob(
                    uid = submission_meta['uid'],
                    hotkey = submission_meta['hotkey'],
                    model_path = file_name,
                    step = step
                )
            )

    return miner_jobs