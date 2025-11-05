# checkpoint_manager.py
import os
import sys
import time
import json
import hashlib
import logging
import signal
from pathlib import Path
from typing import Optional, Tuple

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError as ReqConnectionError

import bittensor 
from mycelia.shared.chain import serve_axon, get_status
from mycelia.shared.checkpoint import get_resume_info
from mycelia.shared.config import MinerConfig, parse_args           
from mycelia.miner.client import submit_model, download_model           
from mycelia.shared.app_logging import structlog, configure_logging

configure_logging()
logger = structlog.get_logger(__name__)

def _should_submit() -> bool:
    return False

def scan_for_new_model(
    current_model_version: int,
    current_model_hash: str,
    config,
    subtensor
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

def main(config):
    wallet, subtensor = setup_chain_worker(config)

    current_model_version = 0
    current_model_hash = 'xxx'

    while True:
        try:
            # --------- DOWNLOAD PHASE ---------
            should_download, download_metas = scan_for_new_model(
                current_model_version, current_model_hash, config, subtensor
            )

            if should_download and download_metas:
                download_success = False
                retries = 0
                max_retries = 3
                base_delay_s = 5  # backoff base

                while (not download_success) and (retries < max_retries):
                    for download_meta in download_metas:
                        # Resolve URL if not provided; fall back to ip/port + default route
                        url = download_meta.get("url")
                        if not url:
                            ip = download_meta.get("ip")
                            port = download_meta.get("port")
                            # Best-effort defaults; customize if your API differs
                            protocol = getattr(getattr(config, "miner", object()), "protocol", "http")
                            route = getattr(getattr(config, "miner", object()), "download_route", "/model")
                            if ip and port:
                                url = f"{protocol}://{ip}:{port}{route}"
                            else:
                                logger.warning("Skipping meta without URL or ip:port: %s", download_meta)
                                continue

                        # Output path (reload each attempt so block height is fresh)
                        out_path = Path(config.miner.validator_checkpoint_path) / (
                            f"server_uid{download_meta.get('uid')}_block{subtensor.block}.pt"
                        )

                        logger.info("Checking for new checkpoint from validator %s ...", download_meta.get("uid"))
                        try:
                            download_model(url=url, token=getattr(config.miner, "token", ""), out=out_path)
                            # If download_model doesn't raise, consider it a success
                            download_success = True
                            current_model_version = download_meta["model_version"]
                            current_model_hash = download_meta["model_hash"]
                            logger.info("✅ Downloaded checkpoint to %s (version=%s, hash=%s)",
                                        out_path, current_model_version, current_model_hash)
                            break
                        except Exception as e:
                            logger.warning("Download failed from %s (uid=%s): %s", url, download_meta.get("uid"), e)

                    if not download_success:
                        retries += 1
                        if retries < max_retries:
                            delay = base_delay_s * (2 ** (retries - 1))
                            logger.info("Retrying in %ss (attempt %d/%d)...", delay, retries + 1, max_retries)
                            time.sleep(delay)

                if not download_success:
                    logger.error("❌ All download attempts failed after %d retries.", retries)


            # # --------- SUBMISSION PHASE ---------
            # # Your code: resume, start_step, latest_checkpoint_path = get_resume_info(rank=0, config=config)
            # resume, start_step, latest_checkpoint_path = get_resume_info(rank=0, config=config)
            # latest_checkpoint_path = Path(latest_checkpoint_path)
            # model_file = latest_checkpoint_path / "model.pt"

            # if not model_file.is_file():
            #     log.debug("No model found at %s; skipping submit", model_file)
            
            # else:
            #     if _should_submit():
            #         log.info("Submitting checkpoint: step=%s path=%s", start_step, model_file)
            #         submit_model(
            #             url=SUBMIT_URL,
            #             token=SUBMIT_TOKEN,
            #             step = 0,
            #             uid = config.chain.uid,
            #             hotkey = config.chain.hotkey,
            #             model_path=str(model_file),
            #         )
            #         backoff = 1.0
            #     else:
            #         log.debug("Not time to submit yet (step=%s, last_step=%s)")

        except (Timeout, ReqConnectionError) as e:
            log.warning("Network issue: %s", e)
        except RequestException as e:
            log.warning("HTTP error: %s", e)
        except Exception as e:
            log.exception("Unexpected error in loop: %s", e)

        time.sleep(60 * 5)



if __name__ == "__main__":
    args = parse_args()

    if args.path:
        config = MinerConfig.from_path(args.path)
    else:
        config = MinerConfig()

    main(config)
