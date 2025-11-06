
import os
import sys
import time
import json
import hashlib
import logging
import signal
from pathlib import Path
from typing import Optional, Tuple, List, Any

from collections import Counter
import bittensor 
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError as ReqConnectionError

import bittensor 
from mycelia.shared.chain import serve_axon, get_status, commit_status, MinerStatus
from mycelia.shared.checkpoint import get_resume_info, delete_old_checkpoints
from mycelia.shared.config import MinerConfig, parse_args           
from mycelia.miner.client import submit_model, download_model           
from mycelia.shared.app_logging import structlog, configure_logging
from mycelia.shared.cycle import setup_chain_worker, scan_for_new_model, should_submit_model, search_model_submission_destination, get_validator_miner_assignment

configure_logging()
logger = structlog.get_logger(__name__)

def main(config):

    config.write()

    wallet, subtensor = setup_chain_worker(config)

    current_model_version = 0
    current_model_hash = 'xxx'
    last_submission_block = 0

    commit_status(config, wallet, subtensor, MinerStatus(
        expert_group = 1,
    ))

    while True:
        try:
            # --------- DOWNLOAD PHASE ---------
            should_download, download_metas = scan_for_new_model(
                current_model_version, current_model_hash, config, subtensor
            )

            logger.info('check for download', should_download = should_download)
            
            if should_download and download_metas:
                download_success = False
                retries = 0
                max_retries = 3
                base_delay_s = 5  # backoff base

                while (not download_success) and (retries < max_retries):
                    for download_meta in download_metas:
                        
                        logger.info('downloading from candidate... ', download_meta)
                        
                        # Resolve URL if not provided; fall back to ip/port + default route
                        url = download_meta.get("url")
                        if not url:
                            ip = download_meta.get("ip")
                            port = download_meta.get("port")
                            # Best-effort defaults; customize if your API differs
                            protocol = getattr(getattr(config, "miner", object()), "protocol", "http")
                            if ip and port:
                                url =f"{protocol}://{ip}:{port}/get-checkpoint"
                            else:
                                logger.warning("Skipping meta without URL or ip:port: %s", download_meta)
                                continue

                        # Output path (reload each attempt so block height is fresh)
                        out_path = Path(config.miner.validator_checkpoint_path) / (
                            f"validator_{download_meta.get('uid')}_version_{download_meta.get('model_version')}block_{subtensor.block}.pt"
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

            delete_old_checkpoints(config.miner.validator_checkpoint_path , config.ckpt.checkpoint_topk)

            # --------- SUBMISSION PHASE ---------
            resume, start_step, latest_checkpoint_path = get_resume_info(rank=0, config=config)

            start_submit = False
            while not start_submit:
                start_submit, block_till = should_submit_model(config, subtensor, last_submission_block) 
                time.sleep(12)

            destination_axon = search_model_submission_destination(
                wallet = wallet,
                config = config,
                subtensor = subtensor
            )
            
            submit_model(
                url=f"http://{destination_axon.ip}:{destination_axon.port}/submit-checkpoint",
                token="",
                step = 0,
                uid = config.chain.uid,
                hotkey = config.chain.hotkey_ss58,
                model_path=f"{latest_checkpoint_path}/model.pt",
            )

            last_submission_block = subtensor.block
        

        except (Timeout, ReqConnectionError) as e:
            logger.warning("Network issue: %s", e)
        except RequestException as e:
            logger.warning("HTTP error: %s", e)
        except Exception as e:
            logger.exception("Unexpected error in loop: %s", e)

        time.sleep(60)



if __name__ == "__main__":
    args = parse_args()

    if args.path:
        config = MinerConfig.from_path(args.path)
    else:
        config = MinerConfig()

    main(config)
