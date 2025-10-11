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

from mycelia.shared.checkpoint import get_resume_info        
from mycelia.config import MinerConfig, parse_args           
from mycelia.shared.client import submit_model, download_model           

# -------------- Logging --------------
LOG_LEVEL = os.getenv("CKPT_MGR_LOG", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("checkpoint_manager")

# -------------- Defaults / Env --------------
SUBMIT_URL = os.getenv("SUBMIT_URL", "http://localhost:8000/submit-checkpoint")
SUBMIT_TOKEN = os.getenv("SUBMIT_TOKEN", "")
DOWNLOAD_URL = os.getenv("DOWNLOAD_URL", "http://localhost:8000/get-checkpoint")
DOWNLOAD_TOKEN = os.getenv("DOWNLOAD_TOKEN", "")

# poll cadence (seconds)
POLL_SECS = int(os.getenv("CKPT_MGR_POLL_SECS", "15"))
# do not resubmit unless global step advanced by this many steps
SUBMIT_MIN_STEP_DELTA = int(os.getenv("CKPT_MGR_SUBMIT_MIN_STEP_DELTA", "1"))
# minimum time between submissions (seconds)
SUBMIT_MIN_INTERVAL = int(os.getenv("CKPT_MGR_SUBMIT_MIN_INTERVAL", "60"))
# minimum time between downloads (seconds)
DOWNLOAD_MIN_INTERVAL = int(os.getenv("CKPT_MGR_DOWNLOAD_MIN_INTERVAL", "60"))

# where to persist last-known metadata across restarts
STATE_PATH = Path(os.getenv("CKPT_MGR_STATE_PATH", "./.ckpt_manager_state.json")).resolve()


def _should_submit() -> bool:
    return False


def _should_download() -> bool:
    return False

_stop = False
def _handle_sig(signum, frame):
    global _stop
    _stop = True

signal.signal(signal.SIGINT, _handle_sig)
signal.signal(signal.SIGTERM, _handle_sig)


def main(config):
    # state across runs
    log.info("Starting checkpoint manager | submit every ≥%ss & step+%s | download every ≥%ss",
             SUBMIT_MIN_INTERVAL, SUBMIT_MIN_STEP_DELTA, DOWNLOAD_MIN_INTERVAL)

    # backoff on errors to avoid thrashing
    backoff = 1.0

    while not _stop:
        tick_start = time.time()
        try:
            # --------- SUBMISSION PHASE ---------
            # Your code: resume, start_step, latest_checkpoint_path = get_resume_info(rank=0, config=config)
            resume, start_step, latest_checkpoint_path = get_resume_info(rank=0, config=config)
            latest_checkpoint_path = Path(latest_checkpoint_path)
            model_file = latest_checkpoint_path / "model.pt"

            if not model_file.is_file():
                log.debug("No model found at %s; skipping submit", model_file)
            else:
                if _should_submit():
                    log.info("Submitting checkpoint: step=%s path=%s", start_step, model_file)
                    submit_model(
                        url=SUBMIT_URL,
                        token=SUBMIT_TOKEN,
                        step = 0,
                        uid = config.chain.uid,
                        hotkey = config.chain.hotkey,
                        model_path=str(model_file),
                    )
                    backoff = 1.0
                else:
                    log.debug("Not time to submit yet (step=%s, last_step=%s)")

            # --------- DOWNLOAD PHASE ---------
            if _should_download():
                # Reload miner config before deciding output path (your snippet does this each time)
                out_path = (Path(config.miner.validator_checkpoint_path) / f"uid{3}_{'hk3'}.pt")

                log.info("Checking for new checkpoint from validator...")
                raw, step_hint = download_model(url = DOWNLOAD_URL, token = DOWNLOAD_TOKEN, out = out_path)

        except (Timeout, ReqConnectionError) as e:
            log.warning("Network issue: %s", e)
            time.sleep(min(60.0, backoff))
            backoff = min(60.0, backoff * 1.8)
        except RequestException as e:
            log.warning("HTTP error: %s", e)
            time.sleep(min(60.0, backoff))
            backoff = min(60.0, backoff * 1.8)
        except Exception as e:
            log.exception("Unexpected error in loop: %s", e)
            time.sleep(min(60.0, backoff))
            backoff = min(60.0, backoff * 1.8)

        # sleep until next poll, unless stopping
        if _stop:
            break
        elapsed = time.time() - tick_start
        sleep_s = max(0.0, POLL_SECS - elapsed)
        time.sleep(sleep_s)

    log.info("Checkpoint manager stopped.")


if __name__ == "__main__":
    args = parse_args()

    if args.path:
        config = MinerConfig.from_json(args.path)
    else:
        config = MinerConfig()

    main(config)
