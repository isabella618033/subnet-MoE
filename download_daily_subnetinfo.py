#!/usr/bin/env python3
"""
Download Bittensor subnet info once per day since March 1, 2025.

Requirements:
    pip install bittensor

Usage:
    python download_bittensor_subnetinfo_daily.py
"""

import os
import json
import traceback
from dataclasses import asdict
from datetime import datetime, timedelta, timezone, date

import bittensor as bt


# ---------------- Configuration ---------------- #

# Network name: "finney" is mainnet. Adjust if you need testnet, etc.
BT_NETWORK = "wss://archive.cruciblelabs.com:9944"

# Starting date (UTC) for daily snapshots
START_DATE = date(2025, 3, 1)

# Directory where JSON files will be stored
OUTPUT_DIR = "bittensor_subnetinfo_daily"

# How many blocks apart to sample when estimating average block time
BLOCK_TIME_SAMPLE_WINDOW = 1000

# How close (in seconds) block time must be to our target date for us to stop refining
TIME_TOLERANCE_SEC = 60 * 30  # 30 minutes

# ------------------------------------------------ #


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def seri_subnet_info(subnet_info):
    subnet_info_dict = asdict(subnet_info)
    return {k: v if not isinstance(v, bt.Balance) else v.tao for k, v in subnet_info_dict.items()}

def main():
    ensure_output_dir(OUTPUT_DIR)

    # End date = today (UTC)
    today_utc = datetime.now(timezone.utc).date()

    print(f"Connecting to Bittensor network: {BT_NETWORK}")

    # If you have archive endpoints, you can pass archive_endpoints=[...]
    # so old blocks are available.
    subtensor = bt.Subtensor(
        network=BT_NETWORK,
        retry_forever=True,
        # Example:
        # archive_endpoints=["wss://your-archive-node:9944"],
    )
    start_block = 5060000 
    latest_block = subtensor.block
    for block in range(start_block, latest_block, 7200):
        day = (block - start_block) / 7200
        out_path = os.path.join(OUTPUT_DIR, f"subnetinfo_{block}.json")

        # Skip if already downloaded
        if os.path.exists(out_path):
            print(f"[SKIP] {block} -> {out_path} already exists")
            continue


        try:
            block_ts = subtensor.get_timestamp(block=block)
            print(f"  Approx block: {block} (timestamp: {block_ts.isoformat()})")

            # Fetch subnet info at that block
            subnets_info = subtensor.all_subnets(block=block)

            # Convert dataclasses to plain dicts for JSON
            subnets_serializable = [seri_subnet_info(s) for s in subnets_info]

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(subnets_serializable, f, indent=2, sort_keys=True)

            print(f"  [OK] Saved {len(subnets_serializable)} subnets -> {out_path}")

        except Exception as e:
            # Common issues: no archive node so old blocks aren't available, network hiccups, etc.
            print(f"  [ERROR] Failed for day {day} block {block}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
