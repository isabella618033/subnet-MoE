#!/usr/bin/env python3
"""
Async validator pipeline:
1) gather miner info (mocked here)
2) download models from miners (uses provided download(url, token, out, ...))
3) push model jobs into an asyncio.Queue
4) evaluator workers pop jobs, load models, call evaluate_model(...)
5) aggregate scores with MinerScoreAggregator (resets on hotkey change)

Swap out the MOCK sections with your real logic.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mycelia.miner.client import download_model
from mycelia.shared.evaluate import evaluate_model
from mycelia.shared.app_logging import structlog, configure_logging
from mycelia.shared.datasets import get_dataloader, HFStreamingTorchDataset

logger = structlog.get_logger(__name__)

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MinerEvalJob:
    uid: str
    hotkey: str
    model_path: str
    step: int


# -------------------------- Pipeline Config -----------------------------------
MAX_CONCURRENT_DOWNLOADS = 4
EVAL_WORKERS = 2
DOWNLOAD_TIMEOUT_SEC = 60
EVAL_MAX_BATCHES = 50
# ------------------------------------------------------------------------------

def load_model_from_path(path: str, base_model) -> nn.Module:
    sd = torch.load(path, map_location=torch.device("cpu"))['model_state_dict']
    base_model.load_state_dict(sd, strict = False)
    return base_model 

async def evaluator_worker(
    name: str,
    config, 
    jobs_q: "asyncio.Queue[MinerEvalJob]",
    aggregator: MinerScoreAggregator,
    device: "torch.device",
    # eval_dataloader,
    base_model: nn.Module,
    tokenizer, 
    max_eval_batches: int = EVAL_MAX_BATCHES,
    rank: Optional[int] = None,
):
    while True:
        job = await jobs_q.get()
        if job is None:  # type: ignore
            jobs_q.task_done()
            logger.debug(f"{name}: shutdown signal received.")
            break

        try:
            # Load model (potentially blocking) in a thread
            model = await asyncio.to_thread(load_model_from_path, job.model_path, base_model)
            eval_dataloader = await asyncio.to_thread(get_dataloader, config, tokenizer, 0, 10)
            metrics = await asyncio.to_thread(
                evaluate_model, job.step, model, eval_dataloader, device, max_eval_batches, rank
            )
            # choose a primary score (here 'accuracy'); adjust if your evaluate_model returns other keys
            score = float(metrics.get("val_loss", 100))
            aggregator.add_score(job.uid, job.hotkey, score)
            logger.info(f"{name}: uid={job.uid} hotkey={job.hotkey} score={score:.4f} path={job.model_path}")
            
            del eval_dataloader
        
        except Exception as e:
            logger.exception(f"{name}: Evaluation failed for uid={job.uid}: {e}")
        finally:
            jobs_q.task_done()


async def run_evaluation(config, step, device, miners, score_aggregator, base_model: nn.Module, tokenizer):
    # Device & dataloader (MOCK). Replace eval_dataloader with a real one.
    logger.info(f"Discovered {len(miners)} miners.")
    miners_q: asyncio.Queue[MinerEvalJob] = asyncio.Queue()

    # Enqueue miners
    for m in miners:
        await miners_q.put(m)

    # Spin up evaluator workers
    eval_workers = [
        asyncio.create_task(evaluator_worker(f"evaluator-{i+1}", config, miners_q, score_aggregator, device, base_model, tokenizer))
        for i in range(EVAL_WORKERS)
    ]

    # Wait for all miners to be processed
    await miners_q.join()

    # Signal evaluator workers to stop
    for _ in eval_workers:
        await jobs_q.put(None)  # type: ignore
    await asyncio.gather(*eval_workers)
