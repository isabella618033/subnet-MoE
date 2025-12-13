from __future__ import annotations

import asyncio
from dataclasses import dataclass

import torch
import torch.nn as nn

from mycelia.shared.app_logging import structlog
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.evaluate import evaluate_model

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MinerEvalJob:
    uid: str
    hotkey: str
    model_path: str
    step: int


# -------------------------- Pipeline Config -----------------------------------
MAX_CONCURRENT_DOWNLOADS = 4
EVAL_WORKERS = 1
DOWNLOAD_TIMEOUT_SEC = 60
EVAL_MAX_BATCHES = 50
# ------------------------------------------------------------------------------


def load_model_from_path(path: str, base_model, device: torch.device) -> nn.Module:
    sd = torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
    base_model.load_state_dict(sd, strict=False)
    return base_model.to(device)


async def evaluator_worker(
    name: str,
    config,
    jobs_q: asyncio.Queue[MinerEvalJob],
    aggregator: MinerScoreAggregator,
    device: torch.device,
    base_model: nn.Module,
    tokenizer,
    combinded_seed: str,
    max_eval_batches: int = EVAL_MAX_BATCHES,
    rank: int | None = None,
):
    import gc

    while True:
        job = await jobs_q.get()
        if job is None:  # type: ignore
            jobs_q.task_done()
            logger.debug(f"{name}: shutdown signal received.")
            break

        try:
            # Clear memory before loading
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"{name}: Evaluating uid={job.uid}")

            # Load model (potentially blocking) in a thread
            model = await asyncio.to_thread(load_model_from_path, job.model_path, base_model, device)
            eval_dataloader = await asyncio.to_thread(get_dataloader, config, tokenizer, combinded_seed, 0, 10)

            with torch.inference_mode():
                metrics = await asyncio.to_thread(
                    evaluate_model, job.step, model, eval_dataloader, device, max_eval_batches, rank
                )

            # choose a primary score (here 'accuracy'); adjust if your evaluate_model returns other keys
            score = float(metrics.get("val_loss", 100))
            aggregator.add_score(job.uid, job.hotkey, score)
            logger.info(f"{name}: uid={job.uid} score={score:.4f}")

            # Explicit cleanup
            del eval_dataloader, model
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"{name}: OOM for uid={job.uid}")
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.exception(f"{name}: Evaluation failed for uid={job.uid}: {e}")
        finally:
            jobs_q.task_done()


async def run_evaluation(
    config, step, device, miners, score_aggregator, base_model: nn.Module, tokenizer, combinded_seed
):
    # Device & dataloader (MOCK). Replace eval_dataloader with a real one.
    miners_q: asyncio.Queue[MinerEvalJob] = asyncio.Queue()

    # Enqueue miners
    for m in miners:
        await miners_q.put(m)

    # Spin up evaluator workers
    eval_workers = [
        asyncio.create_task(
            evaluator_worker(
                f"evaluator-{i+1}", config, miners_q, score_aggregator, device, base_model, tokenizer, combinded_seed
            )
        )
        for i in range(EVAL_WORKERS)
    ]

    # Wait for all miners to be processed
    await miners_q.join()

    # Signal evaluator workers to stop
    for _ in eval_workers:
        await miners_q.put(None)

    await asyncio.gather(*eval_workers)
