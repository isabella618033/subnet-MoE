"""
Distributed evaluation utilities for sharded model checkpoints (RPC-based).

This module provides:
  * Lightweight RPC helpers (`send_shard`, `mark_done`) for workers to stream
    shard state_dicts to a dedicated evaluator node.
  * An `Evaluator` actor that collects shards per step, assembles the full model,
    runs evaluation, and logs metrics.
  * Helpers to assemble the model and to run the evaluation loop.

Assumptions
-----------
* Workers send CPU tensors (cheaper to serialize) as shard state_dicts.
* There exists a project-level logger named `logger`.
* Project utilities provide:
    - get_base_model(config)       -> nn.Module
    - get_base_tokenizer(config)   -> HF tokenizer
    - get_dataloader(config, ...)  -> eval dataloader
    - MetricLogger(config).log(dict)
"""

from __future__ import annotations

import gc
import threading
import time
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm

import torch
from torch import nn
import logging

from torch.distributed import rpc
from torch.futures import Future
from mycelia.shared.app_logging import structlog
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.model import get_base_model
from mycelia.shared.modeling.modeling_mycelia import get_base_tokenizer
from mycelia.shared.datasets import get_dataloader
from mycelia.shared.expert_manager import ExpertManager

logger = structlog.getLogger(__name__)

tqdm(disable=True, total=0)

@torch.no_grad
def evaluate_model(
    step: int,
    model: nn.Module,
    eval_dataloader,
    device: torch.device,
    max_eval_batches: Optional[int] = 50,
    rank: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run a lightweight eval pass and return scalar metrics.

    Parameters
    ----------
    step : int
        Training step for logging context.
    model : nn.Module
        Fully-assembled model placed on the correct device.
    eval_dataloader :
        Iterable of evaluation batches (dicts of Tensors).
    device : torch.device
        Device to run evaluation on.
    max_eval_batches : Optional[int]
        Optional cap on the number of batches to evaluate.

    Returns
    -------
    Dict[str, float]
        e.g., {"val_loss": 2.345}
    """
    model.eval()
    loss_sum: float = 0.0
    aux_loss_sum: float = 0.0
    with torch.inference_mode():
        for batch_step, batch in enumerate(iterable=eval_dataloader):
            device_batch = {}
            for key in batch.keys():
                device_batch[key] = batch[key].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**device_batch)

            loss_sum += float(outputs.loss.detach().item())
            aux_loss_sum += float(outputs.aux_loss.detach().item()) if outputs.aux_loss is not None else 0

            del outputs, device_batch

            if max_eval_batches is not None and batch_step >= max_eval_batches:
                break

    return {"val_loss": (loss_sum - aux_loss_sum) / batch_step, "val_aux_loss": aux_loss_sum / batch_step}