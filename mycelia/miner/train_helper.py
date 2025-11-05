import os
import gc
import time
import logging
import json
import fsspec
from functools import partial
import copy
from typing import Tuple, Union, Optional, Dict, Any
from collections.abc import Iterable, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_
import torch.distributed.rpc as rpc

from transformers import (
    get_cosine_schedule_with_warmup,
)

from mycelia.shared.config import MinerConfig
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.model import get_base_model, partial_moe
from mycelia.shared.modeling.modeling_mycelia import get_base_tokenizer
from mycelia.shared.datasets import get_dataloader
from mycelia.shared.checkpoint import (
    get_resume_info,
    save_checkpoint,
    load_checkpoint,
    delete_old_checkpoints,
)
from mycelia.shared.expert_manager import (
    ExpertManager,
    create_expert_groups,
    sync_expert_weights,
    sync_weights,
    get_weight_sum,
    broadcast_weights,
)
from mycelia.shared.evaluate import (
    evaluate_model,
)

# Configure the basic logging setup
logger = logging.getLogger("diloco.train")


def free_cuda_models(
    models: Union[torch.nn.Module, Iterable[torch.nn.Module]],
    optimizers: Optional[Iterable[torch.optim.Optimizer]] = None,
    devices: Optional[Sequence[Union[int, str, torch.device]]] = None,
    move_to_cpu_first: bool = True,
    clear_ipc: bool = True,
    sync_before: bool = True,
):
    """
    Release GPU memory held by models/optimizers.

    What it does:
      1) (optional) synchronize CUDA to let queued work finish
      2) move models to CPU (optional but recommended)
      3) drop grads and optimizer state
      4) delete references and run GC
      5) empty CUDA caching allocator (and IPC cache if requested)

    Args:
      models: A model or iterable of models to free.
      optimizers: Any optimizers associated with those models. If provided,
                  their per-param state will be freed.
      devices: CUDA devices to flush caches for (defaults to all visible or the
               current device if none is set).
      move_to_cpu_first: Move models to CPU before deletion (avoids device
                         context surprises and async frees).
      clear_ipc: Also clear CUDA IPC allocations (relevant with multi-process).
      sync_before: Call torch.cuda.synchronize() before freeing (safer when you
                   just launched kernels that touch these tensors).

    Notes:
      - torch.cuda.empty_cache() only returns *unused cached blocks* to the driver.
        Any still-referenced tensor keeps its memory. That’s why we delete refs first.
      - If you used torch.compile, you may also wish to reset compile caches:
            import torch._dynamo as dynamo; dynamo.reset()
        (This is optional and uses internal APIs; only do it if you know you need it.)
    """

    logger.info("Free cuda models start")
    # Normalize inputs
    if isinstance(models, torch.nn.Module):
        models = [models]
    models = list(models)
    optimizers = list(optimizers) if optimizers is not None else []

    # 1) Sync so we don't free buffers that are still in use by queued kernels
    if sync_before and torch.cuda.is_available():
        try:
            if devices is None:
                torch.cuda.synchronize()  # current device (or all streams on it)
            else:
                for d in devices:
                    torch.cuda.synchronize(d)
        except Exception:
            pass

    # 2) Move models to CPU (helps ensure param storages detach from CUDA)
    if move_to_cpu_first:
        for m in models:
            try:
                m.cpu()
            except Exception:
                pass

    # 3) Clear grads and optimizer state
    for m in models:
        try:
            for p in m.parameters():
                p.grad = None
        except Exception:
            pass

    for opt in optimizers:
        try:
            # Clear per-param state (exp_avg, exp_avg_sq, momentum buffers, etc.)
            opt.state.clear()
        except Exception:
            pass

    # 4) Drop strong references and collect
    #    (Caller is expected to drop their own references too!)
    del models
    del optimizers
    gc.collect()

    # 5) Return unused cached blocks to the driver (and clear IPC if requested)
    if torch.cuda.is_available():
        try:
            if devices is None:
                torch.cuda.empty_cache()
                if clear_ipc:
                    torch.cuda.ipc_collect()
            else:
                for d in devices:
                    with torch.cuda.device(d):
                        torch.cuda.empty_cache()
                        if clear_ipc:
                            torch.cuda.ipc_collect()
        except Exception:
            pass

    logger.info("Free cuda models complete")


def get_status(
    config: MinerConfig,
    model: torch.nn.Module,
    step: int,
    training_time: float,
    total_training_time: float,
    inner_optimizer: torch.optim.Optimizer | None = None,
    inner_opt_step: int | None = None,
    global_opt_step: int | None = None,
    loss_batch: torch.Tensor | None | Any = None,
    aux_loss_batch: torch.Tensor | None | Any = None,
) -> Dict[str, Any]:
    """
    Build a dictionary of training metrics for monitoring and logging.
    Times are reported in hours, throughput in tokens/second.
    """

    if inner_opt_step is None and global_opt_step is None: 
        raise ValueError

    if inner_opt_step is not None: 
        total_batch_size = config.data.batch_size * config.local_par.world_size
        total_samples = inner_opt_step * total_batch_size
        total_tokens = total_samples * config.data.sequence_length

    _, expert_sum = get_weight_sum(model, shared=False)

    # Extract current learning rate (assume one param group or take first)
    
    metrics: Dict[str, Any] = {
        "step": step,
        "inner_opt_step": inner_opt_step,
        "global_opt_step": global_opt_step,
        "lr": next(iter(group["lr"] for group in inner_optimizer.param_groups)) if inner_optimizer is not None else None,
        "param_sum": expert_sum.detach().cpu(),
    }

    if inner_opt_step is not None:
        metrics = metrics | {
            "total_samples": total_samples,
            "total_tokens": total_tokens,
        }


    if training_time > 0 and total_training_time > 0:
        metrics = metrics | {
            "inner_step_time_hours": training_time / 3600,
            "total_training_time_hours": total_training_time / 3600,
        }
        if inner_opt_step is not None:
            metrics = metrics | {
                "tokens_per_second": (config.data.sequence_length * total_batch_size) / training_time,
            }

    if loss_batch is not None and loss_batch != 0:
        if aux_loss_batch is not None and aux_loss_batch != 0:
            metrics = metrics | {
                "loss": float(loss_batch.item() - aux_loss_batch.item()),
                "perplexity": float(torch.exp(loss_batch - aux_loss_batch).item()),
                "aux_loss": float(aux_loss_batch.item()),
            }
        else:
            metrics = metrics | {
                "loss": float(loss_batch.item()),
                "perplexity": float(torch.exp(loss_batch).item()),
            }

    return metrics
