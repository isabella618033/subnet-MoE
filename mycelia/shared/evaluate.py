from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from tqdm import tqdm

from mycelia.shared.app_logging import structlog

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
    logger.info("evaluate model", step=step)
    model.to(device)
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
            aux_loss_sum += (
                float(outputs.aux_loss.detach().item())
                if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None
                else 0
            )

            del outputs, device_batch

            if max_eval_batches is not None and batch_step >= max_eval_batches:
                break

    return {"val_loss": (loss_sum - aux_loss_sum) / batch_step, "val_aux_loss": aux_loss_sum / batch_step}
