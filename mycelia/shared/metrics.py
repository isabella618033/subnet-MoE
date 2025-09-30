"""
Metric logging utilities.

This module provides a small logger that:
  1) Appends metrics to a local CSV at `config.metric_path`, harmonizing columns over time.
  2) Optionally logs the same metrics to Weights & Biases (W&B) when `config.log_wandb=True`.

Usage
-----
logger = MetricLogger(config)
logger.log({"step": 10, "val_loss": 1.23})
logger.close()
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import torch
import pandas as pd
import torch
import wandb

from dimoe.config import Config

logger = logging.getLogger("diloco.metrics")


class MetricLogger:
    """
    Write metrics locally (CSV) and optionally to Weights & Biases.

    Parameters
    ----------
    config : Config
        Must provide:
          - metric_path: str (CSV file path)
          - log_wandb: bool
          - wandb_project_name: str
          - run_name: str
          - model_path: str
    """

    def __init__(self, config: Config, rank: int = 0, validation: bool = False) -> None:
        self.csv_path: str = config.metric_path
        self.log_wandb: bool = bool(config.log_wandb)
        self.validation = validation

        # Ensure the metrics directory exists.
        metrics_dir = os.path.dirname(self.csv_path) or "."
        os.makedirs(metrics_dir, exist_ok=True)

        run_name = f"[full] {config.run_name}" if validation else f"[partial] rank{rank}-{config.run_name}"
        self.wandb_run: Optional[wandb.sdk.wandb_run.Run] = None

        if config.wandb_resume:
            if validation:
                run_id = config.wandb_full_id
            else:
                run_id = config.wandb_partial_id[rank]
        else:
            run_id = None

        if self.log_wandb:
            try:
                self.wandb_run = wandb.init(
                    entity="isabella_cl-cruciblelabs",
                    project=f"subnet-expert-{config.wandb_project_name}",
                    name=run_name,
                    tags=[config.run_name],
                    id=run_id,
                    resume="allow",
                    config=config.__dict__,
                )
            except Exception as e:
                logger.warning(f"W&B init failed, disabling W&B logging: {e}")
                self.log_wandb = False
                self.wandb_run = None

    # -------- public API --------
    def close(self) -> None:
        """Finish the W&B run (if enabled)."""
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception as e:
                logger.warning(f"W&B finish failed: {e}")
            finally:
                self.wandb_run = None

    # -------- helpers --------
    def _wandb_log(self, metrics: Dict[str, Any]) -> None:
        try:
            self.wandb_run.log(metrics)

            if "val_loss" in metrics:
                self.wandb_run.alert(title=f"Success log: validation {self.validation}", text=f"{metrics}")

        except Exception as e:
            logger.warning(f"W&B log failed (will continue locally): {e}")

    def log(self, metrics: Mapping[str, Any], print_log=True) -> None:
        """
        Log a single metrics dict to CSV (always) and W&B (optional).

        Any torch tensors will be converted to Python scalars (0-D) or lists (N-D).
        Sequences (list/tuple) are stored as the first element if len==1, else as stringified lists.
        """

        flat = self._flatten_metrics(metrics)

        if print_log:
            logger.info("Metric Log: %s", json.dumps(flat, indent=2))

        self._local_log(flat)
        if self.log_wandb and self.wandb_run is not None:
            self._wandb_log(flat)

    def _local_log(self, metrics: dict):
        # Flatten list/tensor values
        flat_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (list, tuple)):
                flat_metrics[k] = v[0] if len(v) == 1 else str(v)
            elif torch.is_tensor(v):
                flat_metrics[k] = v.item() if v.ndim == 0 else v.detach().cpu().tolist()
            else:
                flat_metrics[k] = v

        new_row = pd.DataFrame([flat_metrics])

        if not os.path.exists(self.csv_path):
            # First write: save with header
            new_row.to_csv(self.csv_path, index=False)
        else:
            # Load existing CSV and merge schemas.
            try:
                existing = pd.read_csv(self.csv_path)
            except Exception as e:
                logger.warning(f"Failed to read existing metrics CSV, rewriting header: {e}")
                new_row.to_csv(self.csv_path, index=False)
                self._fsync_if_supported()
                return

            # Add any new columns to existing (as empty) and to new_row (for missing in new).
            for col in new_row.columns:
                if col not in existing.columns:
                    existing[col] = ""  # backfill empty values for prior rows
            for col in existing.columns:
                if col not in new_row.columns:
                    new_row[col] = ""  # ensure consistent column order

            # Reorder new_row to match existing column order and append.
            new_row = new_row[existing.columns]
            updated = pd.concat([existing, new_row], ignore_index=True)
            updated.to_csv(self.csv_path, index=False)

        self._fsync_if_supported()

    @staticmethod
    def _flatten_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Convert metrics dict into CSV/W&B-safe values.

        Rules
        -----
        * torch.Tensor:
            - 0-D -> .item()
            - N-D -> .detach().cpu().tolist()
        * list/tuple:
            - len == 1 -> first element
            - else -> stringified list (to keep CSV rectangular)
        * everything else -> as-is
        """
        flat: Dict[str, Any] = {}
        for k, v in metrics.items():
            if torch.is_tensor(v):
                flat[k] = v.item() if v.ndim == 0 else v.detach().cpu().tolist()
            elif isinstance(v, (list, tuple)):
                flat[k] = v[0] if len(v) == 1 else str(v)
            else:
                flat[k] = v
        return flat

    @staticmethod
    def _fsync_if_supported() -> None:
        """
        Force a disk flush if the OS provides `os.sync` (Linux).
        Helpful for real-time monitoring tailing the CSV.
        """
        if hasattr(os, "sync"):
            try:
                os.sync()
            except Exception:
                # Best-effort; ignore if the platform denies it.
                pass
