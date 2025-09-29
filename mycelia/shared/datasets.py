# llm_weightnet/shared/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from ..settings import Settings
from mycelia.shared.logging import structlog

log = structlog.get_logger(__name__)

try:
    from datasets import load_dataset  # optional, but very handy
except Exception:  # pragma: no cover
    load_dataset = None


@dataclass
class DatasetBundle:
    """
    A thin wrapper that returns train/eval/test splits in a consistent shape.
    You can return HF datasets, PyTorch dataloaders, etc.
    """
    name: str
    train: Any
    eval: Any
    test: Any


def _load_hf_dataset(dataset_id: str, subset: Optional[str]) -> DatasetBundle:
    if load_dataset is None:
        raise RuntimeError("`datasets` not installed. Add it to dependencies or use local loader.")
    ds = load_dataset(dataset_id, subset) if subset else load_dataset(dataset_id)
    # Expect standard split names; adapt if different
    train = ds.get("train") or ds[list(ds.keys())[0]]
    eval_ = ds.get("validation") or ds.get("eval") or None
    test = ds.get("test") or None
    log.info("datasets.hf_loaded", dataset_id=dataset_id, subset=subset, splits=list(ds.keys()))
    return DatasetBundle(name=f"{dataset_id}{('/'+subset) if subset else ''}", train=train, eval=eval_, test=test)


def _load_local_folder(path: Path) -> DatasetBundle:
    """
    Minimal example for local CSV/JSONL; replace with your format.
    """
    import pandas as pd
    files = list(path.glob("*.csv")) + list(path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No CSV/JSONL files in {path}")
    # Simple: treat first file as train
    train_path = files[0]
    df = pd.read_csv(train_path) if train_path.suffix == ".csv" else pd.read_json(train_path, lines=True)
    return DatasetBundle(name=str(path), train=df, eval=None, test=None)


def load(dataset_id: str, subset: Optional[str] = None, settings: Optional[Settings] = None) -> DatasetBundle:
    """
    Smart loader that understands:
      - HF hub dataset id (e.g., 'imdb', 'cnn_dailymail')
      - Local folder path (e.g., './data/myset')
    """
    cfg = settings or Settings()
    # Heuristic: local folder if it exists
    p = Path(dataset_id)
    if p.exists() and p.is_dir():
        return _load_local_folder(p)

    # Else: try HF datasets
    return _load_hf_dataset(dataset_id, subset)
