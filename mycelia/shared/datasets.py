from __future__ import annotations
import logging
from functools import partial
from typing import Dict, Iterator, Optional

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset as TorchIterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

logger = logging.getLogger("diloco.data")


def tokenize_function(
    example: Dict[str, str], tokenizer: PreTrainedTokenizerBase, sequence_length: int
) -> Dict[str, list]:
    """
    Tokenize a single text example for causal LM.

    Parameters
    ----------
    example : Dict[str, str]
        A dataset row containing a "text" field.
    tokenizer : PreTrainedTokenizerBase
        HF tokenizer (must be configured with pad_token if padding is used).
    sequence_length : int
        Max sequence length to truncate/pad to.

    Returns
    -------
    Dict[str, list]
        Tokenized outputs compatible with DataCollatorForLanguageModeling
        (e.g., "input_ids", "attention_mask").
    """
    text = example.get("text", "")
    return tokenizer(
        text,
        truncation=True,
        max_length=sequence_length,
        padding="max_length",
    )


class HFStreamingTorchDataset(TorchIterableDataset):
    """
    Thin adapter to wrap a Hugging Face streaming (Iterable) dataset so it yields
    tokenized dicts ready for a collator.

    This is useful when you want to keep the tokenization logic explicit and
    avoid relying on `IterableDataset.map(...)` behaviors.
    """

    def __init__(self, hf_iterable, tokenizer: PreTrainedTokenizerBase, seq_length: int):
        """
        Parameters
        ----------
        hf_iterable :
            A split of an HF streaming dataset, e.g. ds["train"] with streaming=True.
        tokenizer : PreTrainedTokenizerBase
            HF tokenizer to use for tokenization.
        seq_length : int
            Max sequence length for truncation/padding.
        """
        self.hf_iterable = hf_iterable
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __iter__(self):
        return (tokenize_function(example, self.tokenizer, self.seq_length) for example in self.hf_iterable)


def get_dataloader(
    config,
    tokenizer: PreTrainedTokenizerBase,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    train: bool = True,
) -> StatefulDataLoader:
    """
    Build a `StatefulDataLoader` over a streaming HF dataset, tokenized on the fly.

    Parameters
    ----------
    config :
        An object with fields:
          - dataset_name (str), data_dir (str), sequence_length (int),
            per_device_train_batch_size (int)
        and optionally:
          - eval_world_size / world_size used by your launcher (provided here for clarity)
    tokenizer : PreTrainedTokenizerBase
        HF tokenizer used for tokenization and by the collator.
    rank : Optional[int]
        Zero-based index of the current process in the node/world. Used for sharding.
    world_size : Optional[int]
        Total number of processes. Used for sharding.
    train : bool
        If True, returns a loader over the training split; else returns a loader for validation
        (or None if the dataset has no validation split).

    Returns
    -------
    Optional[StatefulDataLoader]
        A stateful dataloader for the requested split, or None if the eval split is missing.
    """
    # Prefer provided rank/world_size, else fall back to config (if present), else no sharding.
    world_size = world_size if world_size is not None else config.data.world_size
    rank = rank if rank is not None else config.data.rank

    # Load streaming dataset. `disable_tqdm=True` silences progress bars.
    ds = load_dataset(
        config.data.dataset_name,
        data_dir=config.data.data_dir,
        streaming=True,
    )

    # Select split
    split_name = "train" if train else "validation"
    if split_name not in ds:
        raise ValueError(f"Dataset split '{split_name}' not found for {config.data.dataset_name}:{config.data.data_dir}")

    split = ds[split_name]

    # Shard across processes if rank/world_size are provided.
    # split_dataset_by_node works with streaming datasets and avoids overlapping samples.
    if world_size is not None and rank is not None:
        try:
            split = split_dataset_by_node(split, world_size=world_size, rank=rank)
        except Exception as e:
            logger.warning(f"Falling back to unsharded split due to split_dataset_by_node error: {e}")

    # Tokenize on-the-fly via adapter (safer for streaming than heavy .map chains).
    tokenized_stream = HFStreamingTorchDataset(
        hf_iterable=split,
        tokenizer=tokenizer,
        seq_length=config.data.sequence_length,
    )

    # Collator for causal LM (no MLM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Build loader
    loader = StatefulDataLoader(
        tokenized_stream,  # split
        collate_fn=data_collator,
        batch_size=config.data.per_device_train_batch_size,
        num_workers=4,  # tune based on CPU/disk throughput
    )
    return loader
