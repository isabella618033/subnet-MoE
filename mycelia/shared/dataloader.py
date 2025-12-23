from __future__ import annotations

from collections.abc import Callable
from functools import partial

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset as TorchIterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import DataCollator, DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from mycelia.shared.app_logging import structlog
from mycelia.shared.helper import h256_int, import_from_string

logger = structlog.get_logger(__name__)


def _fractional_index_filter(_example, idx: int, seed: str, threshold: int) -> bool:
    """Deterministically decide whether to keep a sample based on its streaming index."""
    score = h256_int("dataset_selection", str(idx), seed)
    return score <= threshold


# -----------------------------
# Dataset
# -----------------------------
class DefaultStreamingTorchDataset(TorchIterableDataset):
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
        format_example = partial(self.tokenize_and_format, tokenizer=self.tokenizer, sequence_length=self.seq_length)

        # Explicit per-example iteration avoids surprises with HF's streaming `map` api (which
        # can leave original string columns attached when `column_names` is missing), ensuring
        # we only yield the tokenized dict expected by the collator.
        for example in self.hf_iterable:
            yield format_example(example)

    @staticmethod
    def tokenize_and_format(
        example: dict[str, str], tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> dict[str, list]:
        text = example.get("text", "")
        return tokenizer(text, truncation=True, max_length=sequence_length, padding="max_length")  # type: ignore

    @classmethod
    def get_tokenised_dataset(
        cls,
        config,
        tokenizer: PreTrainedTokenizerBase,
        rank: int | None = None,
        world_size: int | None = None,
        train: bool = True,
        seed: str | None = None,
        fraction: float | None = None,
    ):
        # Load streaming dataset. `disable_tqdm=True` silences progress bars.
        ds = load_dataset(
            config.task.data.dataset_name,
            data_dir=config.task.data.data_dir,
            streaming=True,
        )

        # Select split
        split_name = "train" if train else "validation"
        if split_name not in ds:
            raise ValueError(
                f"Dataset split '{split_name}' not found for "
                f"{config.task.data.dataset_name}:{config.task.data.data_dir}"
            )

        split = ds[split_name]

        # Optional deterministic subsampling based on (seed, fraction)
        # Applied *before* sharding on the streaming iterable.
        if seed is not None and fraction is not None and fraction < 1.0:
            if not (0.0 < fraction <= 1.0):
                raise ValueError("fraction must be in (0.0, 1.0].")

            max_int = 2**256 - 1
            threshold = int(max_int * fraction)

            logger.debug("Applying fractional subsampling", seed=seed, fraction=fraction, threshold=threshold)

            # `with_indices=True` gives us a stable index per element in the stream.
            # Wrap with partial instead of relying on fn_kwargs to keep worker execution simple.
            filter_fn = partial(_fractional_index_filter, seed=seed, threshold=threshold)
            split = split.filter(filter_fn, with_indices=True)

        # Shard across processes if rank/world_size are provided.
        # split_dataset_by_node works with streaming datasets and avoids overlapping samples.
        if world_size is not None and rank is not None:
            try:
                split = split_dataset_by_node(split, world_size=world_size, rank=rank)
            except Exception as e:
                logger.warning(f"Falling back to unsharded split due to split_dataset_by_node error: {e}")

        # Tokenize on-the-fly via adapter (safer for streaming than heavy .map chains).
        tokenized_stream = cls(
            hf_iterable=split,
            tokenizer=tokenizer,
            seq_length=config.task.data.sequence_length,
        )

        return tokenized_stream


# -----------------------------
# Dataloader
# -----------------------------
def get_dataloader(
    config,
    tokenizer: PreTrainedTokenizerBase,
    seed: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    train: bool = True,
    format_fn: Callable | None = None,
    data_collator: DataCollator | None = None,
) -> StatefulDataLoader:
    """
    Build a `StatefulDataLoader` over a streaming HF dataset, tokenized on the fly.

    Parameters
    ----------
    config :
        An object with fields:
          - dataset_name (str), data_dir (str), sequence_length (int),
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
    logger.info("get_dataloader", train=train, seed=seed)
    # Prefer provided rank/world_size, else fall back to config (if present), else no sharding.
    world_size = world_size if world_size is not None else config.task.data.world_size
    rank = rank if rank is not None else config.task.data.rank

    dataset_class_path = getattr(config.task.data, "dataset_class", None)
    if dataset_class_path is None:
        DatasetCls = DefaultStreamingTorchDataset
    else:
        DatasetCls = import_from_string(dataset_class_path)

    tokenised_dataset = DatasetCls.get_tokenised_dataset(
        config=config,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        train=train,
        seed=seed,  # e.g. combined validator seed
        fraction=config.task.data.vali_fraction,  # use ~20% of the dataset
    )

    # Collator for causal LM (no MLM)
    if data_collator is None:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Build loader
    loader = StatefulDataLoader(
        tokenised_dataset,  # split
        collate_fn=data_collator,
        batch_size=config.task.data.per_device_train_batch_size,
        num_workers=1,  # tune based on CPU/disk throughput
    )
    return loader
