import os
import torch
import fsspec
from copy import deepcopy
from mycelia.config import Config
from fsspec.generic import GenericFileSystem
from torchdata.stateful_dataloader import StatefulDataLoader
from mycelia.shared.app_logging import structlog

logger = structlog.getLogger(__name__)

def get_resume_info(rank: int, config: Config) -> tuple[bool, int, str | None]:
    """
    Retrieves the resume information for a given rank and checkpoint configuration.

    Args:
        rank (int): The rank of the process.
        ckpt_config (Config): The configuration object for the checkpoint.

    Returns:
        tuple[bool, int, str | None]: A tuple containing a boolean indicating success,
        the checkpoint step, and an optional string message.
    """
    """
    Check if we should resume from a checkpoint, if yes return the path to the checkpoint, otherwise return None
    """
    if config.ckpt.resume_from_ckpt is None:
        return False, None

    elif isinstance(config.ckpt.resume_from_ckpt, bool):
        # Using fsspec to list directory contents
        fs = GenericFileSystem()
        try:
            fs, root = fsspec.core.url_to_fs(config.ckpt.checkpoint_path)  # root has no protocol
            ckpt_files = fs.ls(root, detail=False)
            ckpt_files = [f for f in ckpt_files if filter_ckpt_files(f)]
        except FileNotFoundError:
            logger.info(f"rank {rank}: Checkpoint path {config.ckpt.checkpoint_path} not found, starting from scratch")
            return False, 0, None

        if len(ckpt_files) == 0:
            logger.info(f"rank {rank}: No checkpoints found in {config.ckpt.checkpoint_path}, starting from scratch")
            return False, 0, None

        latest_ckpt = max(ckpt_files, key=lambda f: int(f.split("_")[-1]))
        logger.info(f"rank {rank}: Latest checkpoint found in {latest_ckpt}")
        return True, int(latest_ckpt.split("_")[-1]), latest_ckpt

def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    inner_optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    rank: int,
    outer_optimizer: torch.optim.Optimizer | None = None,
    inner_scaler: torch.amp.GradScaler | None = None,
    outer_scaler: torch.amp.GradScaler | None = None,
    loss: float | None = None,
    data_loader: StatefulDataLoader | None = None,
    save_global_state: bool = True,
) -> None:
    """
    Saves the current model checkpoint.

    Returns:
        None
    """
    # === save model, optimizer ===
    checkpoint = {
        "model_state_dict": {k: v.detach().to("cpu", non_blocking=True) for k, v in model.state_dict().items()},
        "loss": loss,
    }

    target = os.path.join(checkpoint_path, f"model.pt")
    if not os.path.exists(target):
        with fsspec.open(target, "wb") as f:
            torch.save(checkpoint, f)

    # === save optimizer ===
    opt_checkpoint = {
        "optimizer_state_dict": inner_optimizer.state_dict(),
    }

    target = os.path.join(checkpoint_path, f"inner_optimizer.pt")
    if not os.path.exists(target):
        with fsspec.open(target, "wb") as f:
            torch.save(opt_checkpoint, f)

    opt_checkpoint = {
        "optimizer_state_dict": outer_optimizer.state_dict(),
    }
    target = os.path.join(checkpoint_path, f"outer_optimizer.pt")
    if not os.path.exists(target):
        with fsspec.open(target, "wb") as f:
            torch.save(opt_checkpoint, f)

    # === save dataloader ===
    if data_loader is not None:
        rank_state_dict = {}
        rank_state_dict["data_loader"] = data_loader.state_dict()

        target = os.path.join(checkpoint_path, f"dataloader_rank{rank}.pt")
        if not os.path.exists(target):
            with fsspec.open(target, "wb") as f:
                torch.save(rank_state_dict, f)

        del rank_state_dict

    if not save_global_state:
        return

    # === save global state ===
    global_state_dict = {
        "scheduler": scheduler.state_dict(),
        "loss": loss if loss is not None else 0,
    }

    if inner_scaler is not None:
        global_state_dict["inner_scaler_state_dict"] = inner_scaler.state_dict()

    if outer_scaler is not None:
        global_state_dict["outer_scaler_state_dict"] = outer_scaler.state_dict()

    target = os.path.join(checkpoint_path, "global_state.pt")
    if not os.path.exists(target):
        with fsspec.open(target, "wb") as f:
            torch.save(global_state_dict, f)

    logger.info(f"Checkpoint saved to {checkpoint_path}")

    del checkpoint
    del opt_checkpoint
    del global_state_dict

def load_optimizer(checkpoint_path, optimizer):
    def _get_name_to_id(optimizer_state_dict):
        param_name = [pid for g in optimizer_state_dict["param_groups"] for pid in g["param_names"]]
        param_id = [pid for g in optimizer_state_dict["param_groups"] for pid in g["params"]]
        param_name_to_id = {name: pid for name, pid in zip(param_name, param_id)}
        return param_name_to_id

    def _get_name_to_param(optimizer_state_dict):
        """
        Build a mapping: parameter_name -> optimizer_state (for that param).
        Works regardless of the optimizer’s internal param IDs.
        """
        state = optimizer_state_dict["state"]
        name_to_id = _get_name_to_id(optimizer_state_dict)
        name_to_state = {param_name: state[pid] for param_name, pid in name_to_id.items()}
        return name_to_state

    def _update_state_dict(optimizer_state_dict, name_to_param):
        """
        Build a *loadable* optimizer.state_dict() for `target_optimizer` such that:
        - Params that belong to the requested experts get their merged state.
        - All other params are left without state (the optimizer will re-init them).
        """

        optimizer_state_dict["state"] = {}

        target_name_to_id = _get_name_to_id(optimizer_state_dict)

        for pid, name in target_name_to_id.items():
            st = name_to_param.get(name, None)
            if st is not None:
                optimizer_state_dict["state"][pid] = deepcopy(st)  # avoid aliasing

        return optimizer_state_dict

    full_name_to_param = {}
    model_files = fsspec.open_files(checkpoint_path, mode="rb")

    if len(model_files) == 0:
        return

    for f in model_files:
        with f as fh:
            state_dict = torch.load(fh, map_location=torch.device("cpu"))
            full_name_to_param = full_name_to_param | _get_name_to_param(state_dict["optimizer_state_dict"])

    optimizer.load_state_dict(_update_state_dict(optimizer.state_dict(), full_name_to_param))


def load_checkpoint(
    checkpoint_path: str,
    config: Config,
    rank: int | None,
    device: torch.device,
    model: torch.nn.Module | None = None,
    inner_optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    outer_optimizer: torch.optim.Optimizer | None = None,
    inner_scaler: torch.amp.GradScaler | None = None,
    outer_scaler: torch.amp.GradScaler | None = None,
    data_loader: StatefulDataLoader | None = None,
) -> float:
    """Load the model and optimizer state from a checkpoint folder

    Args:
        checkpoint_path: the path to the checkpoint folder
        model: the model to load
        optimizer: the optimizer to load
        scheduler: the scheduler to load
        outer_optimizer: the outer optimizer to load
        data_loader: the data loader to load

    Returns:
        loss: the loss from the checkpoint
    """

    if model is not None:
        full_state_dict = {}
        model_files = fsspec.open_files(os.path.join(checkpoint_path, "model*.pt"), mode="rb")
        model_files += fsspec.open_files(os.path.join(checkpoint_path, "model.pt"), mode="rb")
        for f in model_files:
            with f as fh:
                state_dict = torch.load(fh, map_location=torch.device("cpu"))
                full_state_dict = full_state_dict | state_dict["model_state_dict"]
                logger.info(f"rank {rank}: loaded checkpoint {f} loss {state_dict['loss']:.5f}")

        model.load_state_dict(full_state_dict, strict=True)
        model.to(device)

    if inner_optimizer is not None:
        load_optimizer(os.path.join(checkpoint_path, "inner_optimizer*.pt"), inner_optimizer)

    if outer_optimizer is not None:
        load_optimizer(os.path.join(checkpoint_path, "outer_optimizer*.pt"), outer_optimizer)

    if data_loader is not None:
        with fsspec.open(os.path.join(checkpoint_path, f"dataloader_rank{rank}.pt"), "rb") as f:
            rank_state_dict = torch.load(f, map_location=torch.device("cpu"))
        data_loader.load_state_dict(rank_state_dict["data_loader"])

    with fsspec.open(os.path.join(checkpoint_path, f"global_state.pt"), "rb") as f:
        global_state_dict = torch.load(f, map_location=torch.device("cpu"))

    if scheduler is not None:
        scheduler.load_state_dict(global_state_dict["scheduler"])
        inner_optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]

        # Push optimizer tensors to the same device as the model
        for st in outer_optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)

    if inner_scaler is not None:
        inner_scaler.load_state_dict(global_state_dict["inner_scaler_state_dict"])

    if outer_scaler is not None:
        outer_scaler.load_state_dict(global_state_dict["outer_scaler_state_dict"])

    return global_state_dict["loss"]


def filter_ckpt_files(f: str) -> bool:
    """
    Filters checkpoint files based on specific criteria.

    Args:
        f (str): The filename to check.

    Returns:
        bool: True if the file meets the criteria, False otherwise.
    """
    try:
        int(f.split("_")[-1])
        return True
    except ValueError:
        return False


def delete_old_checkpoints(checkpoint_path: str, topk: int) -> list[str]:
    """
    Deletes old checkpoints, keeping only the top 'k' most recent ones.

    Args:
        checkpoint_path (str): The path to the checkpoint directory.
        topk (int): The number of recent checkpoints to keep.

    Returns:
        list[str]: A list of deleted checkpoint filenames.
    """
    fs = GenericFileSystem()
    ckpt_files = [f for f in fs.ls(checkpoint_path, detail=False) if filter_ckpt_files(f)]
    ckpt_files.sort(key=lambda x: int(x.split("_")[-1]))

    ckpt_deleted = []
    for ckpt_file in ckpt_files[:-topk]:
        fs.rm(ckpt_file, recursive=True)
        ckpt_deleted.append(ckpt_file)
    return ckpt_deleted
