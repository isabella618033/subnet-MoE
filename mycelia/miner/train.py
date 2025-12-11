import copy
import datetime
import gc
import os
import time

import bittensor
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

from mycelia.miner.train_helper import free_cuda_models, get_status
from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import setup_chain_worker
from mycelia.shared.checkpoint import (
    ModelMeta,
    delete_old_checkpoints,
    get_resume_info,
    load_checkpoint,
    save_checkpoint,
    start_model_from,
)
from mycelia.shared.config import MinerConfig, parse_args
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.evaluate import evaluate_model
from mycelia.shared.expert_manager import ExpertManager
from mycelia.shared.helper import get_nested_attr
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.model import load_model
from mycelia.shared.modeling.mycelia import get_base_tokenizer

configure_logging()
logger = structlog.get_logger(__name__)


# this is for local DP only
def init_process(local_rank: int, config: MinerConfig, world_size: int, fn: callable, backend: str = "nccl") -> None:
    """
    Initializes the process for distributed training.

    Args:
        rank (int): The rank of the process.
        world_size (int): The total number of processes.
        fn (callable): The function to run for the process.
        backend (str): The backend to use for distributed training.

    Returns:
        None
    """
    os.environ["MASTER_ADDR"] = config.local_par.ip_address
    os.environ["MASTER_PORT"] = str(config.local_par.port)

    if local_rank == 0:
        print(config)  # pretty JSON

    # torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend,
        rank=local_rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=3600),
        device_id=(
            torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            if local_rank < world_size
            else None
        ),
    )

    fn(local_rank, world_size, config)


def cleanup() -> None:
    """
    Cleans up the distributed training environment.

    Returns:
        None
    """
    dist.destroy_process_group()
    torch.cuda.synchronize()


def setup_training(
    config,
    rank: int,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    current_model_meta: ModelMeta,
) -> tuple[
    torch.nn.Module,  # model
    torch.nn.Module,  # global_model
    torch.optim.Optimizer,  # inner_optimizer
    torch.optim.Optimizer,  # outer_optimizer
    torch.amp.GradScaler,  # inner_scaler
    torch.amp.GradScaler,  # outer_scaler
    torch.optim.lr_scheduler.LRScheduler,  # scheduler
    # int,  # start_step
    "ExpertManager",  # em
    StatefulDataLoader,
    dict,  # current model version
]:
    """
    Build model(s), experts layout, optimizers, scheduler, scaler, and optionally resume from a checkpoint.

    Args:
        config: Training/config object with attributes used here (e.g., lr, outer_lr, warmup_steps, etc.).
        rank (int): Process rank.
        device (str | torch.device): Device for the local model (e.g., "cuda:0").

    Returns:
        model (nn.Module): Local (possibly partial) MoE model placed on `device`.
        global_model (nn.Module): Deep-copied global model on CPU, kept in sync with `model`.
        inner_optimizer (Optimizer): Optimizer for `model`.
        outer_optimizer (Optimizer): Optimizer for `global_model`.
        scaler (torch.cuda.amp.GradScaler): GradScaler (enabled iff `config.model.precision == "fp16-mixed"`).
        scheduler (LRScheduler): LR scheduler attached to `inner_optimizer`.
        start_step (int): Step to resume from (0 if starting fresh).
        expert_groups (Sequence[Sequence[int]]): Grouping returned by `create_expert_groups`; typically a list
            (or other sequence) of groups where each group lists the ranks/experts belonging to it.
        group_ids (int): This rank’s group id from `create_expert_groups`.
        expert_manager (ExpertManager): The instantiated ExpertManager for this model/rank.

    Notes:
        - Param group layouts are taken from the *target* optimizers created here.
        - If `resume_from_ckpt` is set and a checkpoint is found, model/opt/scheduler/scaler states are restored
          before syncing `global_model` from `model`.
    """
    # === checkpoint info ===
    resume = False
    latest_checkpoint_path = None
    if get_nested_attr(config, "ckpt.resume_from_ckpt", False):
        resume, start_step, latest_checkpoint_path = get_resume_info(rank, config)

    # === model & Experts manager ===
    expert_manager = ExpertManager(config)
    model, model_meta = load_model(rank, config, expert_manager, subtensor, wallet)
    model = model.to(device)
    global_model = copy.deepcopy(model).cpu()

    # === optimizers ===
    logger.info(f" rank {rank} optimizer")
    inner_optimizer = torch.optim.AdamW(model.named_parameters(), lr=config.opt.lr, weight_decay=0.1, betas=(0.9, 0.95))
    outer_optimizer = torch.optim.SGD(
        global_model.named_parameters(),
        lr=config.opt.outer_lr,
        momentum=config.opt.outer_momentum,
        nesterov=True,
    )

    # === scheduler === (for inner optimizer)
    logger.info(f" rank {rank} scheduler")
    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=config.sched.warmup_steps,
        num_training_steps=config.sched.total_steps,
    )

    # === scaler ===
    inner_scaler = torch.amp.GradScaler(
        "cuda", enabled=(get_nested_attr(config, "model.precision", "") == "fp16-mixed")
    )
    outer_scaler = torch.amp.GradScaler(
        "cuda", enabled=(get_nested_attr(config, "model.precision", "") == "fp16-mixed")
    )

    # === dataloader ===
    train_dataloader = get_dataloader(config, rank=rank, world_size=config.task.data.world_size, tokenizer=tokenizer)

    # === load checkpoint (if any) ===
    if get_nested_attr(config, "resume_from_ckpt", False) and resume and latest_checkpoint_path:
        _ = load_checkpoint(
            config=config,
            checkpoint_path=latest_checkpoint_path,
            inner_optimizer=inner_optimizer,
            outer_optimizer=outer_optimizer,
            scheduler=scheduler,
            inner_scaler=inner_scaler,
            outer_scaler=outer_scaler,
            rank=rank,
            device=device,
            data_loader=train_dataloader,
        )

    logger.info(f"rank {rank} setup_training: success!")
    return (
        model,
        global_model,
        inner_optimizer,
        outer_optimizer,
        inner_scaler,
        outer_scaler,
        scheduler,
        expert_manager,
        train_dataloader,
        model_meta,
    )


def train_worker(rank: int, world_size: int, config: MinerConfig) -> None:
    """
    The worker function for training in a distributed setting.

    Args:
        rank (int): The rank of the process.
        world_size (int): The total number of processes.
        config (Config): The configuration object for the training.

    Returns:
        None
    """
    eval_rref = None
    if rank == 0:
        config.write()

    # === set logging ===
    metric_logger = MetricLogger(config, rank)

    # === set up chain worker ===
    wallet, subtensor = setup_chain_worker(config)

    # === mis ===
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    tokenizer = get_base_tokenizer(config)

    # TODO: need to split the dataset per miner first, and then split more
    eval_dataloader = get_dataloader(
        config,
        rank=config.local_par.world_size,
        world_size=config.local_par.world_size + 1,
        tokenizer=tokenizer,
    )

    # === set up training ===
    (
        model,
        global_model,
        inner_optimizer,
        outer_optimizer,
        inner_scaler,
        outer_scaler,
        scheduler,
        # start_step,
        expert_manager,
        train_dataloader,
        current_model_meta,
    ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta=None)

    # === training ===
    loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    training_time = 0
    total_training_time = 0
    training_start_time = None

    inner_optimizer.zero_grad()
    outer_optimizer.zero_grad()
    try:
        for step, batch in enumerate(
            iterable=train_dataloader, start=current_model_meta.inner_opt * config.local_par.gradient_accumulation_steps
        ):
            # for each step, we run 1 backward
            # for each inner_opt_step, we run local optimization; gradient_accumulation_steps = 1 real step
            # for each global_opt_interval number of inner_opt_step, we synchronise weight from different ddp worker, and then run global optimization

            inner_opt_step = step // config.local_par.gradient_accumulation_steps
            is_inner_optimizer_step = step % config.local_par.gradient_accumulation_steps == 0
            is_start_step = step == current_model_meta.inner_opt * config.local_par.gradient_accumulation_steps

            # === Training and inner optimization ===
            if (
                not is_start_step
            ):  # skip training when it is the start step, so that we can benchamrk the original model first
                model.train()
                global_model.train()
                if training_start_time is None:
                    training_start_time = time.time()

                batch_device = {}
                for key in batch.keys():
                    batch_device[key] = batch[key].to(device)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(**batch_device)

                    loss = outputs.loss / config.local_par.gradient_accumulation_steps
                    # aux_loss = outputs.aux_loss / config.local_par.gradient_accumulation_steps if outputs.aux_loss is not None else torch.tensor(0)
                    aux_loss = torch.tensor(0)

                loss_batch += loss.detach()
                aux_loss_batch += aux_loss.detach()

                inner_scaler.scale(loss).backward()

                del loss, batch_device, outputs

                # === inner optimizer ===
                if is_inner_optimizer_step:
                    logger.info("inner opt step", loss_batch, aux_loss_batch)

                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(world_size)

                    inner_scaler.unscale_(optimizer=inner_optimizer)

                    clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping

                    inner_scaler.step(inner_optimizer)

                    inner_scaler.update()

                    scheduler.step()

                    inner_optimizer.zero_grad()

                    training_time = time.time() - training_start_time
                    total_training_time += training_time
                    training_start_time = None

            # === Log metric ===
            if (
                is_inner_optimizer_step
                and inner_opt_step % max(round(config.local_par.global_opt_interval * 0.02), 1) == 0
            ):
                logger.info("optimizer step", loss_batch, aux_loss_batch)
                metrics = get_status(
                    config=config,
                    model=model,
                    step=step,
                    inner_opt_step=inner_opt_step,
                    training_time=training_time,
                    total_training_time=total_training_time,
                    inner_optimizer=inner_optimizer,
                    loss_batch=loss_batch,
                    aux_loss_batch=aux_loss_batch,
                )
                metric_logger.log(metrics, print_log=False)

            # === local validation and log metric ===
            if is_inner_optimizer_step and inner_opt_step % config.log.metric_interval == 0:
                logger.info("reached barrier, waiting for partial evaluation")
                dist.barrier(device_ids=[rank])

                logger.info("start partial evaluation")

                val_metric = evaluate_model(
                    rank=rank, step=inner_opt_step, model=model, eval_dataloader=eval_dataloader, device=device
                )

                logger.info(f"evaluation before log {val_metric}")
                metrics = (
                    get_status(
                        config=config,
                        model=model,
                        step=step,
                        inner_opt_step=inner_opt_step,
                        training_time=training_time,
                        total_training_time=total_training_time,
                        inner_optimizer=inner_optimizer,
                        loss_batch=loss_batch,
                        aux_loss_batch=aux_loss_batch,
                    )
                    | val_metric
                )

                metric_logger.log(metrics)

                logger.info("reached barrier, waiting for partial validation and metric logging to complete")
                dist.barrier(device_ids=[rank])

            # === save checkpoint ===
            if (
                is_inner_optimizer_step
                and config.ckpt.checkpoint_interval is not None
                and inner_opt_step % config.ckpt.checkpoint_interval == 0
            ):
                logger.info("saving checkpoint")
                ckpt_path = os.path.join(
                    config.ckpt.checkpoint_path, f"globalver_{current_model_meta.global_ver}_inneropt_{inner_opt_step}"
                )

                save_checkpoint(
                    checkpoint_path=ckpt_path,
                    model=model,
                    inner_optimizer=inner_optimizer,
                    outer_optimizer=outer_optimizer,
                    scheduler=scheduler,
                    loss=loss_batch.item(),
                    inner_scaler=inner_scaler,
                    outer_scaler=outer_scaler,
                    data_loader=train_dataloader,
                    save_global_state=rank == 0,
                    rank=rank,
                )

                if rank == 0:
                    if config.ckpt.checkpoint_topk is not None:
                        ckpt_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, config.ckpt.checkpoint_topk)
                        if ckpt_deleted:
                            logger.info(f"Deleted old checkpoints: {ckpt_deleted}")

                logger.info("reached barrier, waiting for complete checkpoint saving")
                dist.barrier(device_ids=[rank])

            # === reload model ===
            if (
                is_inner_optimizer_step
                and start_model_from(
                    rank,
                    config,
                    primary_ckpt_path=config.ckpt.validator_checkpoint_path,
                    secondary_ckpt_path=config.ckpt.checkpoint_path,
                )[1]
                == current_model_meta
            ):
                dist.barrier(device_ids=[rank])  # make sure everything is saved and everyone is ready to load
                logger.info("freeing cuda memory")
                free_cuda_models(models=[model, global_model], optimizers=[inner_optimizer], devices=[device])
                logger.info("restarting model")
                (
                    model,
                    global_model,
                    inner_optimizer,
                    outer_optimizer,
                    inner_scaler,
                    outer_scaler,
                    scheduler,
                    # start_step,
                    expert_manager,
                    train_dataloader,
                    current_model_version,
                ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta)

            # === Clean up ===
            if is_inner_optimizer_step:
                loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
                aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
                gc.collect()
                torch.cuda.empty_cache()

    except Exception:
        logger.error("Quit training", exc_info=True)
        cleanup()
        metric_logger.close()

        if rank == 0:
            torch.save(global_model.state_dict(), "mycelia_final.pt")


def run_distributed_training() -> None:
    """
    Runs the distributed training process.

    Returns:
        None
    """
    args = parse_args()

    if args.path:
        config = MinerConfig.from_path(args.path)
    else:
        config = MinerConfig()

    mp.spawn(
        init_process,
        args=(config, config.local_par.world_size, train_worker),
        nprocs=config.local_par.world_size,
    )


if __name__ == "__main__":
    run_distributed_training()
