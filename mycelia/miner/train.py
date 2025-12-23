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
from mycelia.shared.helper import get_model_hash, get_nested_attr
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.model import freeze_parameters, load_model
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
    torch.optim.Optimizer,  # inner_optimizer
    torch.amp.GradScaler,  # inner_scaler
    torch.optim.lr_scheduler.LRScheduler,  # scheduler
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
    logger.info("(0) Setup training")

    # === model & Experts manager ===
    logger.info(f"init - model and expert manager")
    expert_manager = ExpertManager(config)
    model, model_meta = load_model(rank, config, expert_manager, subtensor, wallet)
    model = model.to(device)
    model = freeze_parameters(model=model, expert_manager=expert_manager, expert_group_id=config.task.expert_group_id)

    # === optimizers ===
    logger.info(f"init - optimizer")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    inner_optimizer = torch.optim.AdamW(trainable_params, lr=config.opt.lr, weight_decay=0.1, betas=(0.9, 0.95))

    # === scheduler === (for inner optimizer)
    logger.info(f"init - scheduler")
    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=config.sched.warmup_steps,
        num_training_steps=config.sched.total_steps,
    )

    # === scaler ===
    logger.info(f"init - inner scaler")
    inner_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,  # enabled=(get_nested_attr(config, "model.precision", "") == "fp16-mixed")
    )

    # === dataloader ===
    logger.info(f"init - train dataloader")
    train_dataloader = get_dataloader(config, rank=rank, world_size=config.task.data.world_size, tokenizer=tokenizer)

    # === load checkpoint (if any) ===
    logger.info(f"init - load checkpoint")
    resume = False
    latest_checkpoint_path = None
    if get_nested_attr(config, "ckpt.resume_from_ckpt", False):
        resume, start_step, latest_checkpoint_path = get_resume_info(rank, config)

    if get_nested_attr(config, "resume_from_ckpt", False) and resume and latest_checkpoint_path:
        _ = load_checkpoint(
            config=config,
            checkpoint_path=latest_checkpoint_path,
            inner_optimizer=inner_optimizer,
            scheduler=scheduler,
            inner_scaler=inner_scaler,
            rank=rank,
            device=device,
            data_loader=train_dataloader,
        )

    logger.info(f"setup_training: success!")
    return (
        model,
        inner_optimizer,
        inner_scaler,
        scheduler,
        expert_manager,
        train_dataloader,
        model_meta,
    )


def sum_model_gradients(model):
    """
    Returns the sum of absolute gradients of all model parameters.
    Assumes backward() has already been called.
    """
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total += param.grad.abs().sum().item()
        return total


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

    eval_dataloader = get_dataloader(
        config,
        rank=config.local_par.world_size,
        world_size=config.local_par.world_size + 1,
        tokenizer=tokenizer,
    )

    # === set up training ===
    (
        model,
        inner_optimizer,
        inner_scaler,
        scheduler,
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
    try:
        for step, batch in enumerate(
            iterable=train_dataloader,
            start=max(0, current_model_meta.inner_opt) * config.local_par.gradient_accumulation_steps,
        ):
            # for each step, we run 1 backward
            # for each inner_opt_step, we run local optimization; gradient_accumulation_steps = 1 real step
            # for each global_opt_interval number of inner_opt_step, we synchronise weight from different ddp worker, and then run global optimization

            inner_opt_step = step // config.local_par.gradient_accumulation_steps
            is_inner_optimizer_step = step % config.local_par.gradient_accumulation_steps == 0
            is_start_step = step == current_model_meta.inner_opt * config.local_par.gradient_accumulation_steps
            current_model_meta.inner_opt = inner_opt_step

            # === Training and inner optimization ===
            if is_inner_optimizer_step:
                logger.info(
                    "(1) Start epoch training",
                    step=step,
                    inner_opt_step=inner_opt_step,
                    is_inner_optimizer_step=is_inner_optimizer_step,
                    gradient_accumulation_steps=config.local_par.gradient_accumulation_steps,
                    current_model_meta=current_model_meta,
                )
            if (
                not is_start_step
            ):  # skip training when it is the start step, so that we can benchamrk the original model first
                model.train()
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

                loss_batch += loss.item()
                aux_loss_batch += aux_loss.item()
                logger.info("training", loss=loss, grad_sum=sum_model_gradients(model))

                inner_scaler.scale(loss).backward()

                # === Aggressively free intermediate tensors ===
                del loss, aux_loss, batch_device, outputs
                gc.collect()

            # === inner optimizer ===
            if not is_start_step and is_inner_optimizer_step:
                old_model_hash = get_model_hash(model.state_dict())

                for n, p in model.named_parameters():
                    if p.grad is None or torch.isnan(p.grad.sum()):
                        continue
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world_size)

                inner_scaler.unscale_(optimizer=inner_optimizer)

                grad_norm = clip_grad_norm_(
                    [p for p in model.parameters() if p.grad is not None and not torch.isnan(p.grad.sum())], 1.0
                )  # gradient clipping # <- turned grad to nan

                scale_before = inner_scaler.get_scale() if inner_scaler.is_enabled() else None
                step_result = inner_scaler.step(inner_optimizer)
                step_skipped = inner_scaler.is_enabled() and step_result is None

                if step_skipped:
                    logger.warning(
                        "GradScaler skipped optimizer step due to inf/NaN gradients",
                        grad_norm=float(grad_norm),
                        scale_before=scale_before,
                    )

                else:
                    logger.info(
                        "Optimizer step applied",
                        grad_norm=float(grad_norm),
                        scale_before=scale_before,
                    )

                inner_scaler.update()
                if inner_scaler.is_enabled():
                    logger.info(
                        "Scaler updated",
                        scale_after=inner_scaler.get_scale(),
                    )

                scheduler.step()

                inner_optimizer.zero_grad()

                training_time = time.time() - training_start_time
                total_training_time += training_time
                training_start_time = None

                # === Clear memory after optimizer step ===
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Memory cleared after optimizer step")

                new_model_hash = get_model_hash(model.state_dict())
                logger.info(f"Updated model", old_model_hash=old_model_hash, new_model_hash=new_model_hash)

            # === Log metric ===
            if (
                is_inner_optimizer_step
                and inner_opt_step % max(round(config.local_par.global_opt_interval * 0.02), 1) == 0
            ):
                logger.info("(2) Logging step", loss_batch=loss_batch, aux_loss_batch=aux_loss_batch)
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
                logger.info("(3) Local evaluation")

                val_metric = evaluate_model(
                    rank=rank, step=inner_opt_step, model=model, eval_dataloader=train_dataloader, device=device
                )

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
                # dist.barrier(device_ids=[rank])

            # === save checkpoint ===
            if (
                is_inner_optimizer_step
                and config.ckpt.checkpoint_interval is not None
                and inner_opt_step % config.ckpt.checkpoint_interval == 0
            ):
                logger.info("(4) Saving checkpoint")

                ckpt_path = os.path.join(
                    config.ckpt.checkpoint_path,
                    f"globalver_{current_model_meta.global_ver}_inneropt_{inner_opt_step}",
                )

                save_checkpoint(
                    checkpoint_path=ckpt_path,
                    model=model,
                    inner_optimizer=inner_optimizer,
                    scheduler=scheduler,
                    loss=loss_batch.item(),
                    inner_scaler=inner_scaler,
                    data_loader=train_dataloader,
                    save_global_state=rank == 0,
                    rank=rank,
                    save_model_by_expert_group=True,
                    expert_manager=expert_manager,
                )

                if config.ckpt.checkpoint_topk is not None:
                    ckpt_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, config.ckpt.checkpoint_topk)
                    if ckpt_deleted:
                        logger.info(f"Deleted old checkpoints: {ckpt_deleted}")

                logger.info("reached barrier, waiting for complete checkpoint saving")
                # dist.barrier(device_ids=[rank])

            # === reload model ===
            if is_inner_optimizer_step:
                logger.info("(5) Reload Model")

                newest_checkpoint = start_model_from(
                    rank,
                    config,
                    primary_ckpt_path=config.ckpt.validator_checkpoint_path,
                    secondary_ckpt_path=config.ckpt.checkpoint_path,
                )[1]

                if newest_checkpoint > current_model_meta:
                    logger.info(
                        "Should reload model",
                        newest_checkpoint=newest_checkpoint,
                        current_model_meta=current_model_meta,
                    )
                    # dist.barrier(device_ids=[rank])  # make sure everything is saved and everyone is ready to load
                    logger.info("freeing cuda memory")
                    free_cuda_models(models=[model], optimizers=[inner_optimizer], devices=[device])
                    logger.info(
                        "restarting model",
                        current_model_meta=current_model_meta,
                        largest_avail_model=start_model_from(
                            rank,
                            config,
                            primary_ckpt_path=config.ckpt.validator_checkpoint_path,
                            secondary_ckpt_path=config.ckpt.checkpoint_path,
                        )[1],
                    )
                    (
                        model,
                        inner_optimizer,
                        inner_scaler,
                        scheduler,
                        expert_manager,
                        train_dataloader,
                        current_model_version,
                    ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta)
                else:
                    logger.info(
                        "No need to reload model",
                        newest_checkpoint=newest_checkpoint,
                        current_model_meta=current_model_meta,
                    )

            # === Clean up ===
            if is_inner_optimizer_step:
                logger.info("(6) Clean up")
                loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
                aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Clean up completed")

    except Exception:
        logger.error("Quit training", exc_info=True)
        dist.destroy_process_group()
        torch.cuda.synchronize()
        metric_logger.close()

        if rank == 0:
            torch.save(model.state_dict(), "mycelia_final.pt")


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
