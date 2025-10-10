import os
import gc
import asyncio
import time
import logging
import json
import fsspec
import copy
import datetime
from functools import partial
import copy
from typing import Tuple, Union, Optional, Dict, Any, List
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_
import torch.distributed.rpc as rpc
from torchdata.stateful_dataloader import StatefulDataLoader

from transformers import get_cosine_schedule_with_warmup, PreTrainedTokenizerBase

from mycelia.config import MinerConfig, ValidatorConfig, parse_args
from mycelia.shared.app_logging import structlog, configure_logging
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.model import load_base_model 
from mycelia.shared.modeling.modeling_mycelia import get_base_tokenizer, partial_moe
from mycelia.shared.datasets import get_dataloader, HFStreamingTorchDataset
from mycelia.miner.train_helper import free_cuda_models, get_status
from mycelia.shared.evaluate import evaluate_model
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
    populate_global_grads_from_local,
)
from mycelia.shared.helper import *
from mycelia.validator.aggregator import MinerScoreAggregator
from mycelia.validator.evaluator import run_evaluation, gather_miner_info, MinerInfo

configure_logging()
logger = structlog.get_logger(__name__)


def init_process(rank: int, config: MinerConfig, world_size: int, fn: callable, backend: str = "nccl") -> None:
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
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(config.port)

    if rank == 0:
        print(config)  # pretty JSON

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=3600),
        device_id=torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu") if rank < world_size else None,
    )

    if config.eval_world_size >= 1:
        opts = rpc.TensorPipeRpcBackendOptions(
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            num_worker_threads=16,  # tweak as needed
        )

        rpc.init_rpc(
            name=("evaluator" if rank == config.world_size + config.eval_world_size - 1 else f"worker{rank}"),
            rank=rank,
            world_size=config.world_size + config.eval_world_size,
            rpc_backend_options=opts,
        )

    if config.eval_world_size >= 1 and rank == config.world_size + config.eval_world_size - 1:
        logger.info(f"adding evaluation {config.eval_world_size}")
        # on evaluator rank only
        _init_evaluator_rref(config, rank, num_groups=config.num_worker_groups)

        # Evaluator rank
        EVAL_INSTANCE = Evaluator(config, rank, num_groups=config.num_worker_groups)
        # Block here until RPC shutdown (training workers will shut us down)
        rpc.shutdown()
        # Make sure background thread stops
        EVAL_INSTANCE.stop()


def cleanup() -> None:
    """
    Cleans up the distributed training environment.

    Returns:
        None
    """
    dist.destroy_process_group()
    torch.cuda.synchronize()


def setup_training(
    config, rank: int, device: torch.device, tokenizer: PreTrainedTokenizerBase
) -> Tuple[
    torch.nn.Module,  # model
    torch.nn.Module,  # global_model
    torch.optim.Optimizer,  # outer_optimizer
    torch.amp.GradScaler,  # outer_scaler
    int,  # start_step
    "ExpertManager",  # em
    StatefulDataLoader,
]:
    """
    Build model(s), experts layout, optimizers, scheduler, scaler, and optionally resume from a checkpoint.
    """
    # === checkpoint info ===
    resume = False
    start_step = 0
    latest_checkpoint_path = None
    if get_nested_attr(config,"ckpt.resume_from_ckpt", False):
        resume, start_step, latest_checkpoint_path = get_resume_info(rank, config)

    # === model & Experts manager ===
    model, em = load_base_model(rank, config)
    model = model.to(device)
    global_model = copy.deepcopy(model).cpu()

    # === optimizers ===
    logger.info(f" rank {rank} optimizer")
    outer_optimizer = torch.optim.SGD(
        global_model.named_parameters(),
        lr=config.opt.outer_lr,
        momentum=config.opt.outer_momentum,
        nesterov=True,
    )

    # === scheduler === (for inner optimizer)
    logger.info(f" rank {rank} scheduler")

    # === scaler ===
    outer_scaler = torch.amp.GradScaler("cuda", enabled=(get_nested_attr(config, "model.precision", "") == "fp16-mixed"))

    # === dataloader ===
    train_dataloader = get_dataloader(config, rank=rank, world_size=config.data.world_size, tokenizer=tokenizer)

    # === load checkpoint (if any) ===
    if get_nested_attr(config,"resume_from_ckpt", False) and resume and latest_checkpoint_path:
        # logger.info(
        #     "rank %s setup training: resuming from %s (start_step=%s)", rank, latest_checkpoint_path, start_step
        # )
        _ = load_checkpoint(
            config=config,
            checkpoint_path=latest_checkpoint_path,
            outer_optimizer=outer_optimizer,
            outer_scaler=outer_scaler,
            rank=rank,
            device=device,
            data_loader=train_dataloader,
        )

    logger.info(f"rank {rank} setup_training: success!")
    return (
        model,
        global_model,
        outer_optimizer,
        outer_scaler,
        start_step,
        em,
        train_dataloader,
    )

# TODO 
# def renew
# dataset, validator connection, miner connection

def run_global_optimization(
        model: nn.Module,
        global_model: nn.Module,
        device: torch.device,
        rank: int,
        logp: callable,
        outer_optimizer: torch.optim.Optimizer,
        miners: List[MinerInfo],
):
    
    # --- sync + outer step ---
    # keep global model on device for syncing/stepping, then move back to CPU
    global_model.to(device)

    old_shared_name, old_shared_sum = get_weight_sum(model, shared=True)
    old_expert_name, old_expert_sum = get_weight_sum(model, shared=False)

    # dist.barrier(device_ids=[rank])
    logp("start syncing shared weights")

    populate_global_grads_from_every_local(config, global_model, miners)
    # TODO: sync grad across validators

    outer_optimizer.step()
    outer_optimizer.zero_grad()

    # copy updated global weights back into the worker model safely
    with torch.no_grad():
        model.load_state_dict(global_model.state_dict(), strict=True)

    new_shared_name, new_shared_sum = get_weight_sum(model, shared=True)
    new_expert_name, new_expert_sum = get_weight_sum(model, shared=False)

    logp(
        f"outer optimizer step (shared) {old_shared_name}, {old_shared_sum:.8f} "
        f"-> {new_shared_name}, {new_shared_sum:.8f}"
    )
    # dist.barrier(device_ids=[rank])
    logp(
        f"outer optimizer step (expert) {old_expert_name}, {old_expert_sum:.8f} "
        f"-> {new_expert_name}, {new_expert_sum:.8f}"
    )

    global_model.to("cpu")
    del old_shared_name, old_shared_sum
    del old_expert_name, old_expert_sum
    del new_shared_name, new_shared_sum
    del new_expert_name, new_expert_sum
    gc.collect()
    torch.cuda.empty_cache()


def populate_global_grads_from_every_local(config, global_model, miners):
    for miner in miners: 
        miner_model = copy.deepcopy(global_model) 
        path =config.vali.miner_submission_path / f"{miner.uid}_{miner.hotkey}.pt"
        sd = torch.load(path, map_location=torch.device("cpu"))['model_state_dict']
        miner_model.load_state_dict(sd, strict = False)
        populate_global_grads_from_local(global_model, miner_model, weight = 1 / len(miners))

def run(rank: int, world_size: int, config: MinerConfig) -> None:
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

    # === create checkpoint directory ===
    os.makedirs(config.ckpt.base_checkpoint_path, exist_ok=True)
    os.makedirs(config.ckpt.checkpoint_path, exist_ok=True)
    os.makedirs(config.log.base_metric_path, exist_ok=True)
    os.makedirs(config.vali.miner_submission_path, exist_ok=True)

    # === set logging ===
    logger.info(config)
    metric_logger = MetricLogger(config, rank)

    # === mis ===
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    tokenizer = get_base_tokenizer(config)

    # TODO: need to split the dataset per miner first, and then split more
    eval_dataloader = get_dataloader(
        config,
        rank=0,
        world_size=10,
        tokenizer=tokenizer,
    )

    # === set up training ===
    (
        model,
        global_model,
        outer_optimizer,
        outer_scaler,
        start_step,
        em,
        train_dataloader,
    ) = setup_training(config, rank, device, tokenizer)

    # === ===
    aggregator = MinerScoreAggregator()

    # === training ===
    loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    training_time = 0
    total_training_time = 0
    training_start_time = None

    outer_optimizer.zero_grad()
    
    global_opt_step = start_step
    try:
        while True: 
            # for each step, we run 1 backward
            # for each inner_opt_step, we run local optimization; gradient_accumulation_steps = 1 real step
            # for each global_opt_interval number of inner_opt_step, we synchronise weight from different ddp worker, and then run global optimization

            def logp(
                msg: str, loss_batch: torch.Tensor | None = None, aux_loss_batch: torch.Tensor | None = None
            ) -> None:
                if aux_loss_batch is not None:
                    msg = f"aux loss_batch {aux_loss_batch:.4f} | {msg}"

                if loss_batch is not None:
                    if aux_loss_batch is not None:
                        msg = f"loss_batch {loss_batch-aux_loss_batch:.4f} | {msg}"
                    else:
                        msg = f"loss_batch {loss_batch:.4f} | {msg}"

                logger.info(f"rank {rank} | gloabl_opt {global_opt_step} | {msg}")

            # === Get miner ===
            miners = gather_miner_info()

            # === Download miner model and evaluate the miners ===
            asyncio.run(run_evaluation(
                config = config,
                step = global_opt_step,
                device = model.device,
                miners = miners,
                aggregator = aggregator,
                base_model = model,
                tokenizer = tokenizer
            ))

            logp(msg = "eval result" + str(aggregator.uid_score_pairs()))

            # === global optimizer ===
            logp(f"reached global opt barrier", loss_batch, aux_loss_batch)
            run_global_optimization(
                model = model,
                global_model = global_model,
                device = device,
                rank = rank,
                logp = logp,
                outer_optimizer = outer_optimizer,
                miners = miners,
            )

            # === validation and log metric ===
            logp(f"reached barrier, waiting for partial evaluation")
            # dist.barrier(device_ids=[rank])

            logp(f"start partial evaluation")

            val_metric = evaluate_model(
                rank=rank, step=global_opt_step, model=model, eval_dataloader=eval_dataloader, device=device
            )

            logp(f"evaluation before log {val_metric}")
            metrics = (
                get_status(
                    config = config,
                    model = model,
                    step = global_opt_step,
                    training_time = training_time,
                    total_training_time = total_training_time,
                    inner_opt_step = None,
                    global_opt_step = global_opt_step,
                    loss_batch=loss_batch,
                    aux_loss_batch=aux_loss_batch,
                )
                | val_metric
            )

            metric_logger.log(metrics)

            logp(f"reached barrier, waiting for partial validation and metric logging to complete")
            # dist.barrier(device_ids=[rank])

            # === save checkpoint ===
            logp(f"saving checkpoint")
            ckpt_path = os.path.join(config.ckpt.checkpoint_path, f"global_opt_step_{int(global_opt_step)}")

            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                outer_optimizer=outer_optimizer,
                loss=loss_batch.item(),
                outer_scaler=outer_scaler,
                data_loader=train_dataloader,
                save_global_state= rank == 0,
                rank=rank,
            )

            if rank == 0:
                if config.ckpt.checkpoint_topk is not None:
                    ckpt_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, config.ckpt.checkpoint_topk)
                    if ckpt_deleted:
                        logp(f"Deleted old checkpoints: {ckpt_deleted}")
            
            logp(f"reached barrier, waiting for complete checkpoint saving")
            # dist.barrier(device_ids=[rank])

            # === Clean up ===
            loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
            aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
            gc.collect()
            torch.cuda.empty_cache()

            global_opt_step += 1

    except Exception as E:
        logger.error("Quit training", exc_info=True)
        cleanup()
        metric_logger.close()

        if rank == 0:
            torch.save(global_model.state_dict(), "mycelia_final.pt")

if __name__ == "__main__":
    args = parse_args()

    if args.path:
        config = ValidatorConfig.from_json(args.path)
    else:
        config = Validator()

    run(0, 1, config)