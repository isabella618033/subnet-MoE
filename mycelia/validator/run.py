import os
import gc
import asyncio
import time
import logging
import json
import fsspec
import copy
import datetime
import secrets
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

import bittensor
from hivemind.averaging import DecentralizedAverager 

from mycelia.mock.validator.run import add_grad_noise
from mycelia.shared.chain import commit_status, ValidatorStatus
from mycelia.shared.config import MinerConfig, ValidatorConfig, parse_args
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
from mycelia.shared.chain import serve_axon
from mycelia.validator.aggregator import MinerScoreAggregator
from mycelia.validator.inter_validator_connection import connect_with_peers, build_averagers_from_buff, build_grad_buff_from_model, pack_grads, unpack_to_grads
from mycelia.validator.evaluator import run_evaluation, MinerEvalJob, load_model_from_path
from mycelia.shared.cycle import gather_validation_job, should_start_validation


configure_logging()
logger = structlog.get_logger(__name__)


def cleanup() -> None:
    """
    Cleans up the distributed training environment.

    Returns:
        None
    """
    torch.cuda.synchronize()

def setup_chain_worker(
    config
):
    wallet = bittensor.wallet(name = config.chain.coldkey_name, hotkey = config.chain.hotkey_name)
    subtensor = bittensor.subtensor(network = config.chain.network) 
    serve_axon(
        config = config,
        wallet = wallet,
        subtensor = subtensor,
    )
    return wallet, subtensor

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

async def aggregate_miner_gradient_change(
        base_model: nn.Module,
        global_model: nn.Module,
        device: torch.device,
        rank: int,
        logp: callable,
        outer_optimizer: torch.optim.Optimizer,
        miner_jobs: List[MinerEvalJob],
        score_aggregator: MinerScoreAggregator,        
):
    miner_models: Dict[str, nn.Module] = {}
    for miner_job in miner_jobs:
        if score_aggregator.is_in_top(uid = miner_job.uid, cutoff = 3, how = 'avg'): # TODO: change it to ema
            miner_models[miner_job.uid] = await asyncio.to_thread(load_model_from_path, miner_job.model_path, base_model)

    # each validator is only expected to validate 1 expert group at a time
    for uid, miner_model in miner_models.items():
        populate_global_grads_from_local(global_model, miner_model, weight = 1 / len(miner_models))

def sync_grad_across_validators(
    group_averagers: Dict[str | int, DecentralizedAverager],
    group_grad_buff_meta: Dict[str | int, Any]
):
    for group_id, avg in group_averagers.items():
        pack_grads(group_grad_buff_meta[group_id])
        info = avg.step(gather={"group": group_id},  allow_retries = False)
        unpack_to_grads(group_grad_buff_meta[group_id])
        
        logger.info(group_id, "->", ("averaged" if info else "no group"))
    
def run_global_optimization(
        model: nn.Module,
        global_model: nn.Module,
        device: torch.device,
        rank: int,
        logp: callable,
        outer_optimizer: torch.optim.Optimizer,
        miner_jobs: List[MinerEvalJob],
        score_aggregator: MinerScoreAggregator,
):
    # --- sync + outer step ---
    # keep global model on device for syncing/stepping, then move back to CPU
    global_model.to(device)

    old_shared_name, old_shared_sum = get_weight_sum(model, shared=True)
    old_expert_name, old_expert_sum = get_weight_sum(model, shared=False)

    logp("start syncing shared weights")

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

def run(rank: int, world_size: int, config: ValidatorConfig) -> None:
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
    
    global_opt_step = start_step

    # === set up score aggregator ===
    score_aggregator = MinerScoreAggregator()

    # === set up averager ===
    group_grad_buff_meta = build_grad_buff_from_model(model = model, expert_group_assignment = em.expert_group_assignment)

    dht = connect_with_peers()

    group_averagers = build_averagers_from_buff(
        group_buff_metas = group_grad_buff_meta, 
        dht = dht
    )

    # === set up chain worker ===
    wallet, subtensor = setup_chain_worker(config)
    
    commit_status(config, wallet, subtensor, ValidatorStatus(
        model_hash = 'xxx',
        model_version = global_opt_step,
        expert_group = 1,
        miner_seed = secrets.randbits(24) # this should reveal later
    ))

    # === training ===
    loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    training_time = 0
    total_training_time = 0
    training_start_time = None

    outer_optimizer.zero_grad()
    
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
            start_validation = False
            while not start_validation:
                start_validation, block_till = should_start_validation(config, subtensor) 
                if block_till > 0:
                    time.sleep((block_till) * 12)

            miner_jobs = gather_validation_job(
                config, subtensor, step = global_opt_step
            )
            logger.info('gathered miner job', global_opt_step = global_opt_step, miner_jobs = miner_jobs)

            # # === Download miner model and evaluate the miners ===
            # asyncio.run(run_evaluation(
            #     config = config,
            #     step = global_opt_step,
            #     device = model.device,
            #     miners = miner_jobs,
            #     score_aggregator = score_aggregator,
            #     base_model = model,
            #     tokenizer = tokenizer
            # ))

            # logp(msg = "eval result" + str(score_aggregator.uid_score_pairs()))

            # # === aggragate miner gradient change locally ===
            # logp(f"aggregate miner gradient change", loss_batch, aux_loss_batch)
            # asyncio.run(aggregate_miner_gradient_change(
            #     base_model = model,
            #     global_model = global_model,
            #     device = device,
            #     rank = rank,
            #     logp = logp,
            #     outer_optimizer = outer_optimizer,
            #     miner_jobs = miner_jobs,
            #     score_aggregator = score_aggregator
            # ))

            # # === aggragate miner gradient change ===
            # logp(f"sync gradient across validators", loss_batch, aux_loss_batch)
            # sync_grad_across_validators(
            #     group_averagers,
            #     group_grad_buff_meta
            # )

            # # === global optimizer ===
            # run_global_optimization(
            #     model = model,
            #     global_model = global_model,
            #     device = device,
            #     rank = rank,
            #     logp = logp,
            #     outer_optimizer = outer_optimizer,
            #     miner_jobs = miner_jobs,
            #     score_aggregator = score_aggregator
            # )

            # # === validation and log metric ===
            # logp(f"start local evaluation")
            # val_metric = evaluate_model(
            #     rank=rank, step=global_opt_step, model=model, eval_dataloader=eval_dataloader, device=device
            # )

            # logp(f"evaluation before log {val_metric}")
            # metrics = (
            #     get_status(
            #         config = config,
            #         model = model,
            #         step = global_opt_step,
            #         training_time = training_time,
            #         total_training_time = total_training_time,
            #         inner_opt_step = None,
            #         global_opt_step = global_opt_step,
            #         loss_batch=loss_batch,
            #         aux_loss_batch=aux_loss_batch,
            #     )
            #     | val_metric
            # )

            # metric_logger.log(metrics)

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

            # === Comit to chain for new model ===
            commit_status(config, wallet, subtensor, ValidatorStatus(
                model_hash = 'xxx',
                model_version = global_opt_step,
                expert_group = 1,
                miner_seed = secrets.randbits(24) # this should reveal later
            ))

            if rank == 0:
                if config.ckpt.checkpoint_topk is not None:
                    ckpt_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, config.ckpt.checkpoint_topk)
                    if ckpt_deleted:
                        logp(f"Deleted old checkpoints: {ckpt_deleted}")
            
            # # === Clean up ===
            # loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
            # aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
            # gc.collect()
            # torch.cuda.empty_cache()%

            time.sleep(60)

            global_opt_step += 1

    except Exception as E:
        logger.error("Quit training", exc_info=True)
        cleanup()
        metric_logger.close()
        for idx, a in group_averagers.items():
            a.shutdown()

        if rank == 0:
            torch.save(global_model.state_dict(), "mycelia_final.pt")

if __name__ == "__main__":
    args = parse_args()

    if args.path:
        config = ValidatorConfig.from_path(args.path)
    else:
        config = ValidatorConfig()

    run(0, 1, config)