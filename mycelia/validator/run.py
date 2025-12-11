import asyncio
import copy
import gc
import os
import secrets
from typing import Any

import bittensor
import torch
import torch.nn as nn
from hivemind.averaging import DecentralizedAverager
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from mycelia.miner.train_helper import get_status
from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import ValidatorChainCommit, commit_status, setup_chain_worker
from mycelia.shared.checkpoint import (
    ModelMeta,
    delete_old_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from mycelia.shared.config import ValidatorConfig, parse_args
from mycelia.shared.cycle import gather_validation_job, get_combined_validator_seed, wait_till
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.evaluate import evaluate_model
from mycelia.shared.expert_manager import (
    ExpertManager,
    get_weight_sum,
    populate_global_grads_from_local,
)
from mycelia.shared.helper import get_model_hash, get_nested_attr
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.model import load_model
from mycelia.shared.modeling.mycelia import get_base_tokenizer
from mycelia.sn_owner.cycle import PhaseNames
from mycelia.validator.aggregator import MinerScoreAggregator
from mycelia.validator.evaluator import (
    MinerEvalJob,
    load_model_from_path,
    run_evaluation,
)
from mycelia.validator.inter_validator_connection import (
    build_averagers_from_buff,
    build_grad_buff_from_model,
    connect_with_peers,
    pack_grads,
    unpack_to_grads,
)

configure_logging()
logger = structlog.get_logger(__name__)


def cleanup() -> None:
    """
    Cleans up the distributed training environment.

    Returns:
        None
    """
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
    latest_checkpoint_path = None

    # === model & Experts manager ===
    logger.info(f"rank {rank} setup training - load model and expert manager")
    expert_manager = ExpertManager(config)
    base_model, model_meta = load_model(rank, config, expert_manager, subtensor, wallet, current_model_meta)
    base_model = base_model.to(device)
    global_model = copy.deepcopy(base_model).cpu()

    # === optimizers ===
    logger.info(f"rank {rank} setup training - load optimizer")
    outer_optimizer = torch.optim.SGD(
        global_model.named_parameters(),
        lr=config.opt.outer_lr,
        momentum=config.opt.outer_momentum,
        nesterov=True,
    )

    # === scaler ===
    logger.info(f"rank {rank} setup training - load scaler")
    outer_scaler = torch.amp.GradScaler(
        "cuda", enabled=(get_nested_attr(config, "model.precision", "") == "fp16-mixed")
    )

    # === dataloader ===
    logger.info(f"rank {rank} setup training - load dataloader")
    train_dataloader = get_dataloader(config, rank=rank, world_size=config.task.data.world_size, tokenizer=tokenizer)

    # === load checkpoint (if any) ===
    logger.info(f"rank {rank} setup training - load past checkpoint")
    if get_nested_attr(config, "resume_from_ckpt", False) and resume and latest_checkpoint_path:
        _ = load_checkpoint(
            config=config,
            checkpoint_path=latest_checkpoint_path,
            outer_optimizer=outer_optimizer,
            outer_scaler=outer_scaler,
            rank=rank,
            device=device,
            data_loader=train_dataloader,
        )

    logger.info(f"rank {rank} setup_training - completed successfully!")
    return (
        base_model,
        global_model,
        outer_optimizer,
        outer_scaler,
        model_meta.global_ver,
        expert_manager,
        train_dataloader,
    )


async def aggregate_miner_gradient_change(
    base_model: nn.Module,
    global_model: nn.Module,
    device: torch.device,
    rank: int,
    outer_optimizer: torch.optim.Optimizer,
    miner_jobs: list[MinerEvalJob],
    score_aggregator: MinerScoreAggregator,
):
    global_model.to(device)
    miner_models: dict[str, nn.Module] = {}
    for miner_job in miner_jobs:
        if score_aggregator.is_in_top(uid=miner_job.uid, cutoff=3, how="avg"):  # TODO: change it to ema
            miner_models[miner_job.uid] = await asyncio.to_thread(
                load_model_from_path, miner_job.model_path, base_model, device
            )

    # each validator is only expected to validate 1 expert group at a time
    for _, miner_model in miner_models.items():
        populate_global_grads_from_local(global_model, miner_model, weight=1 / len(miner_models))


def sync_grad_across_validators(
    group_averagers: dict[str | int, DecentralizedAverager], group_grad_buff_meta: dict[str | int, Any]
):
    for group_id, avg in group_averagers.items():
        if avg.total_size <= 0:
            logger.info("skip averager", group_id=group_id, mode=avg.mode, total_size=avg.total_size)
            continue

        logger.info("begin sync grad across validator", group=group_id, mode=avg.mode)
        pack_grads(group_grad_buff_meta[group_id])
        info = avg.step(allow_retries=False)
        unpack_to_grads(group_grad_buff_meta[group_id])

        logger.info(
            "sync grad across validator", group=group_id, mode=avg.mode, successes=("averaged" if info else "no group")
        )

    return


def run_global_optimization(
    model: nn.Module,
    global_model: nn.Module,
    device: torch.device,
    rank: int,
    outer_optimizer: torch.optim.Optimizer,
    miner_jobs: list[MinerEvalJob],
    score_aggregator: MinerScoreAggregator,
):
    # --- sync + outer step ---
    # keep global model on device for syncing/stepping, then move back to CPU
    global_model.to(device)

    old_shared_name, old_shared_sum = get_weight_sum(model, shared=True)
    old_expert_name, old_expert_sum = get_weight_sum(model, shared=False)

    logger.info("start syncing shared weights")

    outer_optimizer.step()
    outer_optimizer.zero_grad()

    # copy updated global weights back into the worker model safely
    with torch.no_grad():
        model.load_state_dict(global_model.state_dict(), strict=True)

    new_shared_name, new_shared_sum = get_weight_sum(model, shared=True)
    new_expert_name, new_expert_sum = get_weight_sum(model, shared=False)

    logger.info(
        "outer optimizer step (shared)",
        param_name=old_shared_name,
        old_sum=round(float(old_shared_sum), 6),
        new_sum=round(float(new_shared_sum), 6),
    )
    logger.info(
        "outer optimizer step (expert)",
        param_name=old_expert_name,
        old_sum=round(float(old_expert_sum), 6),
        new_sum=round(float(new_expert_sum), 6),
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
    if rank == 0:
        config.write()

    # === create checkpoint directory ===
    os.makedirs(config.ckpt.base_checkpoint_path, exist_ok=True)
    os.makedirs(config.ckpt.checkpoint_path, exist_ok=True)
    os.makedirs(config.log.base_metric_path, exist_ok=True)
    os.makedirs(config.ckpt.miner_submission_path, exist_ok=True)

    # === set up chain worker ===
    wallet, subtensor = setup_chain_worker(config)

    # === set logging ===
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
        base_model,
        global_model,
        outer_optimizer,
        outer_scaler,
        start_step,
        expert_manager,
        train_dataloader,
    ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta=None)

    global_opt_step = start_step

    # === set up score aggregator ===
    score_aggregator = MinerScoreAggregator()

    # === set up averager ===
    group_grad_buff_meta = build_grad_buff_from_model(
        model=base_model, expert_group_assignment=expert_manager.expert_group_assignment
    )

    dht = connect_with_peers()

    group_averagers = build_averagers_from_buff(group_buff_metas=group_grad_buff_meta, dht=dht)

    commit_status(
        config,
        wallet,
        subtensor,
        ValidatorChainCommit(
            model_hash="xxx",
            model_version=global_opt_step,
            expert_group=1,
            miner_seed=0,  # this should reveal later
        ),
    )

    # === training ===
    loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    training_time = 0
    total_training_time = 0

    outer_optimizer.zero_grad()

    current_model_hash = "xxx"

    try:
        while True:
            # for each step, we run 1 backward
            # for each inner_opt_step, we run local optimization; gradient_accumulation_steps = 1 real step
            # for each global_opt_interval number of inner_opt_step, we synchronise weight from different ddp worker, and then run global optimization

            # === Wait till commit phase to submit random seed ===
            _, phase_end_block = wait_till(config, PhaseNames.commit)
            logger.info("(0) Commit new seed for next validation")

            commit_status(
                config,
                wallet,
                subtensor,
                ValidatorChainCommit(
                    model_hash=current_model_hash,
                    model_version=global_opt_step,
                    expert_group=1,
                    miner_seed=secrets.randbits(24),  # this should reveal later
                    encrypt=True,
                    n_block=subtensor.block - phase_end_block,
                ),
            )

            # === Wait till validation phase to start the validation procedure ===
            wait_till(config, PhaseNames.validate)
            logger.info("(1) Start validation pahse")

            # === Get miner ===
            logger.info("(2) Gathering miner job")
            miner_jobs = gather_validation_job(config, subtensor, step=global_opt_step)
            logger.info("miner job(s)", global_opt_step=global_opt_step, miner_jobs=miner_jobs)

            # === Get miner model and evaluate the miners ===
            logger.info("(3) Evaluating miners")
            asyncio.run(
                run_evaluation(
                    config=config,
                    step=global_opt_step,
                    device=base_model.device,
                    miners=miner_jobs,
                    score_aggregator=score_aggregator,
                    base_model=base_model,
                    tokenizer=tokenizer,
                    combinded_seed=get_combined_validator_seed(config, subtensor),
                )
            )

            logger.info("eval result", scores=score_aggregator.uid_score_pairs())

            # === aggragate miner gradient change locally ===
            logger.info("(4) Aggregating miner gradient change")
            asyncio.run(
                aggregate_miner_gradient_change(
                    base_model=base_model,
                    global_model=global_model,
                    device=torch.device("cpu"),  # all gradient aggregation done on cpu
                    rank=rank,
                    outer_optimizer=outer_optimizer,
                    miner_jobs=miner_jobs,
                    score_aggregator=score_aggregator,
                )
            )

            # === wait till merging phase and aggragate miner gradient change ===
            wait_till(config, PhaseNames.merge)
            logger.info("(5) Syncing gradient across validators")
            sync_grad_across_validators(group_averagers, group_grad_buff_meta)

            # === global optimizer ===
            logger.info("(6) Running global model optimisation step")
            run_global_optimization(
                model=base_model,
                global_model=global_model,
                device=device,
                rank=rank,
                outer_optimizer=outer_optimizer,
                miner_jobs=miner_jobs,
                score_aggregator=score_aggregator,
            )

            # === save checkpoint ===
            logger.info("(7) Saving checkpoint")
            ckpt_path = os.path.join(config.ckpt.checkpoint_path, f"globalopt_{int(global_opt_step)}")

            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=base_model,
                outer_optimizer=outer_optimizer,
                loss=loss_batch.item(),
                outer_scaler=outer_scaler,
                data_loader=train_dataloader,
                save_global_state=rank == 0,
                rank=rank,
                expert_manager=expert_manager,
                save_model_by_expert_group=True,
            )

            # === Comit to chain for new model ===
            current_model_hash = get_model_hash(ckpt_path / "model.pt")
            commit_status(
                config,
                wallet,
                subtensor,
                ValidatorChainCommit(
                    model_hash=current_model_hash,
                    model_version=global_opt_step,
                    expert_group=1,
                    miner_seed=0,
                ),
            )

            if rank == 0:
                if config.ckpt.checkpoint_topk is not None:
                    ckpt_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, config.ckpt.checkpoint_topk)
                    if ckpt_deleted:
                        logger.info(f"Deleted old checkpoints: {ckpt_deleted}")

            # === validation and log metric ===
            logger.info("(8) Start local evaluation")
            val_metric = evaluate_model(
                rank=rank, step=global_opt_step, model=global_model, eval_dataloader=eval_dataloader, device=device
            )

            metrics = (
                get_status(
                    config=config,
                    model=base_model,
                    step=global_opt_step,
                    training_time=training_time,
                    total_training_time=total_training_time,
                    inner_opt_step=None,
                    global_opt_step=global_opt_step,
                    loss_batch=loss_batch,
                    aux_loss_batch=aux_loss_batch,
                )
                | val_metric
            )

            metric_logger.log(metrics)

            # # === Clean up ===
            # loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
            # aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
            # gc.collect()
            # torch.cuda.empty_cache()%

            global_opt_step += 1

    except Exception:
        logger.error("Quit training", exc_info=True)
        cleanup()
        metric_logger.close()
        for _, a in group_averagers.items():
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
