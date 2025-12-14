import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue
from threading import Lock, Thread

from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import MinerChainCommit, _subtensor_lock, commit_status
from mycelia.shared.checkpoint import (
    ModelMeta,
    compile_full_state_dict_from_path,
    delete_old_checkpoints,
    get_resume_info,
)
from mycelia.shared.client import submit_model
from mycelia.shared.config import MinerConfig, parse_args
from mycelia.shared.cycle import search_model_submission_destination, setup_chain_worker, wait_till
from mycelia.shared.helper import get_model_hash
from mycelia.shared.model import fetch_model_from_chain
from mycelia.shared.schema import sign_message
from mycelia.sn_owner.cycle import PhaseNames

configure_logging()
logger = structlog.get_logger(__name__)


# --- Job definitions ---


class JobType(Enum):
    DOWNLOAD = auto()
    SUBMIT = auto()
    COMMIT = auto()


@dataclass
class Job:
    job_type: JobType
    payload: dict | None = None
    phase_end_block: int | None = None


@dataclass
class SharedState:
    current_model_version: int | None = None
    current_model_hash: str | None = None
    latest_checkpoint_path: str | None = None
    lock: Lock = field(default_factory=Lock, repr=False)


class FileNotReadyError(RuntimeError):
    pass


# --- Scheduler service ---
def scheduler_service(
    config,
    download_queue: Queue,
    commit_queue: Queue,
    submit_queue: Queue,
    poll_fallback_seconds: float = 5.0,
):
    """
    Periodically checks whether to start download/submit phases and enqueues jobs.
    """
    while True:
        # --------- DOWNLOAD SCHEDULING ---------
        wait_till(config, phase_name=PhaseNames.distribute, poll_fallback_seconds=poll_fallback_seconds)
        download_queue.put(Job(job_type=JobType.DOWNLOAD))

        # --------- COMISSION SCHEDULING ---------
        _, phase_end_block = wait_till(
            config, phase_name=PhaseNames.commit, poll_fallback_seconds=poll_fallback_seconds
        )
        commit_queue.put(Job(job_type=JobType.COMMIT, phase_end_block=phase_end_block))

        # --------- SUBMISSION SCHEDULING ---------
        wait_till(config, phase_name=PhaseNames.submission, poll_fallback_seconds=poll_fallback_seconds)
        logger.info("A submission phase started, enqueue submit job")
        submit_queue.put(Job(job_type=JobType.SUBMIT))


# --- Workers ---
def download_worker(
    config,
    download_queue: Queue,
    current_model_version,
    current_model_hash,
    shared_state: SharedState,
):
    """
    Consumes DOWNLOAD jobs and runs the download phase logic.
    """
    while True:
        job = download_queue.get()
        try:
            # Read current version/hash snapshot
            with shared_state.lock:
                current_model_version = shared_state.current_model_version
                current_model_hash = shared_state.current_model_hash

            download_meta = fetch_model_from_chain(
                ModelMeta(global_ver=current_model_version, model_hash=current_model_hash), config, subtensor, wallet
            )

            if (
                not isinstance(download_meta, dict)
                or "model_version" not in download_meta
                or "model_hash" not in download_meta
            ):
                raise FileNotReadyError(f"download_meta from chain is not ready/invalid: {download_meta}")

            # Update shared state with new version/hash
            with shared_state.lock:
                shared_state.current_model_version = download_meta["model_version"]
                shared_state.current_model_hash = download_meta["model_hash"]

            delete_old_checkpoints(
                config.ckpt.validator_checkpoint_path,
                config.ckpt.checkpoint_topk,
            )

        except FileNotReadyError as e:
            logger.warning(f"[download_worker] File not ready error: {e}")

        except Exception as e:
            logger.error(f"[download_worker] Error while handling job: {e}")
            traceback.print_exc()

        finally:
            download_queue.task_done()
            logger.info(f"Phase <{PhaseNames.distribute}> obligation completed.")


def commit_worker(
    config,
    commit_queue: Queue,
    wallet,
    subtensor,
    shared_state: SharedState,
):
    """
    Consumes COMMIT model and runs the submission phase logic.
    """
    while True:
        job = commit_queue.get()
        try:
            _, _, latest_checkpoint_path = get_resume_info(rank=0, config=config)

            with shared_state.lock:
                shared_state.latest_checkpoint_path = latest_checkpoint_path

            if latest_checkpoint_path is None:
                raise FileNotReadyError("Not checkpoint found, skip commit.")

            model_path = f"{latest_checkpoint_path}/model.pt"
            model_hash = get_model_hash(compile_full_state_dict_from_path(model_path)).hex()
            with _subtensor_lock:
                current_block = subtensor.block

            n_blocks = max(1, job.phase_end_block - current_block)  # Ensure positive
            commited_message = commit_status(
                config,
                wallet,
                subtensor,
                MinerChainCommit(
                    expert_group=config.task.expert_group_id,
                    model_hash=model_hash,
                    # signed_model_hash=sign_message(origin_hotkey_ss58=wallet.hotkey, message=model_hash),
                ),
                encrypted=True,
                n_blocks=n_blocks,
            )
            logger.info(f"Committed with hash: {commited_message}.")
        except FileNotReadyError as e:
            logger.warning(f"[commit_worker] File not ready error: {e}")

        except Exception as e:
            logger.error(f"[commit_worker] Error while handling job: {e}")
            traceback.print_exc()

        finally:
            commit_queue.task_done()
            logger.info(f"Phase <{PhaseNames.commit}> obligation completed.")


def submit_worker(
    config,
    submit_queue: Queue,
    wallet,
    subtensor,
    shared_state: SharedState,
):
    """
    Consumes SUBMIT jobs and runs the submission phase logic.
    """
    while True:
        job = submit_queue.get()
        logger.info("B submit_worker picked up a job")
        try:
            logger.info("C submit_worker acquiring lock to read latest_checkpoint_path")
            with shared_state.lock:
                latest_checkpoint_path = shared_state.latest_checkpoint_path

            logger.info("D submit_worker released lock after reading latest_checkpoint_path")
            if latest_checkpoint_path is None:
                raise FileNotReadyError("Not checkpoint found, skip submission.")

            logger.info("E submit_worker searching for model submission destination")
            with _subtensor_lock:
                destination_axon = search_model_submission_destination(
                    wallet=wallet,
                    config=config,
                    subtensor=subtensor,
                )
                block = subtensor.block

            logger.info("F submit_worker found destination, submitting model")
            submit_model(
                url=f"http://{destination_axon.ip}:{destination_axon.port}/submit-checkpoint",
                token="",
                my_hotkey=wallet.hotkey,  # type: ignore
                target_hotkey_ss58=destination_axon.hotkey,
                block=block,
                model_path=f"{latest_checkpoint_path}/model.pt",
            )

        except FileNotReadyError as e:
            logger.warning(f"[submit_worker] File not ready error: {e}")

        except Exception as e:
            logger.error(f"[submit_worker] Error while handling job: {e}")
            traceback.print_exc()

        finally:
            submit_queue.task_done()
            logger.info(f"Phase <{PhaseNames.submission}> obligation completed.")


# --- Wiring it all together ---
def run_system(config, wallet, subtensor, current_model_version: int = -1, current_model_hash: str = "xxx"):
    download_queue = Queue()
    commit_queue = Queue()
    submit_queue = Queue()
    shared_state = SharedState(current_model_version, current_model_hash)

    # Start workers
    Thread(
        target=download_worker,
        args=(config, download_queue, current_model_version, current_model_hash, shared_state),
        daemon=True,
    ).start()

    Thread(
        target=commit_worker,
        args=(config, commit_queue, wallet, subtensor, shared_state),
        daemon=True,
    ).start()

    Thread(
        target=submit_worker,
        args=(config, submit_queue, wallet, subtensor, shared_state),
        daemon=True,
    ).start()

    # Start scheduler (runs in foreground)
    scheduler_service(
        config=config,
        download_queue=download_queue,
        commit_queue=commit_queue,
        submit_queue=submit_queue,
    )


if __name__ == "__main__":
    args = parse_args()

    if args.path:
        config = MinerConfig.from_path(args.path)
    else:
        config = MinerConfig()

    config.write()

    wallet, subtensor = setup_chain_worker(config)

    run_system(config, wallet, subtensor)
