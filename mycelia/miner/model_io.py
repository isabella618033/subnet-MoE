from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue
from threading import Thread

from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import MinerChainCommit, commit_status
from mycelia.shared.checkpoint import delete_old_checkpoints, get_resume_info
from mycelia.shared.client import submit_model
from mycelia.shared.config import MinerConfig, parse_args
from mycelia.shared.cycle import search_model_submission_destination, setup_chain_worker, wait_till
from mycelia.shared.helper import get_model_hash
from mycelia.shared.model import fetch_model_from_chain
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

        # --------- SUBMISSION SCHEDULING ---------
        _, phase_end_block = wait_till(
            config, phase_name=PhaseNames.commit, poll_fallback_seconds=poll_fallback_seconds
        )
        commit_queue.put(Job(job_type=JobType.COMMIT, phase_end_block=phase_end_block))

        # --------- SUBMISSION SCHEDULING ---------
        wait_till(config, phase_name=PhaseNames.submit, poll_fallback_seconds=poll_fallback_seconds)
        submit_queue.put(Job(job_type=JobType.SUBMIT))


# --- Workers ---
def download_worker(
    config,
    download_queue: Queue,
    current_model_version,
    current_model_hash,
):
    """
    Consumes DOWNLOAD jobs and runs the download phase logic.
    """
    while True:
        job = download_queue.get()
        try:
            # --------- DOWNLOAD PHASE ---------
            fetch_model_from_chain(current_model_version, current_model_hash, config)
            delete_old_checkpoints(
                config.ckpt.validator_checkpoint_path,
                config.ckpt.checkpoint_topk,
            )
        except Exception as e:
            # TODO: log/record failure, send alert, retry, etc.
            print(f"[download_worker] Error while handling job: {e}")
        finally:
            download_queue.task_done()


def commit_worker(
    config,
    commit_queue: Queue,
    wallet,
    subtensor,
):
    """
    Consumes COMMIT model and runs the submission phase logic.
    """
    while True:
        job = commit_queue.get()
        try:
            # --------- SUBMISSION PHASE ---------
            _, _, latest_checkpoint_path = get_resume_info(rank=0, config=config)
            model_path = (f"{latest_checkpoint_path}/model.pt",)
            model_hash = get_model_hash(model_path)
            commit_status(
                config,
                wallet,
                subtensor,
                MinerChainCommit(expert_group=config.task.expert_group_id, model_hash=model_hash),
                encrypted=True,
                n_blocks=job.phase_end_block - subtensor.block,
            )

        except Exception as e:
            # TODO: log/record failure, send alert, retry, etc.
            print(f"[commit_worker] Error while handling job: {e}")
        finally:
            commit_queue.task_done()


def submit_worker(
    config,
    submit_queue: Queue,
    wallet,
    subtensor,
):
    """
    Consumes SUBMIT jobs and runs the submission phase logic.
    """
    while True:
        job = submit_queue.get()
        try:
            # --------- SUBMISSION PHASE ---------
            _, _, latest_checkpoint_path = get_resume_info(rank=0, config=config)

            destination_axon = search_model_submission_destination(
                wallet=wallet,
                config=config,
                subtensor=subtensor,
            )

            submit_model(
                url=f"http://{destination_axon.ip}:{destination_axon.port}/submit-checkpoint",
                token="",
                my_hotkey=wallet.hotkey,  # type: ignore
                target_hotkey_ss58=destination_axon.hotkey,
                block=subtensor.block,
                model_path=f"{latest_checkpoint_path}/model.pt",
            )
        except Exception as e:
            # TODO: log/record failure, send alert, retry, etc.
            print(f"[submit_worker] Error while handling job: {e}")
        finally:
            submit_queue.task_done()


# --- Wiring it all together ---
def run_system(config, wallet, subtensor, current_model_version, current_model_hash):
    download_queue = Queue()
    commit_queue = Queue()
    submit_queue = Queue()

    # Start workers
    Thread(
        target=download_worker,
        args=(config, download_queue, current_model_version, current_model_hash),
        daemon=True,
    ).start()

    Thread(
        target=commit_worker,
        args=(config, commit_queue, wallet, subtensor),
        daemon=True,
    ).start()

    Thread(
        target=submit_worker,
        args=(config, submit_queue, wallet, subtensor),
        daemon=True,
    ).start()

    # Start scheduler (runs in foreground)
    scheduler_service(
        config=config,
        download_queue=download_queue,
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
