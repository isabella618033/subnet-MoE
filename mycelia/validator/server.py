import contextlib
import glob
import hashlib
import mimetypes
import os
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import bittensor
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import _subtensor_lock
from mycelia.shared.checkpoint import (
    delete_old_checkpoints_by_hotkey,
    get_resume_info,
)
from mycelia.shared.config import ValidatorConfig, parse_args
from mycelia.shared.schema import (
    construct_block_message,
    construct_model_message,
    verify_message,
)

SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
configure_logging()
logger = structlog.get_logger(__name__)
app = FastAPI(title="Checkpoint Sender", version="1.0.0")

# ---- Settings via environment variables ----
AUTH_TOKEN = os.getenv("AUTH_TOKEN")  # optional bearer token for auth


# ---- Helpers ----
def require_auth(authorization: str | None) -> None:
    if not AUTH_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")


def file_response_for(path: Path, download_name: str | None = None) -> FileResponse:
    logger.info("file response for", path=path)
    if not path.exists() or not path.is_file():
        logger.info("file response for, path dosent exist", path.exists(), path.is_file())
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # Default to binary; guess if we can
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    stat = path.stat()
    last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    headers = {
        # Encourage efficient delivery; Starlette handles Range requests automatically for FileResponse
        "Cache-Control": "no-store",
        "Last-Modified": last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT"),
        # Nginx/Envoy tip to avoid buffering big files
        "X-Accel-Buffering": "no",
    }

    return FileResponse(
        path=path,
        media_type=media_type,
        filename=download_name or path.name,  # sets Content-Disposition: attachment; filename="..."
        headers=headers,
    )


# Optional: centralize where uploads go (env var overrides default)
def get_upload_root() -> Path:
    root = os.getenv("CHECKPOINT_INBOX", "/var/lib/validator/checkpoints/incoming")
    p = Path(root).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- Schemas ----
class SendBody(BaseModel):
    # Optional override; if not provided, uses CHECKPOINT_PATH
    path: str | None = None
    # Optional download file name (e.g., "model-v1.ckpt")
    download_name: str | None = None


@app.on_event("startup")
async def _startup():
    configure_logging()  # <— configure ONCE
    structlog.get_logger(__name__).info("startup_ok")


# ---- Routes ----
@app.get("/")
async def index(request: Request):
    return JSONResponse(
        {
            "name": config.chain.hotkey_ss58,
            "service": "Checkpoint Sender",
            "version": "0.0.0",
            "endpoints": {
                "GET /ping": "Health check",
                "GET /checkpoint": "Download the configured checkpoint",
                "POST /send-checkpoint": {
                    "body": {"path": "optional string", "download_name": "optional string"},
                    "desc": "Download checkpoint (optionally overriding path/name)",
                },
            },
            "auth": "Set AUTH_TOKEN env var to require 'Authorization: Bearer <token>'",
        }
    )


@app.get("/ping")
async def ping():
    """Health check."""
    print("ping")
    logger = structlog.get_logger(__name__)
    logger.info("ping")  # <— this will now print
    return {"status": "ok"}


@app.get("/status")
async def status():
    # respond to what is the current status of validator, the newest model id
    pass


# miners / client get model from validator
@app.get("/get-checkpoint")
async def get_checkpoint(
    authorization: str | None = Header(default=None),
    target_hotkey_ss58: str = Form(None, description="Receiver's hotkey"),
    origin_hotkey_ss58: str = Form(None, description="Sender's hotkey"),
    block: int = Form(
        None, description="The block that the message was sent."
    ),  # insecure, do not use this field for validation, TODO: change it to block hash?
    signature: str = Form(None, description="Signed message"),
    expert_group_ids: list[int] | None = Form(None, description="List of expert groups to fetch"),
):
    """GET to download the configured checkpoint immediately."""
    logger.info("checkpoint, A")
    require_auth(authorization)
    logger.info("checkpoint, B")

    verify_message(
        origin_hotkey_ss58=origin_hotkey_ss58,
        message=construct_block_message(
            target_hotkey_ss58=target_hotkey_ss58,  # TODO: assert hotkey is valid within the metagraph
            block=block,  # TODO: change to block hash and assert it is not too far from current
        ),
        signature_hex=signature,
    )

    resume, model_meta, latest_checkpoint_path = get_resume_info(rank=0, config=config)
    logger.info("checkpoint, C", latest_checkpoint_path)

    if not latest_checkpoint_path:
        raise HTTPException(status_code=500, detail="CHECKPOINT_PATH env var is not set")

    # Case 1: no expert group requested → old behavior
    if expert_group_ids is None:
        latest_checkpoint_path = os.path.join(latest_checkpoint_path, "model.pt")
        logger.info(f"checkpoint, last {latest_checkpoint_path}")
        result = file_response_for(Path(latest_checkpoint_path), f"step{model_meta.global_ver}")
        return result

    else:
        # Case 2: specific expert group requested → zip pre-split files directly
        ckpt_dir = latest_checkpoint_path  # directory that contains group_*.pt / shared*.pt
        expert_group_ids.sort()

        # Deterministic zip name for this combination of groups + step
        zip_name = f"expert_group_step{model_meta.global_ver}_{','.join(str(x) for x in expert_group_ids)}.zip"
        zip_path = os.path.join(ckpt_dir, zip_name)

        # If the zip already exists, skip re-creating it
        if os.path.exists(zip_path):
            return FileResponse(
                zip_path,
                filename=zip_name,
                media_type="application/zip",
            )

        # All files related to this expert group, e.g.:
        #   group_{gid}.pt
        #   group_{gid}_rank0.pt
        #   group_{gid}_shard1.pt
        model_files = []
        for expert_group_id in expert_group_ids:
            group_pattern = os.path.join(ckpt_dir, f"group_{expert_group_id}*.pt")
            model_files += glob.glob(group_pattern)

        # All shared files, e.g.:
        #   shared.pt
        #   shared_rank0.pt
        shared_pattern = os.path.join(ckpt_dir, "shared*.pt")
        model_files += glob.glob(shared_pattern)

        # Create a temp dir for the zip
        tmp_dir = tempfile.mkdtemp(
            prefix=zip_name.replace(".zip", "_"),
            dir=ckpt_dir,
        )

        zip_name = f"expert_group_{','.join(str(x) for x in expert_group_ids)}.zip"
        zip_path = os.path.join(tmp_dir, zip_name)

        # Write all related files into the zip (group + shared)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in model_files:
                arcname = os.path.basename(path)
                zf.write(path, arcname=arcname)

        return FileResponse(
            zip_path,
            filename=zip_name,
            media_type="application/zip",
        )


# miners submit checkpoint
@app.post("/submit-checkpoint")
async def submit_checkpoint(
    authorization: str | None = Header(default=None),
    target_hotkey_ss58: str = Form(None, description="Receiver's hotkey"),
    origin_hotkey_ss58: str = Form(None, description="Sender's hotkey"),
    signature: str = Form(None, description="Signed message"),
    file: UploadFile = File(..., description="The checkpoint file, e.g. model.pt"),
):
    """
    POST a checkpoint to the validator.

    Accepts multipart/form-data with fields:
      - step (int, required)
      - checksum_sha256 (str, optional)
      - file (UploadFile, required)
    """
    require_auth(authorization)

    # Basic filename safety (avoid path tricks). We'll still rename it server-side.
    original_name = file.filename or ""
    if not SAFE_NAME.match(Path(original_name).name):
        raise HTTPException(status_code=400, detail="Unsafe filename")

    # Restrict to common checkpoint extensions (adjust if you use others)
    allowed_exts = {".pt", ".bin", ".ckpt", ".safetensors", ".tar"}
    ext = Path(original_name).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

    # Stream write + compute SHA256
    block = subtensor.block
    model_name = f"hotkey_{origin_hotkey_ss58}_block_{block}.pt"
    hasher = hashlib.sha256()
    bytes_written = 0
    dest_path = config.ckpt.miner_submission_path / model_name
    try:
        async with aiofiles.open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MiB
                if not chunk:
                    break
                hasher.update(chunk)
                await out.write(chunk)
                bytes_written += len(chunk)

    except Exception as e:
        # Clean up partial writes
        with contextlib.suppress(Exception):
            dest_path.unlink(missing_ok=True)
        logger.exception("Failed to store uploaded checkpoint")
        raise HTTPException(status_code=500, detail="Failed to store file") from e
    finally:
        await file.close()

    computed = hasher.hexdigest()

    logger.info(f"Stored checkpoint at {dest_path} ({bytes_written} bytes) sha256={computed}")

    verify_message(
        origin_hotkey_ss58=origin_hotkey_ss58,
        message=construct_model_message(model_path=dest_path, target_hotkey_ss58=target_hotkey_ss58, block=block),
        signature_hex=signature,
    )

    logger.info(f"Verified submission at {dest_path}.")

    delete_old_checkpoints_by_hotkey(config.ckpt.miner_submission_path)

    return {
        "status": "ok",
        "stored_path": str(dest_path),
        "bytes": bytes_written,
        "sha256": computed,
    }


if __name__ == "__main__":
    args = parse_args()

    global config
    global subtensor

    if args.path:
        config = ValidatorConfig.from_path(args.path)
    else:
        config = ValidatorConfig()

    config.write()

    subtensor = bittensor.Subtensor(network=config.chain.network)

    uvicorn.run(app, host=config.chain.ip, port=config.chain.port)
