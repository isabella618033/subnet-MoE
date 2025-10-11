# app.py
import os
import re
import aiofiles
import hashlib
import mimetypes
import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import uvicorn

from fastapi import FastAPI, HTTPException, Header, Request, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from mycelia.config import MinerConfig, ValidatorConfig
from mycelia.shared.checkpoint import (
    get_resume_info,
    load_checkpoint,
)

from mycelia.config import MinerConfig, parse_args
from mycelia.shared.app_logging import structlog, configure_logging
from mycelia.shared.app_logging import configure_logging, structlog


SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
configure_logging()
logger = structlog.get_logger(__name__)
app = FastAPI(title="Checkpoint Sender", version="1.0.0")

# ---- Settings via environment variables ----
AUTH_TOKEN = os.getenv("AUTH_TOKEN")            # optional bearer token for auth

# ---- Helpers ----
def require_auth(authorization: Optional[str]) -> None:
    if not AUTH_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")


def file_response_for(path: Path, download_name: Optional[str] = None) -> FileResponse:
    logger.info("file response for", path = path)
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
    path: Optional[str] = None
    # Optional download file name (e.g., "model-v1.ckpt")
    download_name: Optional[str] = None


@app.on_event("startup")
async def _startup():
    configure_logging()  # <— configure ONCE
    structlog.get_logger(__name__).info("startup_ok")

# ---- Routes ----

@app.get("/")
async def index(request: Request):
    return JSONResponse(
        {
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
async def get_checkpoint(authorization: Optional[str] = Header(default=None)):
    """GET to download the configured checkpoint immediately."""
    logger.info(f"checkpoint, A")
    require_auth(authorization)
    logger.info(f"checkpoint, B")
    resume, start_step, latest_checkpoint_path = get_resume_info(rank = 0, config = config)
    logger.info(f"checkpoint, C", latest_checkpoint_path)

    if not latest_checkpoint_path:
        raise HTTPException(status_code=500, detail="CHECKPOINT_PATH env var is not set")
    
    latest_checkpoint_path = os.path.join(latest_checkpoint_path, f"model.pt")
    logger.info(f"checkpoint, last {latest_checkpoint_path}")
    result = file_response_for(Path(latest_checkpoint_path), f"step{start_step}")
    return result


# miners submit checkpoint
@app.post("/submit-checkpoint")
async def submit_checkpoint(
    authorization: Optional[str] = Header(default=None),
    step: int = Form(...),
    uid: str = Form(..., description="Unique client/user identifier"),
    hotkey: Optional[str] = Form(None, description="Optional client routing key"),
    checksum_sha256: Optional[str] = Form(None, description="Optional SHA256 hex digest for integrity check"),
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
    model_name = f"uid{uid}_{hotkey}.pt"
    hasher = hashlib.sha256()
    bytes_written = 0
    dest_path = config.vali.miner_submission_path / model_name
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
    if checksum_sha256 and checksum_sha256.lower() != computed.lower():
        # Integrity check failed; remove the file
        with contextlib.suppress(Exception):
            dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Checksum mismatch")

    logger.info(f"Stored checkpoint for step={step} at {dest_path} ({bytes_written} bytes) sha256={computed}")

    # Optionally, write a small metadata file next to it
    meta_path = config.vali.miner_submission_path / f"uid{uid}_{hotkey}_metadata.json"
    try:
        async with aiofiles.open(meta_path, "w") as meta:
            await meta.write(
                (
                    '{'
                    f'"step": {step}, '
                    f'"filename": "{dest_path.name}", '
                    f'"bytes": {bytes_written}, '
                    f'"sha256": "{computed}"'
                    '}'
                )
            )
    except Exception:
        # Non-fatal; metadata can be recreated
        logger.warning("Failed to write metadata.json", exc_info=True)

    return {
        "status": "ok",
        "step": step,
        "stored_path": str(dest_path),
        "bytes": bytes_written,
        "sha256": computed,
    }


if __name__ == "__main__":
    args = parse_args()

    global config
    
    if args.path:
        config = ValidatorConfig.from_json(args.path)
    else:
        config = ValidatorConfig()

    uvicorn.run(app, host=config.chain.ip, port=config.chain.port)  
