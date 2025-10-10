# app.py
import os
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import uvicorn

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from mycelia.config import MinerConfig
from mycelia.shared.checkpoint import (
    get_resume_info,
    load_checkpoint,
)

from mycelia.config import MinerConfig, parse_args
from mycelia.shared.app_logging import structlog, configure_logging
from mycelia.shared.app_logging import configure_logging, structlog

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
@app.get("/ping")
async def ping():
    """Health check."""
    print("ping")
    logger = structlog.get_logger(__name__)
    logger.info("ping")  # <— this will now print
    return {"status": "ok"}


@app.get("/checkpoint")
async def get_checkpoint(authorization: Optional[str] = Header(default=None)):
    """GET to download the configured checkpoint immediately."""
    require_auth(authorization)
    config = MinerConfig()
    resume, start_step, latest_checkpoint_path = get_resume_info(rank = 0, config = config)

    if not latest_checkpoint_path:
        raise HTTPException(status_code=500, detail="CHECKPOINT_PATH env var is not set")
    
    latest_checkpoint_path = os.path.join(latest_checkpoint_path, f"model.pt")
    logger.info(f"checkpoint, last {latest_checkpoint_path}")
    result = file_response_for(Path(latest_checkpoint_path), f"step{start_step}")
    return result

# Optional: simple index for convenience
@app.get("/")
async def index(request: Request):
    return JSONResponse(
        {
            "service": "Checkpoint Sender",
            "version": "1.0.0",
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

if __name__ == "__main__":
    args = parse_args()

    if args.path:
        config = MinerConfig.from_json(args.path)
    else:
        config = MinerConfig()

    uvicorn.run(app, host=config.chain.ip, port=config.chain.port)  
