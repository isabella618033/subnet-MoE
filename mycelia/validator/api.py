# llm_weightnet/validator/api.py
from __future__ import annotations

import hashlib
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse

from mycelia.settings import Settings
from mycelia.shared.model import export_model_artifact, TinyMLP
from mycelia.shared.app_logging import structlog
from mycelia.shared.storage import put_bytes  # optional if you want to also publish to remote storage

logger = structlog.get_logger(__name__)

app = FastAPI(title="Validator API", version="0.1.0")
_settings = Settings()

# In-memory singleton model for example. In practice, load your canonical model here.
MODEL = TinyMLP()


@app.get("/v1/health")
def health():
    return {"status": "ok", "role": "validator"}


@app.get("/v1/model/manifest")
def model_manifest():
    """
    Returns a *manifest* telling miners where to fetch the model.
    You can choose to serve bytes directly (artifact_bytes_url) or via object storage (artifact_uri).
    Below we serve inline via /v1/model/bytes and also (optionally) publish to storage.
    """
    blob, fmt = export_model_artifact(MODEL)
    sha = hashlib.sha256(blob).hexdigest()

    # Option A: inline bytes via this API:
    manifest = {
        "format": fmt,
        "sha256": sha,
        "artifact_bytes_url": f"{_settings.external_api_base}/v1/model/bytes",  # set this in your settings
    }

    # Option B (optional): also put into remote storage and expose a URI:
    try:
        uri = put_bytes(blob, path_hint=f"validator/current_model.{fmt}.bin")
        manifest["artifact_uri"] = uri
    except Exception as e:
        logger.info("validator.manifest_no_remote_uri", error=str(e))

    return JSONResponse(manifest)


@app.get("/v1/model/bytes")
def model_bytes():
    """
    Streams the current model bytes.
    """
    blob, _ = export_model_artifact(MODEL)
    return StreamingResponse(iter([blob]), media_type="application/octet-stream")
