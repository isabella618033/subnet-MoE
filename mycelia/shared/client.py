#!/usr/bin/env python3
import argparse
import hashlib
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from requests import Response
from requests.exceptions import (
    ConnectionError as ReqConnectionError,
)
from requests.exceptions import (
    RequestException,
    Timeout,
)
from substrateinterface import Keypair

from mycelia.shared.app_logging import structlog
from mycelia.shared.schema import (
    SignedDownloadRequestMessage,
    SignedModelSubmitMessage,
    construct_block_message,
    construct_model_message,
    sign_message,
)

logger = structlog.get_logger(__name__)

CHUNK = 1024 * 1024  # 1 MiB


def human(n):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


_CHUNK = 1024 * 1024  # 1 MiB


def _sha256_file(path: str, chunk_size: int = _CHUNK) -> str:
    """Stream the file to compute SHA256 without loading into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def submit_model(
    url: str,
    token: str,
    model_path: str,
    my_hotkey: Keypair,
    target_hotkey_ss58: str,
    block: int,
    timeout_s: int = 300,
    retries: int = 3,
    backoff: float = 1.8,
    extra_form: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Upload a model checkpoint with retries and robust error handling.

    Returns parsed JSON on success. Raises RuntimeError with context on failure.
    """
    logger.info(
        "Starting model submission",
        url=url,
        model_path=model_path,
        target_hotkey=target_hotkey_ss58,
        block=block,
        retries=retries,
    )

    # --- preflight checks ---
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"File not found: {model_path}")

    file_size = os.path.getsize(model_path)
    if file_size == 0:
        raise ValueError(f"File is empty: {model_path}")

    logger.info("Model file validated", file_size=file_size, human_size=human(file_size))

    data = SignedModelSubmitMessage(
        target_hotkey_ss58=target_hotkey_ss58,
        origin_hotkey_ss58=my_hotkey.ss58_address,
        block=block,
        signature=sign_message(
            my_hotkey,
            construct_model_message(target_hotkey_ss58=target_hotkey_ss58, block=block, model_path=model_path),
        ),
    ).to_dict()

    if extra_form:
        # stringify non-bytes for safety in form data
        for k, v in extra_form.items():
            data[k] = v if isinstance(v, str | bytes) else str(v)

    logger.info("Request payload prepared", extra_fields=list(extra_form.keys()) if extra_form else None)

    # --- retry loop for transient failures ---
    attempt = 0
    last_exc: Exception | None = None

    while attempt <= retries:
        try:
            logger.info(f"Submission attempt {attempt + 1}/{retries + 1}")
            with open(model_path, "rb") as fh:
                files = {"file": (os.path.basename(model_path), fh)}
                logger.info("Uploading file to server", attempt=attempt + 1)
                resp: Response = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    files=files,
                    data=data,
                    timeout=timeout_s,
                )
            logger.info("HTTP response received", status_code=resp.status_code, attempt=attempt + 1)

            # Raise on non-2xx
            try:
                resp.raise_for_status()
            except requests.HTTPError as http_err:
                # Try to extract JSON error payload for context
                err_body = None
                try:
                    err_body = resp.json()
                except ValueError:
                    err_body = resp.text[:1000]  # truncate long HTML
                # Some 4xx are not retryable (auth, validation)
                non_retryable = {400, 401, 403, 404, 405, 409, 422}
                detail = f"HTTP {resp.status_code}: {err_body}"
                if resp.status_code in non_retryable or attempt == retries:
                    logger.error(
                        "Submission failed (non-retryable or final attempt)",
                        status_code=resp.status_code,
                        error_body=err_body,
                        attempt=attempt + 1,
                    )
                    raise RuntimeError(f"Upload failed: {detail}") from http_err
                else:
                    last_exc = RuntimeError(detail)
                    logger.warning(
                        "HTTP error, will retry",
                        status_code=resp.status_code,
                        error_body=err_body,
                        attempt=attempt + 1,
                    )
                    # fall through to retry
                    raise

            # Parse success JSON (server should return metadata)
            try:
                result = resp.json()
                logger.info("Submission successful", response=result)
                return result
            except ValueError:
                # Not JSON; return minimal info
                logger.info("Submission successful (non-JSON response)", status_code=resp.status_code)
                return {"status": "ok", "http_status": resp.status_code, "text": resp.text}

        except (Timeout, ReqConnectionError) as net_err:
            # Retry timeouts / connection errors unless we exhausted attempts
            last_exc = net_err
            logger.warning("Network error during submission, will retry", error=str(net_err), attempt=attempt + 1)
        except RequestException as req_err:
            # Generic requests error: retry unless final attempt
            last_exc = req_err
            logger.warning("Request error during submission, will retry", error=str(req_err), attempt=attempt + 1)
        except Exception as e:
            # File I/O during open/read already handled above; treat others as fatal
            logger.error("Unexpected error during upload", error=str(e))
            raise RuntimeError(f"Unexpected error during upload: {e}") from e

        # If we got here, we plan to retry
        attempt += 1
        if attempt <= retries:
            sleep_s = backoff**attempt
            logger.info("Retrying after backoff", sleep_seconds=sleep_s, attempt=attempt + 1)
            time.sleep(sleep_s)

    # Exhausted retries
    logger.error("Submission failed after all retries exhausted", total_attempts=retries + 1, last_error=str(last_exc))
    raise RuntimeError(f"Upload failed after {retries + 1} attempts: {last_exc}")


def download_model(
    url: str,
    my_hotkey: Keypair,
    target_hotkey_ss58: str,
    block: int,
    token: str,
    out: str | Path,
    expert_group_id: int | None = None,
    resume: bool = False,
    timeout: int = 30,
):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    mode = "wb"
    start_at = 0

    if resume and os.path.exists(out):
        start_at = os.path.getsize(out)
        if start_at > 0:
            headers["Range"] = f"bytes={start_at}-"
            mode = "ab"

    data = SignedDownloadRequestMessage(
        target_hotkey_ss58=target_hotkey_ss58,
        origin_hotkey_ss58=my_hotkey.ss58_address,
        expert_group_id=expert_group_id,
        block=block,
        signature=sign_message(my_hotkey, construct_block_message(target_hotkey_ss58, block=block)),
    ).to_dict()

    with requests.get(url, headers=headers, stream=True, timeout=timeout, data=data) as r:
        if r.status_code in (401, 403):
            sys.exit(f"Auth failed (HTTP {r.status_code}). Check your token.")
        if r.status_code == 416:
            print("Nothing to resume; file already complete.")
            return
        if resume and r.status_code not in (200, 206):
            print(f"Server did not honor range request (HTTP {r.status_code}). Restarting full download.")
            # retry full download
            headers.pop("Range", None)
            mode = "wb"
            start_at = 0
            r.close()
            r = requests.get(url, headers=headers, stream=True, timeout=timeout)

        r.raise_for_status()

        total = r.headers.get("Content-Length")
        total = int(total) + start_at if total is not None else None

        downloaded = start_at
        t0 = time()
        last_print = t0

        with open(out, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time()
                if now - last_print >= 0.5:
                    if total:
                        pct = downloaded / total * 100
                        bar = f"{human(downloaded)} / {human(total)} ({pct:5.1f}%)"
                    else:
                        bar = f"{human(downloaded)}"
                    rate = (downloaded - start_at) / max(1e-6, (now - t0))
                    print(f"\rDownloading: {bar} @ {human(rate)}/s", end="", flush=True)
                    last_print = now

        # --- If the downloaded file is a zip, unzip it ---
        try:
            if zipfile.is_zipfile(out):
                print(f"Detected zip archive at {out}, extracting...")
                with zipfile.ZipFile(out, "r") as zf:
                    zf.extractall(out.parent)
                print(f"Extracted files to {out.parent}")
                # optional: remove the zip after extraction
                # out.unlink()
        except Exception as e:
            # Don't fail the whole download if extraction goes wrong
            print(f"Warning: failed to extract zip archive {out}: {e}")

        # final line
        elapsed = max(1e-6, time() - t0)
        rate = (downloaded - start_at) / elapsed
        if total:
            bar = f"{human(downloaded)} / {human(total)} (100.0%)"
        else:
            bar = f"{human(downloaded)}"
        print(f"\rDone:       {bar} in {elapsed:.1f}s @ {human(rate)}/s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download a checkpoint with optional bearer auth.")
    p.add_argument("--url", default="http://localhost:8000/checkpoint", help="Download URL")
    p.add_argument("--token", default="supersecrettoken", help="Bearer auth token (omit if not required)")
    p.add_argument("-o", "--output", default="model.pt", help="Output file path")
    p.add_argument("--resume", action="store_true", help="Resume if partial file exists")
    p.add_argument("--timeout", type=int, default=30, help="Request timeout seconds")
    args = p.parse_args()

    try:
        download_model(args.url, args.token, args.output, resume=args.resume, timeout=args.timeout)
    except requests.RequestException as e:
        sys.exit(f"Network error: {e}")
    except KeyboardInterrupt:
        sys.exit("\nCanceled.")
