#!/usr/bin/env python3
# download_checkpoint.py
import argparse
import os
import sys
import requests
import hashlib
from time import time
import hashlib
import os
from typing import Any, Dict, Optional
import requests
from requests import Response
from requests.exceptions import RequestException, Timeout, ConnectionError as ReqConnectionError


CHUNK = 1024 * 1024  # 1 MiB

def human(n):
    for unit in ["B","KiB","MiB","GiB","TiB"]:
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
    uid: int, 
    hotkey: str,
    step: int = 12000,
    timeout_s: int = 300,
    retries: int = 3,
    backoff: float = 1.8,
    extra_form: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Upload a model checkpoint with retries and robust error handling.

    Returns parsed JSON on success. Raises RuntimeError with context on failure.
    """
    # --- preflight checks ---
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"File not found: {model_path}")

    if os.path.getsize(model_path) == 0:
        raise ValueError(f"File is empty: {model_path}")

    try:
        checksum = _sha256_file(model_path)
    except PermissionError as e:
        raise PermissionError(f"Cannot read file (permission denied): {model_path}") from e
    except OSError as e:
        raise OSError(f"Failed to read file: {model_path}") from e

    data = {"step": str(step), "checksum_sha256": checksum, "uid": uid, "hotkey": hotkey}
    if extra_form:
        # stringify non-bytes for safety in form data
        for k, v in extra_form.items():
            data[k] = v if isinstance(v, (str, bytes)) else str(v)

    # --- retry loop for transient failures ---
    attempt = 0
    last_exc: Optional[Exception] = None

    while attempt <= retries:
        try:
            with open(model_path, "rb") as fh:
                files = {"file": (os.path.basename(model_path), fh)}
                resp: Response = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    files=files,
                    data=data,
                    timeout=timeout_s,
                )

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
                    raise RuntimeError(f"Upload failed: {detail}") from http_err
                else:
                    last_exc = RuntimeError(detail)
                    # fall through to retry
                    raise

            # Parse success JSON (server should return metadata)
            try:
                return resp.json()
            except ValueError:
                # Not JSON; return minimal info
                return {"status": "ok", "http_status": resp.status_code, "text": resp.text}

        except (Timeout, ReqConnectionError) as net_err:
            # Retry timeouts / connection errors unless we exhausted attempts
            last_exc = net_err
        except RequestException as req_err:
            # Generic requests error: retry unless final attempt
            last_exc = req_err
        except Exception as e:
            # File I/O during open/read already handled above; treat others as fatal
            raise RuntimeError(f"Unexpected error during upload: {e}") from e

        # If we got here, we plan to retry
        attempt += 1
        if attempt <= retries:
            sleep_s = backoff ** attempt
            time.sleep(sleep_s)

    # Exhausted retries
    raise RuntimeError(f"Upload failed after {retries + 1} attempts: {last_exc}")

def download_model(url: str, token: str, out: str, resume: bool = False, timeout: int = 30):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    mode = "wb"
    start_at = 0

    if resume and os.path.exists(out):
        start_at = os.path.getsize(out)
        if start_at > 0:
            headers["Range"] = f"bytes={start_at}-"
            mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
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