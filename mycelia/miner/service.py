from .scheduler import MinerScheduler
from mycelia.shared.metrics import start_metrics_server

# llm_weightnet/miner/cli.py
from __future__ import annotations

import typer
from typing import Optional

from mycelia.shared.app_logging import configure_logging, log
from mycelia.settings import Settings
from service import run as run_service

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command(help="Run the miner service (blocks; exports metrics).")
def run(
    config: Optional[str] = typer.Option(None, "--config", help="Path to YAML/ENV config if applicable.")
):
    configure_logging()
    if config:
        logger.info("miner.config_path", path=config)
        # If you wired YAML → Settings, load/merge here. For now, Settings() reads env.
    cfg = Settings()
    logger.info("miner.start", broker_url=cfg.broker_url, device=cfg.device, model=cfg.model_name)
    try:
        run_service()
    except KeyboardInterrupt:
        logger.info("miner.stopped")

@app.command(help="Quick health check (exits 0 if import/startup is OK).")
def health():
    configure_logging()
    logger.info("miner.health", status="ok")

def run_app():
    app()


