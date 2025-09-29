# llm_weightnet/shared/logging.py
from __future__ import annotations

import logging
import sys
import os

import structlog


def configure_logging() -> None:
    """
    Configure global logging for miners and validators.

    - In DEV (LOG_FORMAT=console): pretty colored logs
    - In PROD (LOG_FORMAT=json): machine-parseable JSON logs
    """
    log_format = os.getenv("LOG_FORMAT", "console").lower()

    shared_processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        # JSON logs: good for log aggregation systems
        processors = shared_processors + [
            structlog.processors.JSONRenderer()
        ]
    else:
        # Human-readable console logs
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging (so libraries log consistently)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


# Module-level logger you can import directly
structlog.configure_once(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
log = structlog.get_logger()
