# llm_weightnet/shared/logging.py
from __future__ import annotations

import logging
import sys
import os

import structlog


def configure_logging() -> None:
    """
    Pretty console in dev, JSON in prod. Force with pretty=True/False.
    Env overrides:
      LOG_FORMAT=pretty|json
      LOG_LEVEL=DEBUG|INFO|...
      LOG_UTC=1  (timestamp in UTC)
    """
    level = "INFO"
    fmt   = "pretty"
    use_utc = True

    # 1) stdlib baseline so third-party libs (uvicorn, requests) show up
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")

    # 2) structlog processors (common)
    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=use_utc)
    common = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        processors = common + [structlog.processors.JSONRenderer()]
    else:
        # Pretty, aligned columns. pad_event aligns the event text; key=value follow neatly.
        processors = common + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                pad_event=28,      # adjust to your taste
                exception_formatter=structlog.dev.rich_traceback,  # nicer tracebacks if 'rich' installed
            )
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
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

logger = structlog.get_logger()