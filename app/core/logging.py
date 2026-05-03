import logging
import os

from pythonjsonlogger import json as jsonlogger


def configure_logging() -> None:
    """Configure the root logger to emit JSON lines to stdout.

    Reads the desired log level from the LOG_LEVEL environment variable
    (case-insensitive). Falls back to INFO if the variable is unset or
    contains an unrecognised value.

    Clears existing handlers before adding a new one so that repeated
    calls (e.g. in tests) do not produce duplicate log entries.
    """
    level_name = os.getenv("LOG_LEVEL", "info").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        # Rename fields to match common JSON logging conventions.
        rename_fields={"levelname": "level", "asctime": "timestamp"},
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name.

    Thin wrapper around :func:`logging.getLogger` so callers don't need
    to import the standard ``logging`` module directly.
    """
    return logging.getLogger(name)
