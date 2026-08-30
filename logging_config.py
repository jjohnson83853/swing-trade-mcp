"""Structured JSON logging.

Mirrors the pattern established in gh-cron-mcp/app/logging_config.py: stdlib
logging + a JSON Formatter with a message/extra_fields split, plus redaction
for secret-shaped field names and credentials embedded in URL values.
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone

_SECRET_KEY_MARKERS = ("token", "key", "secret", "password")
# Scrubs credentials embedded in URLs (e.g. an error string echoing back
# https://user:pass@host/...) -- key-name redaction alone can't catch a
# secret sitting inside a string value like this.
_CREDENTIAL_URL_RE = re.compile(r"://[^/@\s]*@")


def scrub_credentials(text: str) -> str:
    """Scrub credentials embedded in URLs, e.g. in raw error/response text."""
    return _CREDENTIAL_URL_RE.sub("://***@", text)


def _scrub(value):
    if isinstance(value, str):
        return scrub_credentials(value)
    if isinstance(value, dict):
        return _redact(value)
    return value


def _redact(fields: dict) -> dict:
    def _is_secret_key(k: str) -> bool:
        return any(m in k.lower() for m in _SECRET_KEY_MARKERS)

    # Only string values can hold a secret - a bool/int field whose name
    # happens to share a marker substring is never the secret itself.
    return {
        k: "***REDACTED***" if isinstance(v, str) and _is_secret_key(k) else _scrub(v)
        for k, v in fields.items()
    }


class JsonFormatter(logging.Formatter):
    """Formats each log record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_fields"):
            log_data.update(_redact(record.extra_fields))
        return json.dumps(log_data)


def setup_logging(name: str = "swing-trade-mcp", stream=None) -> logging.Logger:
    """Configure JSON logging.

    Defaults to stderr, matching this server's prior logging.basicConfig()
    default (stdout is not reserved for anything special here, but stderr is
    the safer default for a server process).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    return logger
