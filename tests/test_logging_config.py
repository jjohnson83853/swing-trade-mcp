"""Unit tests for the structured JSON logging setup (logging_config.py).

Covers the JsonFormatter's field shape, the extra_fields merge, and secret
redaction -- this is the formatter every call site in server.py now relies on
to keep dynamic data (ticker symbols, error text, cache stats) queryable
instead of buried in an f-string.
"""
import json
import logging

import pytest

from logging_config import JsonFormatter, setup_logging


def _make_record(level=logging.INFO, msg="hello", extra_fields=None):
    record = logging.LogRecord(
        name="swing-trade-mcp", level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    if extra_fields is not None:
        record.extra_fields = extra_fields
    return record


def test_json_formatter_includes_core_fields():
    record = _make_record(msg="tool_failed")
    data = json.loads(JsonFormatter().format(record))

    assert data["level"] == "INFO"
    assert data["logger"] == "swing-trade-mcp"
    assert data["message"] == "tool_failed"
    assert "time" in data


def test_json_formatter_merges_extra_fields():
    record = _make_record(extra_fields={"symbol": "AAPL", "reason": "exception"})
    data = json.loads(JsonFormatter().format(record))

    assert data["symbol"] == "AAPL"
    assert data["reason"] == "exception"


@pytest.mark.parametrize("key", ["api_key", "token", "secret", "password", "Auth_Token"])
def test_json_formatter_redacts_secret_shaped_keys(key):
    record = _make_record(extra_fields={key: "super-secret-value"})
    data = json.loads(JsonFormatter().format(record))

    assert data[key] == "***REDACTED***"


def test_json_formatter_scrubs_credentials_in_url_values():
    record = _make_record(extra_fields={"url": "https://user:pass@example.com/path"})
    data = json.loads(JsonFormatter().format(record))

    assert "pass" not in data["url"]
    assert data["url"] == "https://***@example.com/path"


def test_setup_logging_returns_named_logger_with_json_handler():
    logger = setup_logging("test-swing-logger")

    assert logger.name == "test-swing-logger"
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0].formatter, JsonFormatter)
