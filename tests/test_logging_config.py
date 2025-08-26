import json
import logging
import re
import sys
import contextlib
from pathlib import Path

import pytest

# Attempt to import the module as "logging_config". If it's not importable by name,
# skip the suite with a helpful message so CI shows a clear reason.
try:
    import logging_config as lc
except Exception:
    lc = None

pytestmark = pytest.mark.skipif(lc is None, reason="logging_config module not importable: ensure logging_config.py is on PYTHONPATH or project root")

# ------------------------------
# Helpers
# ------------------------------

class RecordCatcher(logging.Handler):
    """Simple handler that stores emitted LogRecords for inspection."""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def reset_root_logger():
    """Utility to reset root logger handlers/state between tests."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    # Reset level to WARNING default to avoid cross-test bleed
    root.setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def _reset_between_tests():
    # Clear correlation id and reset root logger between tests
    if lc:
        with contextlib.suppress(Exception):
            lc.clear_request_id()
    reset_root_logger()
    yield
    if lc:
        with contextlib.suppress(Exception):
            lc.clear_request_id()
    reset_root_logger()


# ------------------------------
# Request ID utilities
# ------------------------------

@pytest.mark.parametrize("explicit", [None, "fixed-id-123"])
def test_request_id_set_get_clear_stability(explicit):
    rid_before = lc.get_request_id()
    assert isinstance(rid_before, str) and len(rid_before) > 0

    # set explicit or random
    new_id = lc.set_request_id(explicit)
    assert isinstance(new_id, str)
    assert lc.get_request_id() == new_id

    # Ensure stable until cleared
    again = lc.get_request_id()
    assert again == new_id

    # Clear then new value differs (for explicit None may differ; for explicit fixed remains until cleared)
    lc.clear_request_id()
    rid_after_clear = lc.get_request_id()
    assert isinstance(rid_after_clear, str)
    assert rid_after_clear != new_id


def test_set_request_id_generates_uuid_when_none():
    lc.clear_request_id()
    gen = lc.set_request_id(None)
    # UUIDv4 format sanity (not strict validation)
    assert re.fullmatch(r"[0-9a-fA-F-]{36}", gen)


# ------------------------------
# SensitiveDataFilter
# ------------------------------

def test_sensitive_filter_sanitizes_message_components(caplog):
    filt = lc.SensitiveDataFilter()

    logger = logging.getLogger("test.sensitive.message")
    logger.setLevel(logging.DEBUG)
    h = logging.StreamHandler(stream=sys.stdout)
    h.addFilter(filt)
    # Use human formatter to keep string message visible
    h.setFormatter(lc.HumanReadableFormatter())
    logger.addHandler(h)

    # Compose message with email, url, api key like token (>=32 alnum)
    api_key_like = "A" * 32
    msg = f"Contact john.doe@example.com visit https://secret.example.com and key {api_key_like}"
    with caplog.at_level(logging.INFO):
        logger.info(msg)

    # Check captured human-readable output via caplog
    combined = " ".join(r.message for r in caplog.records)
    assert "[EMAIL]" in combined
    assert "[URL]" in combined
    assert "[API_KEY]" in combined
    assert "john.doe@example.com" not in combined
    assert "https://secret.example.com" not in combined
    assert api_key_like not in combined


def test_sensitive_filter_sanitizes_args_dict_and_collections(caplog):
    filt = lc.SensitiveDataFilter()
    logger = logging.getLogger("test.sensitive.args")
    logger.setLevel(logging.DEBUG)
    h = logging.StreamHandler(stream=sys.stdout)
    h.addFilter(filt)
    h.setFormatter(lc.HumanReadableFormatter())
    logger.addHandler(h)

    args_dict = {
        "password": "supersecret",
        "nested": {"token": "t0k3n", "note": "ok"},
        "list": ["val", "abc@example.com", "https://x.y"],
    }
    with caplog.at_level(logging.INFO):
        logger.info("Testing args %s", args_dict)

    # The filter sanitizes record.args, so formatted string should have redacted secrets and sanitized email/url
    text = " ".join(r.message for r in caplog.records)
    assert "password" in text and "[REDACTED]" in text
    assert "abc@example.com" not in text and "[EMAIL]" in text
    assert "https://x.y" not in text and "[URL]" in text
    # Nested token should be redacted
    assert "'token': '[REDACTED]'" in text


@pytest.mark.xfail(reason="Current SensitiveDataFilter does not sanitize values passed via 'extra' attributes")
def test_sensitive_filter_should_sanitize_extra_fields_but_currently_does_not(caplog):
    # This captures a likely gap: sensitive data provided via 'extra' is not sanitized by the filter.
    lc.setup_logging(level="INFO", format_type="json")
    logger = lc.get_logger("test.sensitive.extra")
    with caplog.at_level(logging.INFO):
        logger.info("Testing", extra={"api_key": "supersecret", "password": "pw123"})
    # Parse the JSON log line
    lines = [rec.message for rec in caplog.records if rec.name == "test.sensitive.extra"]
    assert lines, "Expected at least one record"
    payload = json.loads(lines[-1])
    # Expected redaction, but current implementation will leak. Marked xfail to document/guard.
    assert payload.get("api_key") == "[REDACTED]"
    assert payload.get("password") == "[REDACTED]"


# ------------------------------
# RequestCorrelationFilter
# ------------------------------

def test_request_correlation_filter_adds_request_id_and_timestamp():
    filt = lc.RequestCorrelationFilter()
    catcher = RecordCatcher()
    logger = logging.getLogger("test.correlation")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(catcher)
    logger.addFilter(filt)

    logger.info("hello")

    assert catcher.records, "No records captured"
    rec = catcher.records[-1]
    assert hasattr(rec, "request_id")
    assert isinstance(rec.request_id, str) and len(rec.request_id) > 0
    assert hasattr(rec, "timestamp")
    # Basic ISO format sanity
    assert "T" in rec.timestamp or ":" in rec.timestamp


# ------------------------------
# StructuredFormatter
# ------------------------------

def test_structured_formatter_outputs_valid_json_with_expected_fields():
    lc.clear_request_id()
    lc.set_request_id("req-123")
    formatter = lc.StructuredFormatter()
    catcher = RecordCatcher()
    catcher.setFormatter(formatter)

    logger = logging.getLogger("test.structured")
    logger.setLevel(logging.INFO)
    logger.addHandler(catcher)
    logger.addFilter(lc.RequestCorrelationFilter())

    logger.info("Hello %s", "world", extra={"custom": 42})

    assert catcher.records, "Expected a record to be captured"
    formatted = catcher.format(catcher.records[-1])
    data = json.loads(formatted)

    # core fields present
    for key in (
        "timestamp", "level", "logger", "message", "request_id", "module", "function", "line"
    ):
        assert key in data

    assert data["level"] == "INFO"
    assert data["logger"] == "test.structured"
    assert data["message"] == "Hello world"
    assert data["request_id"] == "req-123"
    # extra fields are included
    assert data.get("custom") == 42


def test_structured_formatter_includes_exception_info():
    formatter = lc.StructuredFormatter()
    catcher = RecordCatcher()
    catcher.setFormatter(formatter)
    logger = logging.getLogger("test.structured.exc")
    logger.setLevel(logging.ERROR)
    logger.addHandler(catcher)
    logger.addFilter(lc.RequestCorrelationFilter())

    try:
        raise ValueError("boom")
    except Exception:
        logger.exception("Failure while processing")

    formatted = catcher.format(catcher.records[-1])
    data = json.loads(formatted)
    assert "exception" in data
    assert "ValueError" in data["exception"]


# ------------------------------
# HumanReadableFormatter
# ------------------------------

def test_human_readable_formatter_includes_key_parts(caplog):
    lc.set_request_id("human-req")
    formatter = lc.HumanReadableFormatter()
    logger = logging.getLogger("test.human")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addFilter(lc.RequestCorrelationFilter())

    with caplog.at_level(logging.INFO):
        logger.info("Hi there")

    line = caplog.text
    assert "[test.human]" in line
    assert "[human-req]" in line
    # For INFO, module:lineno should not necessarily be present, but should not crash
    assert "Hi there" in line


# ------------------------------
# setup_logging
# ------------------------------

@pytest.mark.parametrize("fmt", ["json", "human"])
def test_setup_logging_configures_console_handler_with_filters_and_formatter(fmt, caplog):
    lc.setup_logging(level="DEBUG", format_type=fmt)
    root = logging.getLogger()
    # exactly one console handler should exist (file handler optional unless specified)
    handlers = root.handlers
    assert handlers, "Expected handlers on root"
    # ensure at least one handler has our filters and formatter type
    has_expected = False
    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            # Check formatter class
            if fmt == "json":
                assert isinstance(h.formatter, lc.StructuredFormatter)
            else:
                assert isinstance(h.formatter, lc.HumanReadableFormatter)
            # Filters present
            filter_types = {type(f) for f in h.filters}
            assert lc.SensitiveDataFilter in filter_types
            assert lc.RequestCorrelationFilter in filter_types
            has_expected = True
    assert has_expected, "No console handler matched expected configuration"


def test_setup_logging_invalid_inputs_raise():
    with pytest.raises(ValueError):
        lc.setup_logging(level="BAD", format_type="json")
    with pytest.raises(ValueError):
        lc.setup_logging(level="INFO", format_type="xml")


def test_setup_logging_with_file_handler(tmp_path):
    log_file = tmp_path / "logs" / "app.log"
    lc.setup_logging(level="INFO", format_type="json", log_file=str(log_file), max_size=1024, backup_count=2)

    root = logging.getLogger()
    from logging.handlers import RotatingFileHandler
    file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert file_handlers, "Expected a RotatingFileHandler when log_file is provided"
    fh = file_handlers[0]
    # Handler points to the correct file
    assert Path(fh.baseFilename) == log_file
    assert fh.backupCount == 2
    # Logging writes to file
    logger = lc.get_logger("file.writer")
    logger.info("to file", extra={"k": "v"})
    # Ensure file is created and has content
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert content.strip() != ""


# ------------------------------
# get_logger
# ------------------------------

def test_get_logger_returns_named_logger():
    name = "custom.logger.name"
    logger = lc.get_logger(name)
    assert isinstance(logger, logging.Logger)
    assert logger.name == name


# ------------------------------
# log_function_call decorator
# ------------------------------

def _succeeding(x, y):
    return x + y


def _failing(x):
    raise RuntimeError(f"bad {x}")


def test_log_function_call_success_and_error(caplog, monkeypatch):
    # Wrap functions with decorator
    wrapped_ok = lc.log_function_call(_succeeding)
    wrapped_bad = lc.log_function_call(_failing)

    lc.setup_logging(level="DEBUG", format_type="human")

    with caplog.at_level(logging.DEBUG):
        res = wrapped_ok(2, 3)
    assert res == 5
    texts = [r.getMessage() for r in caplog.records]
    # should include entry and success messages
    assert any("Function _succeeding called" in t for t in texts)
    assert any("Function _succeeding completed successfully" in t for t in texts)

    # Error path captures message and includes exc_info
    caplog.clear()
    with pytest.raises(RuntimeError), caplog.at_level(logging.ERROR):
        wrapped_bad("x")
    texts = [r.getMessage() for r in caplog.records]
    assert any("Function _failing failed:" in t for t in texts)
    assert any("RuntimeError" in (r.exc_text or "") or r.exc_info is not None for r in caplog.records)


# ------------------------------
# log_error_with_context
# ------------------------------

def test_log_error_with_context_includes_context_and_exc_info(caplog):
    lc.setup_logging(level="ERROR", format_type="json")
    try:
        raise KeyError("missing")
    except Exception as e:
        with caplog.at_level(logging.ERROR):
            lc.log_error_with_context(e, {"context_k": "context_v", "user_id": "123"})

    # Parse the last JSON line and assert fields
    recs = [r for r in caplog.records if r.levelno == logging.ERROR and r.name.endswith("logging_config")]
    assert recs, "Expected an ERROR record from logging_config"
    payload = json.loads(recs[-1].message)
    assert payload.get("error_type") == "KeyError"
    assert "missing" in payload.get("error_message", "")
    assert payload.get("context_k") == "context_v"
    assert "exception" in payload


# ------------------------------
# Integration style smoke: ensure both formats produce output that includes request_id
# ------------------------------

@pytest.mark.parametrize("fmt", ["json", "human"])
def test_smoke_logging_includes_request_id(fmt, caplog):
    lc.set_request_id("rid-smoke")
    lc.setup_logging(level="INFO", format_type=fmt)
    logger = lc.get_logger("smoke")
    with caplog.at_level(logging.INFO):
        logger.info("hello")

    assert caplog.records, "Expected at least one record"
    if fmt == "json":
        data = json.loads(caplog.records[-1].message)
        assert data.get("request_id") == "rid-smoke"
    else:
        assert "[rid-smoke]" in caplog.text