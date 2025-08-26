# Tests for logging_config module
# Testing framework: pytest (with built-in caplog and tmp_path fixtures where suitable)

import io
import json
import logging
import re
from pathlib import Path

import pytest

# Import the module under test
import importlib

logging_config = importlib.import_module("logging_config")

# ---------- Helper utilities ----------

class LoggerState:
    """Snapshot/restore root logger handlers and level to avoid cross-test leakage."""
    def __init__(self):
        self.root = logging.getLogger()
        self.handlers = list(self.root.handlers)
        self.level = self.root.level

    def restore(self):
        for h in list(self.root.handlers):
            self.root.removeHandler(h)
        for h in self.handlers:
            self.root.addHandler(h)
        self.root.setLevel(self.level)

def make_logger_with(handler):
    """Create an isolated logger with specified handler and propagate disabled."""
    logger = logging.getLogger(f"test_logger_{id(handler)}")
    logger.propagate = False
    # Clean existing handlers on this logger only
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

# ---------- Tests for SensitiveDataFilter ----------

@pytest.mark.parametrize(
    "msg,expected_pattern",
    [
        ("plain message", r"^plain message$"),
        ("secret is 1234567890abcdef1234567890ABCDEF", r"\[API_KEY\]"),
        ("contact me at user@example.com", r"\[EMAIL\]"),
        ("fetch https://api.example.com/v1?token=abc", r"\[URL\]"),
    ],
)
def test_sensitive_filter_sanitizes_strings(msg, expected_pattern):
    handler = logging.StreamHandler(io.StringIO())
    handler.addFilter(logging_config.SensitiveDataFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = make_logger_with(handler)
    logger.info(msg)
    output = handler.stream.getvalue().strip()
    assert re.search(expected_pattern, output), f"Output '{output}' should match {expected_pattern}"

def test_sensitive_filter_sanitizes_dict_args_and_nested_structures():
    handler = logging.StreamHandler(io.StringIO())
    handler.addFilter(logging_config.SensitiveDataFilter())
    handler.setFormatter(logging.Formatter("%(message)s %(api_key)s %(profile)s"))
    logger = make_logger_with(handler)
    extra = {
        "api_key": "topsecret",
        "profile": {
            "email": "u@example.com",
            "auth_token": "xyz",
            "nested": [{"password": "p@ss"}, "see https://example.com/path?secret=1"],
        },
    }
    # Message also contains sensitive-like token
    logger.info("user login 1234567890abcdef1234567890abcdef", extra=extra)
    output = handler.stream.getvalue()
    # api_key is redacted by key name
    assert "topsecret" not in output and "[REDACTED]" in output
    # email/url in nested values sanitized
    assert "[EMAIL]" in output
    assert "[URL]" in output
    # message API key masked
    assert "[API_KEY]" in output

def test_sensitive_filter_sanitizes_tuple_and_list_args():
    handler = logging.StreamHandler(io.StringIO())
    handler.addFilter(logging_config.SensitiveDataFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = make_logger_with(handler)
    # Use %-style formatting with args to exercise record.args sanitation
    logger.info("creds: %s %s", ("myemail@example.com", "https://x.y/z?k=1"))
    output = handler.stream.getvalue()
    assert "[EMAIL]" in output
    assert "[URL]" in output

# ---------- Tests for RequestCorrelationFilter and request id helpers ----------

def test_request_correlation_filter_adds_request_id_and_timestamp():
    handler = logging.StreamHandler(io.StringIO())
    handler.addFilter(logging_config.RequestCorrelationFilter())
    handler.setFormatter(logging.Formatter("%(request_id)s %(timestamp)s"))
    logger = make_logger_with(handler)
    logging_config.clear_request_id()
    logger.info("hello")
    out = handler.stream.getvalue().strip()
    req_id, ts = out.split(" ", 1)
    # UUID v4-like format
    assert re.match(r"^[0-9a-fA-F-]{36}$", req_id)
    # ISO-like timestamp
    assert "T" in ts or ":" in ts

def test_get_set_clear_request_id_roundtrip():
    logging_config.clear_request_id()
    generated = logging_config.get_request_id()
    assert re.match(r"^[0-9a-fA-F-]{36}$", generated)

    fixed = "00000000-0000-4000-8000-000000000000"
    assert logging_config.set_request_id(fixed) == fixed
    assert logging_config.get_request_id() == fixed

    logging_config.clear_request_id()
    assert logging_config.get_request_id() != fixed  # new uuid created

# ---------- Tests for StructuredFormatter ----------

def test_structured_formatter_outputs_json_with_expected_fields():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(logging_config.RequestCorrelationFilter())
    handler.addFilter(logging_config.SensitiveDataFilter())
    handler.setFormatter(logging_config.StructuredFormatter())
    logger = make_logger_with(handler)

    logging_config.set_request_id("11111111-1111-4111-111111111111")
    logger.info("Hello", extra={"feature": "X", "safe": 1})

    obj = json.loads(stream.getvalue())
    assert obj["message"] == "Hello"
    assert obj["level"] == "INFO"
    assert obj["logger"].startswith("test_logger_")
    assert obj["request_id"] == "11111111-1111-4111-111111111111"
    assert obj["feature"] == "X" and obj["safe"] == 1
    # Ensure standard keys exist
    for key in ("timestamp", "module", "function", "line"):
        assert key in obj

def test_structured_formatter_includes_exception_info():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(logging_config.RequestCorrelationFilter())
    handler.setFormatter(logging_config.StructuredFormatter())
    logger = make_logger_with(handler)

    try:
        raise ValueError("Boom")
    except Exception:
        logger.error("failed", exc_info=True)

    obj = json.loads(stream.getvalue())
    assert obj["level"] == "ERROR"
    assert "exception" in obj
    assert "ValueError" in obj["exception"]

# ---------- Tests for HumanReadableFormatter ----------

def test_human_readable_formatter_includes_core_fields_and_debug_location():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(logging_config.RequestCorrelationFilter())
    handler.setFormatter(logging_config.HumanReadableFormatter())
    logger = make_logger_with(handler)

    # Use DEBUG to include (module:lineno)
    logger.debug("debug here")
    text = stream.getvalue().strip()
    # [timestamp] [LEVEL    ] [logger] [request_id] message (module:line)
    assert text.count("[") >= 4 and "]" in text
    assert "DEBUG" in text
    assert "(" in text and ":" in text and text.endswith(")") is True

def test_human_readable_formatter_includes_exception_text_on_error():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(logging_config.RequestCorrelationFilter())
    handler.setFormatter(logging_config.HumanReadableFormatter())
    logger = make_logger_with(handler)

    try:
        raise RuntimeError("oops")
    except Exception:
        logger.error("got err", exc_info=True)
    text = stream.getvalue()
    assert "RuntimeError" in text
    assert "Traceback" in text

# ---------- Tests for setup_logging ----------

@pytest.mark.parametrize("bad_level", ["VERBOSE", "TRACE", ""])
def test_setup_logging_rejects_invalid_level(bad_level):
    with pytest.raises(ValueError):
        logging_config.setup_logging(level=bad_level)

@pytest.mark.parametrize("bad_fmt", ["xml", "plain", ""])
def test_setup_logging_rejects_invalid_format(bad_fmt):
    with pytest.raises(ValueError):
        logging_config.setup_logging(format_type=bad_fmt)

def test_setup_logging_configures_console_handler_and_sanitizes(capfd):
    # Snapshot and restore to avoid interfering with other tests
    state = LoggerState()
    try:
        logging_config.clear_request_id()
        logging_config.setup_logging(level="INFO", format_type="json")
        logger = logging_config.get_logger("x")
        logger.info("contact me at a@b.com with key 1234567890abcdef1234567890abcdef", extra={"password": "p"})
        out, err = capfd.readouterr()
        # json line on stdout
        assert out.strip(), "Expected output on stdout"
        obj = json.loads(out.strip())
        assert obj["message"].count("[EMAIL]") == 1
        assert "[API_KEY]" in obj["message"]
        # password redacted via extra
        assert obj.get("password") == "[REDACTED]"
    finally:
        state.restore()

def test_setup_logging_file_handler_writes_to_disk(tmp_path: Path):
    state = LoggerState()
    try:
        log_file = tmp_path / "logs" / "app.log"
        logging_config.setup_logging(level="WARNING", format_type="human", log_file=str(log_file), max_size=1024, backup_count=1)
        logger = logging_config.get_logger("filelogger")
        logger.warning("file test")
        # Ensure file was created and contains text
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "file test" in content
    finally:
        state.restore()

def test_setup_logging_clears_existing_handlers():
    state = LoggerState()
    try:
        # Attach a sentinel handler to root
        root = logging.getLogger()
        sentinel_stream = io.StringIO()
        sentinel = logging.StreamHandler(sentinel_stream)
        root.addHandler(sentinel)
        # Configure logging; should remove sentinel
        logging_config.setup_logging(level="ERROR", format_type="human")
        assert sentinel not in logging.getLogger().handlers
    finally:
        state.restore()

# ---------- Tests for get_logger ----------

def test_get_logger_returns_named_logger():
    logger = logging_config.get_logger("my.mod")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "my.mod"

# ---------- Tests for log_function_call decorator ----------

def test_log_function_call_success_path_emits_debug_and_returns_value():
    # Build isolated logger with json formatter to parse extras easily
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(logging_config.RequestCorrelationFilter())
    handler.setFormatter(logging_config.StructuredFormatter())

    logger_name = "decorator.mod"
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    @logging_config.log_function_call
    def add(a, b):
        return a + b

    # Perform call
    result = add(2, 3)
    assert result == 5

    # Parse two debug lines (entry and success)
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    # There might be two JSON logs; ensure both are DEBUG and messages contain expected text
    msgs = [json.loads(line)["message"] for line in lines]
    assert any("called" in m for m in msgs)
    assert any("completed successfully" in m for m in msgs)

def test_log_function_call_exception_path_logs_error_and_reraises():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(logging_config.RequestCorrelationFilter())
    handler.setFormatter(logging_config.StructuredFormatter())
    mod_logger = logging.getLogger("decorator.err")
    mod_logger.propagate = False
    for h in list(mod_logger.handlers):
        mod_logger.removeHandler(h)
    mod_logger.addHandler(handler)
    mod_logger.setLevel(logging.DEBUG)

    @logging_config.log_function_call
    def boom():
        raise KeyError("missing")

    with pytest.raises(KeyError):
        boom()

    obj = json.loads(stream.getvalue().splitlines()[-1])
    assert obj["level"] == "ERROR"
    assert "failed" in obj["message"]
    assert obj.get("error_type") == "KeyError"
    assert "exception" in obj

# ---------- Tests for log_error_with_context ----------

def test_log_error_with_context_emits_extra_fields_json():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(logging_config.RequestCorrelationFilter())
    handler.setFormatter(logging_config.StructuredFormatter())
    make_logger_with(handler)

    try:
        raise ValueError("bad")
    except Exception as e:
        logging_config.log_error_with_context(e, {"test_context": "v", "user_id": "u1"})

    obj = json.loads(stream.getvalue())
    assert obj["level"] == "ERROR"
    assert obj["error_type"] == "ValueError"
    assert obj["error_message"] == "bad"
    assert obj["test_context"] == "v"
    assert obj["user_id"] == "u1"
    assert re.match(r"^[0-9a-fA-F-]{36}$", obj["request_id"])