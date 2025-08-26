# NOTE: Tests assume pytest is available as the project test runner.
# -*- coding: utf-8 -*-
"""
Tests for validation module.

Framework: pytest (function-based tests with fixtures such as tmp_path and caplog).
These tests focus on public interfaces in validation.py and emphasize behavior added/changed in the PR diff
(coverage includes input sanitization, platform normalization, file upload checks, rate limiting, shared store validation,
LLM prompt sanitization, and the composite validate_and_sanitize_inputs flow).

Conventions:
- Descriptive test names
- Happy paths, edge cases, and failure scenarios
- Avoid new dependencies; use stdlib and pytest built-ins
"""

import os
from datetime import datetime, timedelta

import pytest

# Import module under test. Support both "validation.py" at repo root and packages.
# If project structure differs, adjust PYTHONPATH externally or via conftest.
import importlib

try:
    validation = importlib.import_module("validation")
except ModuleNotFoundError:
    # Try common alternative: src/validation.py layout
    import sys
    for candidate in ["src", "app", "backend", "server"]:
        if os.path.isdir(candidate):
            sys.path.insert(0, candidate)
            try:
                validation = importlib.import_module("validation")
                break
            except ModuleNotFoundError:
                sys.path.pop(0)
    else:
        # Last-resort import error that provides clear guidance
        raise

ValidationError = validation.ValidationError


# -------------------------
# sanitize_string
# -------------------------

def test_sanitize_string_happy_path_normalizes_whitespace_and_trims():
    text = "   Hello   world   "
    out = validation.sanitize_string(text, max_length=100)
    assert out == "Hello world"


def test_sanitize_string_removes_control_chars_and_empties_raise():
    # Contains null byte, bell, and whitespace-only after stripping -> should error for empty
    s = "\x00\x07 \t \n "
    with pytest.raises(ValidationError) as exc:
        validation.sanitize_string(s, max_length=10)
    err = exc.value
    assert err.field == "input"
    assert err.code in ("EMPTY_ERROR", "LENGTH_ERROR")


def test_sanitize_string_type_error_for_non_string():
    with pytest.raises(ValidationError) as exc:
        validation.sanitize_string(123)  # type: ignore[arg-type]
    assert exc.value.code == "TYPE_ERROR"
    assert exc.value.field == "input"


def test_sanitize_string_length_error_when_exceeds_max():
    long = "a" * 51
    with pytest.raises(ValidationError) as exc:
        validation.sanitize_string(long, max_length=50)
    assert exc.value.code == "LENGTH_ERROR"


# -------------------------
# validate_topic
# -------------------------

def test_validate_topic_happy_path_and_min_length():
    assert validation.validate_topic("Product launch update") == "Product launch update"


@pytest.mark.parametrize("bad", [
    "<script>alert(1)</script>",
    "Click here: javascript:alert(1)",
    "data:text/html;base64,AAAA",
    "vbscript:foo",
    "<iframe src='x'></iframe>",
    "<object></object>",
    "<embed/>",
    "onload=evil() New topic",
])
def test_validate_topic_rejects_malicious_content(bad):
    with pytest.raises(ValidationError) as exc:
        validation.validate_topic(bad)
    assert exc.value.code == "SECURITY_ERROR"
    assert exc.value.field == "topic"


def test_validate_topic_too_short_raises_length_error():
    with pytest.raises(ValidationError) as exc:
        validation.validate_topic("Hi", min_length=3)
    assert exc.value.code == "LENGTH_ERROR"


def test_validate_topic_excessive_repetition_rejected():
    topic = "Sale Sale Sale Sale big"
    with pytest.raises(ValidationError) as exc:
        validation.validate_topic(topic)
    assert exc.value.code == "CONTENT_ERROR"


# -------------------------
# normalize_platform_name
# -------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("X", "twitter"),
        ("fb", "facebook"),
        (" IG ", "instagram"),
        ("yt", "youtube"),
        ("TT", "tiktok"),
        ("Li", "linkedin"),
        ("twitter", "twitter"),
        ("Unknown", "unknown"),
    ],
)
def test_normalize_platform_name_variants(raw, expected):
    assert validation.normalize_platform_name(raw) == expected


def test_normalize_platform_name_type_error():
    with pytest.raises(ValidationError) as exc:
        validation.normalize_platform_name(123)  # type: ignore[arg-type]
    assert exc.value.code == "TYPE_ERROR"
    assert exc.value.field == "platform"


# -------------------------
# validate_platforms
# -------------------------

def test_validate_platforms_happy_path_string_and_aliases():
    out = validation.validate_platforms(" twitter , li , fb ")
    assert out == ["twitter", "linkedin", "facebook"]


def test_validate_platforms_happy_path_list_input():
    out = validation.validate_platforms(["ig", "yt"])
    assert out == ["instagram", "youtube"]


def test_validate_platforms_type_error_for_non_string_input():
    with pytest.raises(ValidationError) as exc:
        validation.validate_platforms(42)  # type: ignore[arg-type]
    assert exc.value.code == "TYPE_ERROR"
    assert exc.value.field == "platforms"


def test_validate_platforms_empty_error():
    with pytest.raises(ValidationError) as exc:
        validation.validate_platforms(" , , ")
    assert exc.value.code == "EMPTY_ERROR"


def test_validate_platforms_unsupported_platform():
    with pytest.raises(ValidationError) as exc:
        validation.validate_platforms("myspace")
    assert exc.value.code == "UNSUPPORTED_PLATFORM"


def test_validate_platforms_duplicate_platform():
    with pytest.raises(ValidationError) as exc:
        validation.validate_platforms("twitter, X")
    # "X" normalizes to twitter, producing a duplicate
    assert exc.value.code == "DUPLICATE_PLATFORM"


def test_validate_platforms_too_many():
    # Build 11 valid platforms by repeating with suffixes to hit count limit
    # Using same platform would trigger duplicate, so use supported unique ones and then pad with repeats + alias to force count
    supported = list(validation.SUPPORTED_PLATFORMS.keys())
    eleven = supported + ["twitter"]  # 7 + 1; we need >10, so extend more safely
    # Ensure length > 10
    while len(eleven) <= 10:
        eleven.append("linkedin")
    with pytest.raises(ValidationError) as exc:
        validation.validate_platforms(",".join(eleven), max_platforms=10)
    assert exc.value.code == "COUNT_ERROR"


# -------------------------
# validate_brand_bible_content
# -------------------------

def test_validate_brand_bible_content_happy_path_and_size():
    content = "  <brand>Acme</brand>  "
    # sanitize_string will normalize whitespace but malicious tags list includes <form>, <input>, etc., not generic tags
    out = validation.validate_brand_bible_content(content, max_size=1024)
    # Returned content should be sanitized (trimmed, whitespace normalized), but XML-like tags not stripped by this function
    assert out == "<brand>Acme</brand>"


@pytest.mark.parametrize("bad", [
    "<script>alert(1)</script>",
    "javascript:doEvil()",
    "<form action='/steal'>",
    "<input type='text'>",
    "<iframe src='x'>",
    "<object data='x'></object>",
    "<embed></embed>",
    "onload=evil()",
])
def test_validate_brand_bible_content_rejects_malicious(bad):
    with pytest.raises(ValidationError) as exc:
        validation.validate_brand_bible_content(bad)
    assert exc.value.code == "SECURITY_ERROR"
    assert exc.value.field == "brand_bible"


def test_validate_brand_bible_content_type_and_size_errors():
    with pytest.raises(ValidationError) as exc:
        validation.validate_brand_bible_content(123)  # type: ignore[arg-type]
    assert exc.value.code == "TYPE_ERROR"

    big = "a" * 11
    with pytest.raises(ValidationError) as exc:
        validation.validate_brand_bible_content(big, max_size=10)  # bytes == chars here
    assert exc.value.code == "SIZE_ERROR"


# -------------------------
# validate_file_upload
# -------------------------

def _touch(fp: str, data: bytes = b"x"):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "wb") as f:
        f.write(data)


def test_validate_file_upload_happy_path(tmp_path):
    file_path = tmp_path / "ok.txt"
    file_path.write_text("hello", encoding="utf-8")
    meta = validation.validate_file_upload(str(file_path))
    assert meta["path"] == str(file_path)
    assert meta["name"] == file_path.name
    assert meta["size"] == 5
    assert meta["extension"] == ".txt"
    assert isinstance(meta["modified"], datetime)


def test_validate_file_upload_not_found(tmp_path):
    missing = tmp_path / "missing.txt"
    with pytest.raises(ValidationError) as exc:
        validation.validate_file_upload(str(missing))
    assert exc.value.code == "NOT_FOUND"
    assert exc.value.field == "file"


def test_validate_file_upload_size_and_type_errors(tmp_path):
    big = tmp_path / "big.txt"
    big.write_bytes(b"abcd")
    with pytest.raises(ValidationError) as exc:
        validation.validate_file_upload(str(big), max_size=3)
    assert exc.value.code == "SIZE_ERROR"

    bad = tmp_path / "bad.exe"
    bad.write_bytes(b"ok")
    with pytest.raises(ValidationError) as exc:
        validation.validate_file_upload(str(bad))
    assert exc.value.code == "TYPE_ERROR"


def test_validate_file_upload_suspicious_name_detects_directory_traversal(tmp_path):
    # Create real file
    real = tmp_path / "safe.txt"
    real.write_text("x", encoding="utf-8")
    # Reference it using a path string with "../" to trigger the security pattern
    suspicious = os.path.join(str(tmp_path), "folder", "..", "safe.txt")
    # Ensure OS resolves path to the real file (exists check passes)
    assert os.path.exists(suspicious)
    with pytest.raises(ValidationError) as exc:
        validation.validate_file_upload(suspicious)
    assert exc.value.code == "SECURITY_ERROR"


# -------------------------
# check_rate_limit
# -------------------------

def test_check_rate_limit_allows_until_limit_then_blocks(monkeypatch):
    ident = "user-123"
    # Reset store for this identifier
    store = validation._rate_limit_store
    store[ident] = []

    max_req = 5
    window = 60

    # Allow exactly max_req within window
    for _i in range(max_req):
        assert validation.check_rate_limit(ident, max_requests=max_req, window_seconds=window) is True

    # Next one should be blocked
    assert validation.check_rate_limit(ident, max_requests=max_req, window_seconds=window) is False

    # Move the first timestamp out of window and try again -> should allow
    old = datetime.now() - timedelta(seconds=window + 1)
    store[ident][0] = old
    assert validation.check_rate_limit(ident, max_requests=max_req, window_seconds=window) is True


# -------------------------
# validate_shared_store
# -------------------------

def test_validate_shared_store_happy_path():
    shared = {
        "task_requirements": {
            "topic_or_goal": "Announce product launch",
            "platforms": "twitter, linkedin",
        },
        "brand_bible": {
            "xml_raw": "<brand>ACME</brand>",
        },
        "stream": None,
    }
    out = validation.validate_shared_store(shared)
    assert set(out.keys()) == {"task_requirements", "brand_bible", "stream", "validation_timestamp"}
    assert out["task_requirements"]["platforms"] == ["twitter", "linkedin"]
    # timestamp must be ISO format
    datetime.fromisoformat(out["validation_timestamp"])


@pytest.mark.parametrize("mutate, field, code", [
    (lambda s: s.pop("task_requirements", None), "task_requirements", "MISSING"),
    (lambda s: s.update(task_requirements="oops"), "task_requirements", "TYPE_ERROR"),
    (lambda s: s["task_requirements"].pop("topic_or_goal", None), "topic_or_goal", "MISSING"),
    (lambda s: s.update(brand_bible="oops"), "brand_bible", "TYPE_ERROR"),
])
def test_validate_shared_store_detects_structure_errors(mutate, field, code):
    shared = {
        "task_requirements": {
            "topic_or_goal": "Legit topic",
            "platforms": "twitter, linkedin",
        },
        "brand_bible": {},
        "stream": None,
    }
    mutate(shared)
    with pytest.raises(ValidationError) as exc:
        validation.validate_shared_store(shared)
    assert exc.value.field == field
    assert exc.value.code == code


def test_validate_shared_store_platforms_can_be_list_or_string():
    shared = {
        "task_requirements": {
            "topic_or_goal": "Goal",
            "platforms": ["ig", "yt"],
        },
        "brand_bible": {},
    }
    out = validation.validate_shared_store(shared)
    assert out["task_requirements"]["platforms"] == ["instagram", "youtube"]


# -------------------------
# sanitize_for_llm_prompt
# -------------------------

def test_sanitize_for_llm_prompt_happy_path_and_escapes():
    raw = 'Hello <b>World</b>\nHe said: "Yes" and used backslash \\ and tab\tand CR\r'
    out = validation.sanitize_for_llm_prompt(raw)
    # Tags stripped
    assert "<b>" not in out and "</b>" not in out
    # Escapes present
    assert '\\"' in out
    assert "\\\\" in out
    assert "\\n" in out and "\\t" in out and "\\r" in out
    # Quotes escaped
    assert "\\'" not in out  # There was no single quote, ensure not added spuriously


def test_sanitize_for_llm_prompt_non_string_returns_empty():
    assert validation.sanitize_for_llm_prompt(None) == ""  # type: ignore[arg-type]


def test_sanitize_for_llm_prompt_truncates_long_input():
    long = "x" * 10010
    out = validation.sanitize_for_llm_prompt(long)
    assert out.endswith("...")
    assert len(out) == 10003  # 10000 + len("...") == 10003


# -------------------------
# validate_and_sanitize_inputs (integration of pieces)
# -------------------------

def test_validate_and_sanitize_inputs_success_and_logging(caplog):
    caplog.clear()
    with caplog.at_level("INFO"):
        result = validation.validate_and_sanitize_inputs(
            "Announce product launch",
            "twitter, linkedin",
            "<brand>Test</brand>",
        )
    # Assert structure and types
    assert result["task_requirements"]["topic_or_goal"] == "Announce product launch"
    assert result["task_requirements"]["platforms"] == ["twitter", "linkedin"]
    assert result["brand_bible"]["xml_raw"] == "<brand>Test</brand>"
    assert result["stream"] is None
    assert "validation_timestamp" in result
    # Logging contains context fields
    found = False
    for rec in caplog.records:
        if rec.levelname == "INFO" and "Inputs validated successfully" in rec.getMessage():
            found = True
            # metadata present in extra - may not be in message; just ensure no errors thrown
            break
    assert found


def test_validate_and_sanitize_inputs_failure_paths_log_warning(caplog):
    caplog.clear()
    with pytest.raises(ValidationError), caplog.at_level("WARNING"):
        validation.validate_and_sanitize_inputs(
            "Ok topic",
            "invalid_platform",
            "",
        )
    # Warning logged
    assert any(r.levelname == "WARNING" and "Input validation failed" in r.getMessage() for r in caplog.records)