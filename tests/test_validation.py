import sys
import contextlib
from pathlib import Path
from datetime import datetime, timedelta

import pytest

# Under test
# Prefer top-level validation.py (as provided in PR diff)
import importlib

validation = importlib.import_module("validation")
ValidationError = validation.ValidationError

# -----------------------
# Fixtures and utilities
# -----------------------

@pytest.fixture(autouse=True)
def reset_rate_limit_store():
    # Ensure isolation between tests for rate limiting
    with contextlib.suppress(Exception):
        validation._rate_limit_store.clear()
    yield
    with contextlib.suppress(Exception):
        validation._rate_limit_store.clear()


# -----------------------
# sanitize_string
# -----------------------

def test_sanitize_string_happy_path_trims_normalizes_and_returns():
    s = "  Hello \n\t world   "
    out = validation.sanitize_string(s, max_length=100)
    assert out == "Hello world"


def test_sanitize_string_removes_control_chars_and_enforces_length():
    ctrl = "Hi\x00there\x1F!"
    out = validation.sanitize_string(ctrl, max_length=10)
    assert out == "Hithere!"
    # length limit
    with pytest.raises(ValidationError) as ei:
        validation.sanitize_string("a" * 6, max_length=5)
    assert ei.value.code == "LENGTH_ERROR"
    assert ei.value.field == "input"


def test_sanitize_string_type_and_empty_checks():
    with pytest.raises(ValidationError) as ei:
        validation.sanitize_string(123)  # type: ignore
    assert ei.value.code == "TYPE_ERROR"
    with pytest.raises(ValidationError) as ei2:
        validation.sanitize_string("   \n\t   ")
    assert ei2.value.code == "EMPTY_ERROR"


# -----------------------
# validate_topic
# -----------------------

def test_validate_topic_valid_and_min_length_enforced():
    assert validation.validate_topic("Announce product launch") == "Announce product launch"
    with pytest.raises(ValidationError) as ei:
        validation.validate_topic("Hi", min_length=3)
    assert ei.value.code == "LENGTH_ERROR"
    assert ei.value.field == "topic"


def test_validate_topic_blocks_malicious_patterns_script_and_event_handlers():
    for bad in [
        '<script>alert(1)</script>',
        'Check this out javascript:alert(1)',
        'Click me onclick=alert(1)',
        '<iframe src="x"></iframe>',
        '<object data="x"></object>',
        '<embed src="x">',
        'data:text/html;base64,AAA'
    ]:
        with pytest.raises(ValidationError) as ei:
            validation.validate_topic(bad)
        assert ei.value.code == "SECURITY_ERROR"
        assert ei.value.field == "topic"


def test_validate_topic_excessive_repetition():
    # 4 repetitions of "spam" should trigger CONTENT_ERROR
    topic = "spam spam spam spam eggs"
    with pytest.raises(ValidationError) as ei:
        validation.validate_topic(topic)
    assert ei.value.code == "CONTENT_ERROR"
    assert ei.value.field == "topic"


# -----------------------
# normalize_platform_name
# -----------------------

def test_normalize_platform_name_mappings_and_whitespace():
    cases = {
        " X ": "twitter",
        "fb": "facebook",
        "IG": "instagram",
        "yt": "youtube",
        "tt": "tiktok",
        "Li": "linkedin",
        "twitter": "twitter",
    }
    for raw, expected in cases.items():
        assert validation.normalize_platform_name(raw) == expected


def test_normalize_platform_name_type_error():
    with pytest.raises(ValidationError) as ei:
        validation.normalize_platform_name(123)  # type: ignore
    assert ei.value.code == "TYPE_ERROR"
    assert ei.value.field == "platform"


# -----------------------
# validate_platforms
# -----------------------

def test_validate_platforms_happy_path_and_order_preserved():
    platforms = validation.validate_platforms("twitter, linkedin, instagram")
    assert platforms == ["twitter", "linkedin", "instagram"]


def test_validate_platforms_empty_and_type_and_count_and_duplicates():
    with pytest.raises(ValidationError) as ei:
        validation.validate_platforms("")
    assert ei.value.code == "EMPTY_ERROR"
    assert ei.value.field == "platforms"

    with pytest.raises(ValidationError) as ei2:
        validation.validate_platforms(123)  # type: ignore
    assert ei2.value.code == "TYPE_ERROR"
    assert ei2.value.field == "platforms"

    too_many = ",".join(["twitter"] * 11)
    with pytest.raises(ValidationError) as ei3:
        validation.validate_platforms(too_many, max_platforms=10)
    assert ei3.value.code == "COUNT_ERROR"

    # duplicate after normalization: "x" maps to "twitter"
    with pytest.raises(ValidationError) as ei4:
        validation.validate_platforms("twitter, x")
    assert ei4.value.code == "DUPLICATE_PLATFORM"


def test_validate_platforms_unsupported():
    with pytest.raises(ValidationError) as ei:
        validation.validate_platforms("myspace")
    assert ei.value.code == "UNSUPPORTED_PLATFORM"
    assert "Unsupported platform" in ei.value.message


# -----------------------
# validate_brand_bible_content
# -----------------------

def test_validate_brand_bible_content_happy_and_size_and_type_and_security():
    content = "<brand>Test</brand>"
    assert validation.validate_brand_bible_content(content) == content

    with pytest.raises(ValidationError) as ei:
        validation.validate_brand_bible_content(123)  # type: ignore
    assert ei.value.code == "TYPE_ERROR"
    assert ei.value.field == "brand_bible"

    big = "X" * 21
    with pytest.raises(ValidationError) as ei2:
        validation.validate_brand_bible_content(big, max_size=20)
    assert ei2.value.code == "SIZE_ERROR"

    for bad in [
        '<script>alert(1)</script>',
        '<form action="/x">',
        '<input type="text">',
        'javascript:alert(1)',
        'onload=evil()',
        'data:text/html;base64,ZZZ'
    ]:
        with pytest.raises(ValidationError) as ei3:
            validation.validate_brand_bible_content(bad)
        assert ei3.value.code == "SECURITY_ERROR"


# -----------------------
# validate_file_upload
# -----------------------

def test_validate_file_upload_happy_path(tmp_path: Path):
    p = tmp_path / "ok.json"
    p.write_text('{"k": "v"}', encoding="utf-8")
    meta = validation.validate_file_upload(str(p))
    assert meta["path"].endswith("ok.json")
    assert meta["name"] == "ok.json"
    assert meta["size"] == p.stat().st_size
    assert meta["extension"] == ".json"
    assert isinstance(meta["modified"], datetime)


def test_validate_file_upload_not_found(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValidationError) as ei:
        validation.validate_file_upload(str(missing))
    assert ei.value.code == "NOT_FOUND"


def test_validate_file_upload_size_limit(tmp_path: Path):
    p = tmp_path / "big.txt"
    p.write_text("X" * 50, encoding="utf-8")
    with pytest.raises(ValidationError) as ei:
        validation.validate_file_upload(str(p), max_size=10)
    assert ei.value.code == "SIZE_ERROR"


def test_validate_file_upload_extension_check(tmp_path: Path):
    p = tmp_path / "bad.exe"
    p.write_bytes(b"\x00\x01")
    with pytest.raises(ValidationError) as ei:
        validation.validate_file_upload(str(p))
    assert ei.value.code == "TYPE_ERROR"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Creating filenames with reserved characters is unsupported on Windows")
def test_validate_file_upload_suspicious_characters_in_filename(tmp_path: Path):
    # Use a character allowed on POSIX to trigger SECURITY_ERROR
    p = tmp_path / "bad|name.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError) as ei:
        validation.validate_file_upload(str(p))
    assert ei.value.code == "SECURITY_ERROR"


def test_validate_file_upload_directory_traversal_pattern(tmp_path: Path):
    # Create structure: tmp/foo/evil.json exists; path we pass includes '../'
    base = tmp_path / "foo"
    (base / "bar").mkdir(parents=True, exist_ok=True)
    target = base / "evil.json"
    target.write_text("{}", encoding="utf-8")
    traversal_path = str(base / "bar" / ".." / "evil.json")  # contains '../'
    # Sanity check: file exists at normalized path
    assert Path(traversal_path).exists()
    with pytest.raises(ValidationError) as ei:
        validation.validate_file_upload(traversal_path)
    assert ei.value.code == "SECURITY_ERROR"


# -----------------------
# check_rate_limit
# -----------------------

def test_check_rate_limit_allows_until_limit_then_blocks():
    ident = "user-1"
    assert validation.check_rate_limit(ident, max_requests=2, window_seconds=60) is True
    assert validation.check_rate_limit(ident, max_requests=2, window_seconds=60) is True
    assert validation.check_rate_limit(ident, max_requests=2, window_seconds=60) is False


def test_check_rate_limit_cleans_old_entries():
    ident = "user-old"
    now = datetime.now()
    window = 5
    # Insert very old timestamps, which should be cleaned
    validation._rate_limit_store[ident] = [
        now - timedelta(seconds=window + 10),
        now - timedelta(seconds=window + 100),
    ]
    assert validation.check_rate_limit(ident, max_requests=1, window_seconds=window) is True
    # Second call should hit limit now
    assert validation.check_rate_limit(ident, max_requests=1, window_seconds=window) is False


# -----------------------
# validate_shared_store
# -----------------------

def test_validate_shared_store_happy_path():
    shared = {
        "task_requirements": {
            "topic_or_goal": "Launch day update",
            "platforms": ["twitter", "linkedin"],
        },
        "brand_bible": {"xml_raw": "<brand>OK</brand>"},
        "stream": None,
    }
    out = validation.validate_shared_store(shared)
    assert out["task_requirements"]["platforms"] == ["twitter", "linkedin"]
    assert out["task_requirements"]["topic_or_goal"] == "Launch day update"
    assert "validation_timestamp" in out and isinstance(out["validation_timestamp"], str)


def test_validate_shared_store_type_and_required_and_nested_errors():
    with pytest.raises(ValidationError) as ei:
        validation.validate_shared_store(["not-a-dict"])  # type: ignore
    assert ei.value.field == "shared" and ei.value.code == "TYPE_ERROR"

    with pytest.raises(ValidationError) as ei2:
        validation.validate_shared_store({})
    assert ei2.value.field == "task_requirements" and ei2.value.code == "MISSING"

    with pytest.raises(ValidationError) as ei3:
        validation.validate_shared_store({"task_requirements": "oops"})  # not a dict
    assert ei3.value.field == "task_requirements" and ei3.value.code == "TYPE_ERROR"

    with pytest.raises(ValidationError) as ei4:
        validation.validate_shared_store({"task_requirements": {}})
    assert ei4.value.field == "topic_or_goal" and ei4.value.code == "MISSING"

    with pytest.raises(ValidationError) as ei5:
        validation.validate_shared_store({"task_requirements": {"topic_or_goal": "ok"}})
    assert ei5.value.field == "platforms" and ei5.value.code == "MISSING"

    with pytest.raises(ValidationError) as ei6:
        validation.validate_shared_store({"task_requirements": {"topic_or_goal": "ok", "platforms": 123}})
    assert ei6.value.field == "platforms" and ei6.value.code == "TYPE_ERROR"

    with pytest.raises(ValidationError) as ei7:
        validation.validate_shared_store({
            "task_requirements": {
                "topic_or_goal": "ok",
                "platforms": "unknown"
            }
        })
    assert ei7.value.field == "platforms"


def test_validate_shared_store_brand_bible_validation_bubbles_up():
    with pytest.raises(ValidationError) as ei:
        validation.validate_shared_store({
            "task_requirements": {"topic_or_goal": "ok topic", "platforms": "twitter"},
            "brand_bible": {"xml_raw": "<script>alert(1)</script>"},
        })
    assert ei.value.field == "brand_bible" and ei.value.code == "SECURITY_ERROR"


# -----------------------
# sanitize_for_llm_prompt
# -----------------------

def test_sanitize_for_llm_prompt_escapes_and_strips_and_limits_length():
    text = 'Hello "world"\n<script>bad()</script>\tBack\\slash & apos\' '
    out = validation.sanitize_for_llm_prompt(text)
    # No HTML-like tags
    assert "<script>" not in out and "bad()" in out
    # Escapes
    assert '\\"' in out
    assert "\\n" in out and "\\t" in out and "\\r" not in out
    assert "\\\\" in out
    # Quotes escaped
    assert "\\'" in out

    # Length limiting
    long = "x" * 10050
    out2 = validation.sanitize_for_llm_prompt(long)
    assert out2.endswith("...")
    assert len(out2) == 10003


def test_sanitize_for_llm_prompt_non_string_returns_empty():
    assert validation.sanitize_for_llm_prompt(None) == ""  # type: ignore


# -----------------------
# validate_and_sanitize_inputs (integration-ish)
# -----------------------

def test_validate_and_sanitize_inputs_success_and_logging(caplog):
    caplog.clear()
    with caplog.at_level("INFO"):
        out = validation.validate_and_sanitize_inputs(
            topic="Announce product launch",
            platforms_text="twitter, linkedin",
            brand_bible_content="<brand>bbb</brand>",
        )
    # Structure verified
    assert out["task_requirements"]["platforms"] == ["twitter", "linkedin"]
    assert out["task_requirements"]["topic_or_goal"] == "Announce product launch"
    assert out["brand_bible"]["xml_raw"] == "<brand>bbb</brand>"
    # Log record present
    assert any("Inputs validated successfully" in rec.message for rec in caplog.records)


def test_validate_and_sanitize_inputs_raises_and_logs_warning_on_failure(caplog):
    caplog.clear()
    with pytest.raises(ValidationError), caplog.at_level("WARNING"):
        validation.validate_and_sanitize_inputs(
            topic="",  # invalid
            platforms_text="twitter, linkedin",
            brand_bible_content="<brand>bbb</brand>",
        )
    # Warning log captured
    assert any("Input validation failed" in rec.message for rec in caplog.records)


def test_validate_and_sanitize_inputs_duplicate_platform_after_normalization():
    # "x" normalizes to "twitter" -> duplicate
    with pytest.raises(ValidationError) as ei:
        validation.validate_and_sanitize_inputs(
            topic="Legit topic",
            platforms_text="twitter, x",
            brand_bible_content=""
        )
    assert ei.value.code == "DUPLICATE_PLATFORM"