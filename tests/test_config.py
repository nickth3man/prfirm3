# pytest is the chosen testing framework for this repository.
# These tests validate the configuration management module (AppConfig, get_config, create_config_template).
# They focus on environment variable loading, YAML file precedence, validation, and template generation.

import os
import sys
import yaml
from pathlib import Path

import pytest

# Attempt to import the config module from common locations
# If the project uses a src/ layout, ensure it is importable in tests.
try:
    import config  # noqa: F401
except ModuleNotFoundError:
    # Try to add 'src' to path if present
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

from config import AppConfig, get_config, create_config_template  # type: ignore

@pytest.fixture(autouse=True)
def clear_relevant_env(monkeypatch):
    # Clear environment variables that the config reads so tests are isolated
    keys = [
        "LOG_LEVEL", "LOG_FORMAT", "LOG_FILE",
        "GRADIO_PORT", "GRADIO_HOST", "GRADIO_SHARE", "DEMO_PASSWORD",
        "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS",
        "ENABLE_AUTH", "RATE_LIMIT_REQUESTS",
        "REDIS_URL", "ENABLE_CACHE",
        "DEBUG", "CONFIG_FILE"
    ]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    yield
    for k in keys:
        monkeypatch.delenv(k, raising=False)

def test_defaults_are_sensible():
    cfg = AppConfig()
    assert cfg.logging.level == "INFO"
    assert cfg.logging.format == "json"
    assert cfg.gradio.port == 7860
    assert cfg.gradio.host == "0.0.0.0"
    assert cfg.gradio.share is False
    assert cfg.llm.provider == "openai"
    assert cfg.llm.model == "gpt-4o"
    assert cfg.llm.temperature == 0.7
    assert cfg.llm.max_tokens == 2000
    assert cfg.security.enable_auth is False
    assert cfg.security.rate_limit_requests == 60
    assert cfg.cache.enable_cache is True
    assert cfg.cache.ttl == 3600
    assert cfg.debug is False

def test_env_overrides_nested_settings(monkeypatch):
    monkeypatch.setenv("GRADIO_PORT", "8080")
    monkeypatch.setenv("GRADIO_HOST", "127.0.0.1")
    monkeypatch.setenv("GRADIO_SHARE", "yes")
    monkeypatch.setenv("DEMO_PASSWORD", "s3cr3t")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-3-haiku")
    monkeypatch.setenv("LLM_TEMPERATURE", "1.25")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
    monkeypatch.setenv("ENABLE_AUTH", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "123")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("ENABLE_CACHE", "1")
    monkeypatch.setenv("DEBUG", "1")

    cfg = AppConfig()

    assert cfg.gradio.port == 8080
    assert cfg.gradio.host == "127.0.0.1"
    assert cfg.gradio.share is True
    assert cfg.gradio.auth == "admin:s3cr3t"
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == "claude-3-haiku"
    assert cfg.llm.temperature == pytest.approx(1.25)
    assert cfg.llm.max_tokens == 4096
    assert cfg.security.enable_auth is True
    assert cfg.security.rate_limit_requests == 123
    assert cfg.cache.redis_url == "redis://localhost:6379/0"
    assert cfg.cache.enable_cache is True
    assert cfg.debug is True

def test_yaml_file_applies_when_present(tmp_path, caplog):
    config_yaml = tmp_path / "app.yaml"
    yaml_data = {
        "logging": {"level": "DEBUG", "format": "human", "file": "app.log"},
        "gradio": {"port": 9999, "host": "0.0.0.0", "share": True},
        "llm": {"provider": "google", "model": "gemini-pro", "temperature": 0.5, "max_tokens": 1000},
        "security": {"enable_auth": True, "rate_limit_requests": 77},
        "cache": {"enable_cache": False, "ttl": 120, "redis_url": "redis://example:6379/1"},
    }
    config_yaml.write_text(yaml.dump(yaml_data))

    cfg = AppConfig(config_file=str(config_yaml))

    assert cfg.logging.level == "DEBUG"
    assert cfg.logging.format == "human"
    assert cfg.logging.file == "app.log"
    assert cfg.gradio.port == 9999
    assert cfg.gradio.share is True
    assert cfg.llm.provider == "google"
    assert cfg.llm.model == "gemini-pro"
    assert cfg.llm.temperature == pytest.approx(0.5)
    assert cfg.llm.max_tokens == 1000
    assert cfg.security.enable_auth is True
    assert cfg.security.rate_limit_requests == 77
    assert cfg.cache.enable_cache is False
    assert cfg.cache.ttl == 120
    assert cfg.cache.redis_url == "redis://example:6379/1"
    assert any("Loaded configuration from" in rec.message for rec in caplog.records)

def test_env_precedence_over_yaml(tmp_path, monkeypatch):
    config_yaml = tmp_path / "app.yaml"
    yaml_data = {
        "llm": {"model": "file-model"},
        "gradio": {"port": 5555},
    }
    config_yaml.write_text(yaml.dump(yaml_data))

    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("GRADIO_PORT", "7777")

    cfg = AppConfig(config_file=str(config_yaml))

    assert cfg.llm.model == "env-model", "Environment should override file"
    assert cfg.gradio.port == 7777, "Environment should override file"

def test_nonexistent_config_file_logs_warning(tmp_path, caplog):
    fake_file = tmp_path / "missing.yaml"
    cfg = AppConfig(config_file=str(fake_file))
    # Ensure we didn't crash, and default gradio port remains default
    assert cfg.gradio.port == 7860
    assert any("Configuration file not found" in rec.message for rec in caplog.records)

def test_invalid_yaml_logs_error(tmp_path, caplog):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(":\n- this: is: not: valid: yaml")
    _ = AppConfig(config_file=str(bad_yaml))
    assert any("Failed to load configuration file" in rec.message for rec in caplog.records)

def test_apply_dict_ignores_unknown_sections(tmp_path):
    config_yaml = tmp_path / "app.yaml"
    yaml_data = {
        "nonexistent": {"foo": "bar"},
        "logging": {"level": "ERROR"},
    }
    config_yaml.write_text(yaml.dump(yaml_data))

    cfg = AppConfig(config_file=str(config_yaml))
    assert cfg.logging.level == "ERROR"
    # Unknown section should be ignored; ensure attribute not created
    assert not hasattr(cfg, "nonexistent")

def test_validation_resets_invalid_values(caplog):
    # Use environment to feed invalid values then ensure validation corrects them
    with caplog.at_level("WARNING"):
        os.environ["LOG_LEVEL"] = "VERBOSE"  # invalid
        os.environ["GRADIO_PORT"] = "70000"  # invalid > 65535
        os.environ["LLM_TEMPERATURE"] = "5"  # invalid > 2
        os.environ["LLM_MAX_TOKENS"] = "0"   # invalid <= 0
        try:
            cfg = AppConfig()
        finally:
            # Cleanup env
            for k in ["LOG_LEVEL", "GRADIO_PORT", "LLM_TEMPERATURE", "LLM_MAX_TOKENS"]:
                os.environ.pop(k, None)

    assert cfg.logging.level == "INFO"
    assert cfg.gradio.port == 7860
    assert cfg.llm.temperature == pytest.approx(0.7)
    assert cfg.llm.max_tokens == 2000

    msgs = [rec.message for rec in caplog.records]
    assert any("Invalid log level" in m for m in msgs)
    assert any("Invalid port" in m for m in msgs)
    assert any("Invalid temperature" in m for m in msgs)
    assert any("Invalid max_tokens" in m for m in msgs)

def test_get_config_wraps_AppConfig(tmp_path, monkeypatch):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump({"gradio": {"port": 8123}}))
    cfg = get_config(config_file=str(cfg_file))
    assert isinstance(cfg, AppConfig)
    assert cfg.gradio.port == 8123

def test_create_config_template_generates_expected_yaml(tmp_path):
    out = tmp_path / "template.yaml"
    create_config_template(output_path=str(out))
    assert out.exists()
    data = yaml.safe_load(out.read_text())
    # Check high-level structure and required keys
    for top in ["logging", "gradio", "llm", "security", "cache"]:
        assert top in data, f"Missing top-level key: {top}"
    # Spot-check some default values in the template
    assert "level" in data["logging"]
    assert "port" in data["gradio"]
    assert "provider" in data["llm"]
    assert "enable_auth" in data["security"]
    assert "enable_cache" in data["cache"]

@pytest.mark.parametrize("truthy", ["true", "True", "1", "yes", "YES"])
def test_boolean_parsing_truthy(monkeypatch, truthy):
    monkeypatch.setenv("ENABLE_AUTH", truthy)
    monkeypatch.setenv("GRADIO_SHARE", truthy)
    monkeypatch.setenv("ENABLE_CACHE", truthy)
    cfg = AppConfig()
    assert cfg.security.enable_auth is True
    assert cfg.gradio.share is True
    assert cfg.cache.enable_cache is True

@pytest.mark.parametrize("falsy", ["false", "False", "0", "no", "NO"])
def test_boolean_parsing_falsy(monkeypatch, falsy):
    monkeypatch.setenv("ENABLE_AUTH", falsy)
    monkeypatch.setenv("GRADIO_SHARE", falsy)
    monkeypatch.setenv("ENABLE_CACHE", falsy)
    cfg = AppConfig()
    assert cfg.security.enable_auth is False
    assert cfg.gradio.share is False
    assert cfg.cache.enable_cache is False

def test_demo_password_sets_auth(monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", "pw123")
    cfg = AppConfig()
    assert cfg.gradio.auth == "admin:pw123"

def test_rate_limit_requests_env(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "999")
    cfg = AppConfig()
    assert cfg.security.rate_limit_requests == 999

def test_redis_url_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://user:pass@host:6379/2")
    cfg = AppConfig()
    assert cfg.cache.redis_url == "redis://user:pass@host:6379/2"