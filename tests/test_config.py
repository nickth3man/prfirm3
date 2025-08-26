# Tests for config.py
# Framework: pytest (with built-in fixtures: tmp_path, monkeypatch, caplog)
# These tests focus on validating configuration loading, precedence, validation, and template creation.

import yaml
import importlib

import pytest

# Ensure we import the module under test freshly for each scenario

# === BEGIN: generated test suite for config.py ===

@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    # Clear relevant env vars before each test to avoid cross-test contamination
    keys = [
        "LOG_LEVEL","LOG_FORMAT","LOG_FILE",
        "GRADIO_PORT","GRADIO_HOST","GRADIO_SHARE","DEMO_PASSWORD",
        "LLM_PROVIDER","LLM_MODEL","LLM_TEMPERATURE","LLM_MAX_TOKENS",
        "ENABLE_AUTH","RATE_LIMIT_REQUESTS",
        "REDIS_URL","ENABLE_CACHE",
        "DEBUG","CONFIG_FILE"
    ]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    yield


def reload_config_module():
    # Reload the module to ensure no lingering state
    import config as _cfg
    importlib.reload(_cfg)
    return _cfg


def test_defaults_are_applied():
    cfg_mod = reload_config_module()
    cfg = cfg_mod.get_config()

    # AppConfig defaults
    assert cfg.debug is False
    assert cfg.log_level == "INFO"
    assert cfg.config_file is None

    # Logging defaults
    assert cfg.logging.level == "INFO"
    assert cfg.logging.format == "json"
    assert cfg.logging.file is None
    assert cfg.logging.max_size == 10 * 1024 * 1024
    assert cfg.logging.backup_count == 5

    # Gradio defaults
    assert cfg.gradio.port == 7860
    assert cfg.gradio.host == "0.0.0.0"
    assert cfg.gradio.share is False
    assert cfg.gradio.auth is None
    assert cfg.gradio.ssl_verify is True
    assert cfg.gradio.show_error is True

    # LLM defaults
    assert cfg.llm.provider == "openai"
    assert cfg.llm.model == "gpt-4o"
    assert cfg.llm.temperature == 0.7
    assert cfg.llm.max_tokens == 2000
    assert cfg.llm.timeout == 30
    assert cfg.llm.max_retries == 3
    assert cfg.llm.retry_delay == 1

    # Security defaults
    assert cfg.security.enable_auth is False
    assert cfg.security.session_timeout == 3600
    assert cfg.security.rate_limit_requests == 60
    assert cfg.security.rate_limit_window == 60
    assert cfg.security.max_file_size == 10 * 1024 * 1024
    assert cfg.security.allowed_file_types == [".xml", ".json", ".txt"]

    # Cache defaults
    assert cfg.cache.enable_cache is True
    assert cfg.cache.ttl == 3600
    assert cfg.cache.max_size == 1000
    assert cfg.cache.redis_url is None


@pytest.mark.parametrize("val,expected", [
    ("true", True), ("TRUE", True), ("1", True), ("yes", True), ("YeS", True),
    ("false", False), ("0", False), ("no", False), ("anything-else", False)
])
def test_boolean_env_parsing_for_share(monkeypatch, val, expected):
    monkeypatch.setenv("GRADIO_SHARE", val)
    cfg = reload_config_module().get_config()
    assert cfg.gradio.share is expected


def test_env_overrides_basic(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "human")
    monkeypatch.setenv("LOG_FILE", "/tmp/app.log")

    monkeypatch.setenv("GRADIO_PORT", "5050")
    monkeypatch.setenv("GRADIO_HOST", "127.0.0.1")
    monkeypatch.setenv("DEMO_PASSWORD", "s3cr3t")

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-3-opus")
    monkeypatch.setenv("LLM_TEMPERATURE", "1.2")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")

    monkeypatch.setenv("ENABLE_AUTH", "yes")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "100")

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("ENABLE_CACHE", "1")

    monkeypatch.setenv("DEBUG", "true")

    cfg = reload_config_module().get_config()

    assert cfg.logging.level == "DEBUG"
    assert cfg.logging.format == "human"
    assert cfg.logging.file == "/tmp/app.log"

    assert cfg.gradio.port == 5050
    assert cfg.gradio.host == "127.0.0.1"
    assert cfg.gradio.auth == "admin:s3cr3t"

    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == "claude-3-opus"
    assert cfg.llm.temperature == 1.2
    assert cfg.llm.max_tokens == 4096

    assert cfg.security.enable_auth is True
    assert cfg.security.rate_limit_requests == 100

    assert cfg.cache.redis_url == "redis://localhost:6379/0"
    assert cfg.cache.enable_cache is True

    assert cfg.debug is True


def test_validation_resets_invalid_values_from_env(caplog, monkeypatch):
    # Set invalid values to trigger validation
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")  # invalid
    monkeypatch.setenv("GRADIO_PORT", "70000")  # out of range
    monkeypatch.setenv("LLM_TEMPERATURE", "3.5")  # out of range
    monkeypatch.setenv("LLM_MAX_TOKENS", "-10")  # invalid

    caplog.clear()
    caplog.set_level("WARNING")
    cfg = reload_config_module().get_config()

    # Values are corrected
    assert cfg.logging.level == "INFO"
    assert cfg.gradio.port == 7860
    assert cfg.llm.temperature == 0.7
    assert cfg.llm.max_tokens == 2000

    # Warnings logged
    warnings = "\n".join(r.message for r in caplog.records if r.levelname == "WARNING")
    assert "Invalid log level" in warnings
    assert "Invalid port" in warnings
    assert "Invalid temperature" in warnings
    assert "Invalid max_tokens" in warnings


def test_load_from_yaml_file(tmp_path):
    cfg_path = tmp_path / "app.yaml"
    cfg_yaml = {
        "logging": {"level": "ERROR", "format": "human", "file": str(tmp_path / "log.txt")},
        "gradio": {"port": 9000, "host": "localhost", "share": True},
        "llm": {"provider": "google", "model": "gemini-1.5-pro", "temperature": 0.3, "max_tokens": 1234},
        "security": {"enable_auth": True, "rate_limit_requests": 42},
        "cache": {"enable_cache": False, "ttl": 120, "max_size": 10, "redis_url": "redis://example/1"},
    }
    cfg_path.write_text(yaml.safe_dump(cfg_yaml), encoding="utf-8")

    cfg = reload_config_module().get_config(config_file=str(cfg_path))

    assert cfg.logging.level == "ERROR"
    assert cfg.logging.format == "human"
    assert cfg.logging.file == str(tmp_path / "log.txt")

    assert cfg.gradio.port == 9000
    assert cfg.gradio.host == "localhost"
    assert cfg.gradio.share is True

    assert cfg.llm.provider == "google"
    assert cfg.llm.model == "gemini-1.5-pro"
    assert cfg.llm.temperature == 0.3
    assert cfg.llm.max_tokens == 1234

    assert cfg.security.enable_auth is True
    assert cfg.security.rate_limit_requests == 42

    assert cfg.cache.enable_cache is False
    assert cfg.cache.ttl == 120
    assert cfg.cache.max_size == 10
    assert cfg.cache.redis_url == "redis://example/1"


def test_missing_yaml_file_logs_warning(tmp_path, caplog):
    non_existent = tmp_path / "nope.yaml"
    caplog.clear()
    caplog.set_level("WARNING")
    cfg = reload_config_module().get_config(config_file=str(non_existent))
    # Just assert that it did not crash and logged a warning
    warnings = "\n".join(r.message for r in caplog.records if r.levelname == "WARNING")
    assert "Configuration file not found" in warnings
    # Defaults still present
    assert cfg.gradio.port == 7860


def test_invalid_yaml_logs_error(tmp_path, caplog):
    bad = tmp_path / "bad.yaml"
    bad.write_text(":\n- not: valid: yaml: [", encoding="utf-8")

    caplog.clear()
    caplog.set_level("ERROR")
    # AppConfig catches exceptions during _load_from_file and logs an error
    cfg = reload_config_module().get_config(config_file=str(bad))
    errors = "\n".join(r.message for r in caplog.records if r.levelname == "ERROR")
    assert "Failed to load configuration file" in errors
    # Still returns a valid config object
    assert cfg.logging.level == "INFO"


def test_apply_dict_config_ignores_unknown_sections_and_keys(tmp_path):
    cfg_yaml = {
        "unknown_section": {"foo": "bar"},
        "logging": {"level": "ERROR", "unknown_key": True},
    }
    path = tmp_path / "file.yaml"
    path.write_text(yaml.safe_dump(cfg_yaml), encoding="utf-8")

    cfg = reload_config_module().get_config(config_file=str(path))
    # logging.level applied, unknown_key ignored, unknown_section ignored
    assert cfg.logging.level == "ERROR"
    assert not hasattr(cfg.logging, "unknown_key")


def test_env_takes_precedence_over_file(tmp_path, monkeypatch):
    # According to module docstring, priority should be: CLI args > env vars > config file > defaults
    # Here we ensure env overrides values present in the file.
    cfg_yaml = {
        "gradio": {"port": 9999, "host": "filehost"},
        "llm": {"temperature": 0.1}
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg_yaml), encoding="utf-8")

    monkeypatch.setenv("GRADIO_PORT", "1234")
    monkeypatch.setenv("LLM_TEMPERATURE", "1.5")

    cfg = reload_config_module().get_config(config_file=str(p))

    assert cfg.gradio.port == 1234, "Environment variable should take precedence over file value"
    assert cfg.llm.temperature == 1.5, "Environment variable should take precedence over file value"


def test_config_file_can_be_provided_via_env_var(tmp_path, monkeypatch):
    cfg_yaml = {
        "logging": {"level": "ERROR"}
    }
    p = tmp_path / "from_env.yaml"
    p.write_text(yaml.safe_dump(cfg_yaml), encoding="utf-8")

    monkeypatch.setenv("CONFIG_FILE", str(p))
    cfg = reload_config_module().get_config()
    assert cfg.logging.level == "ERROR"


def test_create_config_template_writes_expected_keys(tmp_path, capsys):
    out = tmp_path / "template.yaml"
    cfg_mod = reload_config_module()
    cfg_mod.create_config_template(output_path=str(out))

    # stdout should mention created file
    captured = capsys.readouterr()
    assert str(out) in captured.out

    # Parse YAML and validate presence of key structure and default-derived values
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    for top in ["logging", "gradio", "llm", "security", "cache"]:
        assert top in data

    assert "level" in data["logging"]
    assert "format" in data["logging"]
    assert "file" in data["logging"]

    assert "port" in data["gradio"]
    assert "host" in data["gradio"]
    assert "share" in data["gradio"]

    assert "provider" in data["llm"]
    assert "model" in data["llm"]
    assert "temperature" in data["llm"]
    assert "max_tokens" in data["llm"]

    assert "enable_auth" in data["security"]
    assert "rate_limit_requests" in data["security"]

    assert "enable_cache" in data["cache"]
    assert "ttl" in data["cache"]


def test_allowed_file_types_is_per_instance_not_shared():
    # Ensure mutable defaults are not shared between instances
    cfg1 = reload_config_module().get_config()
    cfg2 = reload_config_module().get_config()
    # Mutate one instance
    cfg1.security.allowed_file_types.append(".pdf")
    # The other instance should not reflect the change
    assert ".pdf" not in cfg2.security.allowed_file_types


def test_debug_env_true_false_variants(monkeypatch):
    for val, expected in [("true", True), ("False", False), ("1", True), ("0", False), ("yes", True), ("no", False)]:
        monkeypatch.setenv("DEBUG", val)
        cfg = reload_config_module().get_config()
        assert cfg.debug is expected


def test_rate_limit_requests_parsing(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "250")
    cfg = reload_config_module().get_config()
    assert cfg.security.rate_limit_requests == 250


def test_redis_url_and_enable_cache(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://cache:6379/2")
    monkeypatch.setenv("ENABLE_CACHE", "yes")
    cfg = reload_config_module().get_config()
    assert cfg.cache.redis_url == "redis://cache:6379/2"
    assert cfg.cache.enable_cache is True


# === END: generated test suite for config.py ===