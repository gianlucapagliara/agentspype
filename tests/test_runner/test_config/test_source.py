"""Tests for ConfigSource protocol and compute_config_hash."""

from __future__ import annotations

from agentspype.runner.config.source import (
    ConfigLoadResult,
    ConfigSource,
    compute_config_hash,
)


class TestComputeConfigHash:
    def test_deterministic(self) -> None:
        config = {"b": 2, "a": 1}
        assert compute_config_hash(config) == compute_config_hash(config)

    def test_order_independent(self) -> None:
        assert compute_config_hash({"a": 1, "b": 2}) == compute_config_hash(
            {"b": 2, "a": 1}
        )

    def test_different_configs_differ(self) -> None:
        assert compute_config_hash({"a": 1}) != compute_config_hash({"a": 2})

    def test_returns_hex_string(self) -> None:
        h = compute_config_hash({"key": "value"})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex length


class TestConfigLoadResult:
    def test_auto_hash(self) -> None:
        result = ConfigLoadResult(config={"x": 1})
        assert result.config_hash == compute_config_hash({"x": 1})

    def test_explicit_hash_preserved(self) -> None:
        result = ConfigLoadResult(config={"x": 1}, config_hash="custom")
        assert result.config_hash == "custom"

    def test_metadata_defaults_to_empty(self) -> None:
        result = ConfigLoadResult(config={})
        assert result.metadata == {}


class TestConfigSourceProtocol:
    def test_protocol_isinstance_check(self) -> None:
        class _FakeSource:
            async def load(self) -> ConfigLoadResult:
                return ConfigLoadResult(config={})

        assert isinstance(_FakeSource(), ConfigSource)

    def test_non_conforming_class_rejected(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), ConfigSource)
