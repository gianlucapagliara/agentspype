"""Tests for the 'agentspype ui' CLI subcommand."""

from __future__ import annotations

from agentspype.runner.cli import build_parser


class TestUISubcommand:
    def test_ui_subcommand_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["ui", "mypackage.agents"])
        assert args.command == "ui"
        assert args.modules == ["mypackage.agents"]

    def test_ui_default_host_and_port(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["ui", "mypackage.agents"])
        assert args.host == "127.0.0.1"
        assert args.port == 8765

    def test_ui_custom_host_and_port(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["ui", "mypackage.agents", "--host", "0.0.0.0", "-p", "9000"]
        )
        assert args.host == "0.0.0.0"
        assert args.port == 9000

    def test_ui_multiple_modules(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["ui", "pkg.a", "pkg.b"])
        assert args.modules == ["pkg.a", "pkg.b"]
