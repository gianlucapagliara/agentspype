"""Tests for the CLI parser and subcommands."""

from __future__ import annotations

from agentspype.runner.cli import build_parser


class TestBuildParser:
    def test_parser_has_run_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "-f", "config.yaml"])
        assert args.command == "run"

    def test_parser_has_create_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["create", "MyAgent", "agents/my_agent"])
        assert args.command == "create"
        assert args.name == "MyAgent"
        assert args.path == "agents/my_agent"

    def test_parser_has_plot_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["plot", "mypackage.agents.my_agent"])
        assert args.command == "plot"
        assert args.modules == ["mypackage.agents.my_agent"]

    def test_run_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "-f", "test.yaml"])
        assert args.log_level == "INFO"
        assert args.shutdown_timeout == 30.0

    def test_run_custom_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "-f",
                "test.yaml",
                "--log-level",
                "DEBUG",
                "--shutdown-timeout",
                "60",
            ]
        )
        assert args.log_level == "DEBUG"
        assert args.shutdown_timeout == 60.0

    def test_create_default_path(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["create", "MyAgent"])
        assert args.path == "."

    def test_plot_multiple_modules(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["plot", "mod1.agent", "mod2.agent"])
        assert args.modules == ["mod1.agent", "mod2.agent"]

    def test_plot_output_dir(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["plot", "mod.agent", "-o", "output"])
        assert args.output_dir == "output"

    def test_no_command_returns_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_parser_is_extensible(self) -> None:
        """Users should be able to add subcommands via build_parser()."""
        parser = build_parser()
        # Access the subparsers action to add a custom subcommand
        subparsers_actions = [
            action
            for action in parser._subparsers._actions
            if isinstance(action, type(parser._subparsers._actions[-1]))
            and hasattr(action, "_parser_class")
        ]
        assert len(subparsers_actions) > 0
