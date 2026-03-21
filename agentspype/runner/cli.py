"""CLI entry point for the agentspype runner.

Usage::

    agentspype run -f config.yaml
    agentspype run -f config.yaml --log-level DEBUG
    agentspype create MyNewAgent agents/my_new_agent
    agentspype plot mypackage.agents.my_agent -o .diagrams
    python -m agentspype.runner.cli run -f config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import warnings
from pathlib import Path

from agentspype.runner.config import (
    LocalConfigSource,
    RunnerConfig,
    generate_instance_prefix,
    load_agent_configs,
)
from agentspype.runner.runner import AgentRunner

# ---------------------------------------------------------------------------
# Concrete runner for CLI usage
# ---------------------------------------------------------------------------


class _CLIRunner(AgentRunner):
    """Minimal concrete runner used by the CLI ``run`` command."""

    pass


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Exposed publicly so that downstream projects can call
    ``build_parser()`` and extend it with additional subcommands before
    invoking ``parser.parse_args()``.
    """
    parser = argparse.ArgumentParser(
        prog="agentspype",
        description="CLI runner for agentspype agents.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- run ----------------------------------------------------------------
    run_parser = subparsers.add_parser("run", help="Run agents from a config file.")
    run_parser.add_argument(
        "-f",
        "--config-file",
        type=Path,
        required=True,
        help="Path to the YAML agent configuration file.",
    )
    run_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO).",
    )
    run_parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for agents to reach final state before releasing runtime resources (default: 30).",
    )

    # -- create -------------------------------------------------------------
    create_parser = subparsers.add_parser(
        "create", help="Scaffold a new agent from the template."
    )
    create_parser.add_argument(
        "name", type=str, help="Name of the new agent (CamelCase or snake_case)."
    )
    create_parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=".",
        help="Output directory for the new agent (default: current directory).",
    )

    # -- plot ---------------------------------------------------------------
    plot_parser = subparsers.add_parser("plot", help="Generate agent diagrams.")
    plot_parser.add_argument(
        "modules",
        type=str,
        nargs="+",
        help="Dotted module path(s) containing Agent subclasses.",
    )
    plot_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".diagrams",
        help="Directory to save diagrams (default: .diagrams).",
    )

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _run_command(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level))

    config = RunnerConfig(
        config_file=args.config_file,
        log_level=args.log_level,
        shutdown_timeout=args.shutdown_timeout,
    )

    source = LocalConfigSource(config.config_file)

    async def _run() -> None:
        result = await source.load()
        prefix = config.instance_prefix or generate_instance_prefix(config.config_file)
        agent_configs = load_agent_configs(result.config, prefix)
        runner = _CLIRunner(
            agent_configs,  # type: ignore[arg-type]  # raw dicts resolved by Agency
            shutdown_timeout=config.shutdown_timeout,
        )
        await runner.run()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logging.getLogger(__name__).error("Run failed: %s", exc)
        raise SystemExit(1) from exc


def _create_command(args: argparse.Namespace) -> None:
    from agentspype.runner.create import copy_template

    try:
        copy_template(args.name, Path(args.path))
    except Exception as exc:
        logging.getLogger(__name__).error("Create failed: %s", exc)
        raise SystemExit(1) from exc


def _plot_command(args: argparse.Namespace) -> None:
    from agentspype.runner.plot import plot_agents

    try:
        plot_agents(args.modules, output_dir=args.output_dir)
    except Exception as exc:
        logging.getLogger(__name__).error("Plot failed: %s", exc)
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    # Suppress DeprecationWarnings from third-party libraries that can be
    # noisy in a CLI context (e.g. pydantic v1 shims, pkg_resources).
    # Only specific categories are filtered to avoid hiding real issues.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        _run_command(args)
    elif args.command == "create":
        _create_command(args)
    elif args.command == "plot":
        _plot_command(args)


if __name__ == "__main__":
    main()
