#!/usr/bin/env python3
"""Slidex CLI package entry."""

import warnings

import typer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*chardet.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")

app = typer.Typer(
    help="Slidex - Source-aware presentation generation and inspection",
    no_args_is_help=True,
)


def _register_commands() -> None:
    """Import heavyweight runtime modules only when the CLI executes."""
    if app.registered_commands:
        return
    from .commands import clean, config, generate, onboard, serve

    app.command()(onboard)
    app.command()(serve)
    app.command()(generate)
    app.command()(config)
    app.command()(clean)


def main() -> None:
    """Run Slidex and warn when invoked through a compatibility alias."""
    import sys
    from pathlib import Path

    executable = Path(sys.argv[0]).name
    if executable in {"pptagent", "deeppresenter"}:
        warnings.warn(
            f"The `{executable}` command is deprecated; use `slidex` instead.",
            FutureWarning,
            stacklevel=2,
        )
    _register_commands()
    app()
