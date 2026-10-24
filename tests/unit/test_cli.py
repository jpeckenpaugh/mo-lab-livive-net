"""CLI surface tests for the command skeleton and the implemented data group."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from modelops_lab.cli import app

runner = CliRunner()

COMMANDS = {
    "train": "W2",
    "evaluate": "W2",
    "promote": "W3",
    "serve": "W4",
    "monitor": "W5",
    "retrain": "W6",
    "rollback": "W4",
    "status": "W5",
}


def test_help_exits_zero_and_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("data", *COMMANDS):
        assert command in result.output


@pytest.mark.parametrize(("command", "wave"), COMMANDS.items())
def test_stub_exits_two_with_wave_message(command: str, wave: str) -> None:
    result = runner.invoke(app, [command])

    assert result.exit_code == 2
    assert f"not implemented until {wave}" in result.stderr
    assert f"modelops {command}" in result.stderr
