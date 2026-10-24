"""CLI surface tests for the command skeleton and implemented commands."""

from __future__ import annotations

from typer.testing import CliRunner

from modelops_lab.cli import app

runner = CliRunner()


def test_help_exits_zero_and_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "data",
        "db",
        "train",
        "evaluate",
        "retrain",
        "promote",
        "serve",
        "rollback",
        "registry",
        "monitor",
        "simulate-traffic",
        "status",
        "events",
    ):
        assert command in result.output


def test_retrain_help_lists_the_workflow_options() -> None:
    result = runner.invoke(app, ["retrain", "--help"])

    assert result.exit_code == 0
    for option in (
        "--dataset",
        "--trigger",
        "--drift-report",
        "--actor",
        "--reason",
        "--run-name",
        "--tracking-uri",
        "--database-url",
    ):
        assert option in result.output


def test_serve_help_lists_host_and_port() -> None:
    result = runner.invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output


def test_rollback_requires_a_reason() -> None:
    result = runner.invoke(app, ["rollback", "--to-version", "1", "--reason", " "])

    assert result.exit_code == 2
    assert "--reason must not be empty" in result.stderr


def test_rollback_requires_a_target_version() -> None:
    result = runner.invoke(app, ["rollback", "--reason", "because"])

    assert result.exit_code == 2
    assert "--to-version" in result.stderr


def test_rollback_rejects_an_empty_target_version() -> None:
    result = runner.invoke(app, ["rollback", "--to-version", " ", "--reason", "because"])

    assert result.exit_code == 2
    assert "--to-version must not be empty" in result.stderr


def test_registry_requires_a_subcommand() -> None:
    result = runner.invoke(app, ["registry"])

    assert result.exit_code != 0
    assert "reconcile" in result.output
