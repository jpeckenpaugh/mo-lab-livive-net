"""Promotion state: active pointer, transitions, and decisions.

Revision ID: 0001_promotion_state
Revises: None
Create Date: 2026-10-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_promotion_state"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "active_model",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=False),
        sa.Column("mlflow_model_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_version", sa.String(length=255), nullable=False),
        sa.Column("dataset_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("decision_id", sa.String(length=128), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_active_model_singleton"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "model_transitions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("from_model_version", sa.String(length=64), nullable=True),
        sa.Column("to_model_version", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("decision_id", sa.String(length=128), nullable=False),
        sa.Column("actor", sa.String(length=255), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "promotion_decisions",
        sa.Column("decision_id", sa.String(length=128), nullable=False),
        sa.Column("candidate_run_id", sa.String(length=64), nullable=False),
        sa.Column("candidate_model_version", sa.String(length=64), nullable=False),
        sa.Column("active_model_version_before", sa.String(length=64), nullable=True),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("checks", sa.JSON(), nullable=False),
        sa.Column("rejection_reasons", sa.JSON(), nullable=False),
        sa.Column("actor", sa.String(length=255), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dataset_version", sa.String(length=255), nullable=False),
        sa.Column("dataset_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("evaluation_split", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("decision_id"),
    )


def downgrade() -> None:
    op.drop_table("promotion_decisions")
    op.drop_table("model_transitions")
    op.drop_table("active_model")
