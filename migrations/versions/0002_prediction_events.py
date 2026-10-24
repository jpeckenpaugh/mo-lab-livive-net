"""Prediction events: one row per served inference request.

Revision ID: 0002_prediction_events
Revises: 0001_promotion_state
Create Date: 2026-10-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_prediction_events"
down_revision: str | None = "0001_promotion_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prediction_events",
        sa.Column("request_id", sa.String(length=200), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "features",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.CheckConstraint("prediction IN (0, 1)", name="ck_prediction_events_binary"),
        sa.PrimaryKeyConstraint("request_id"),
    )
    op.create_index(
        "ix_prediction_events_timestamp",
        "prediction_events",
        ["timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_prediction_events_model_version",
        "prediction_events",
        ["model_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_prediction_events_model_version", table_name="prediction_events")
    op.drop_index("ix_prediction_events_timestamp", table_name="prediction_events")
    op.drop_table("prediction_events")
