"""Delayed ground-truth labels for prediction events.

Revision ID: 0004_prediction_labels
Revises: 0003_prediction_event_id
Create Date: 2026-10-24

``event_id`` is the primary key and a cascading foreign key: a label can never
outlive its event. Relabeling is an application-level upsert. The
``labeled_at`` index supports retention queries.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_prediction_labels"
down_revision: str | None = "0003_prediction_event_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prediction_labels",
        sa.Column("event_id", sa.BigInteger(), nullable=False),
        sa.Column("label", sa.SmallInteger(), nullable=False),
        sa.Column("labeled_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("label IN (0, 1)", name="ck_prediction_labels_binary"),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["prediction_events.event_id"],
            name="fk_prediction_labels_event_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index(
        "ix_prediction_labels_labeled_at",
        "prediction_labels",
        ["labeled_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_prediction_labels_labeled_at", table_name="prediction_labels")
    op.drop_table("prediction_labels")
