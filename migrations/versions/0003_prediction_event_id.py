"""Give prediction events a surrogate identity and make request ids non-unique.

Revision ID: 0003_prediction_event_id
Revises: 0002_prediction_events
Create Date: 2026-10-24

A client may legitimately retry with the same ``X-Request-ID``. The primary key
must not turn that into a lost event, so ``event_id`` becomes the identity and
``request_id`` stays as an indexed, non-unique correlation id.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_prediction_event_id"
down_revision: str | None = "0002_prediction_events"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EVENT_COLUMNS = (
    "request_id",
    "timestamp",
    "features",
    "prediction",
    "probability",
    "model_version",
    "latency_ms",
)


def _postgres_upgrade() -> None:
    # A plain BIGINT with an owned sequence is PostgreSQL's serial/auto-increment
    # shape and matches the SQLAlchemy model's sequence name. Existing rows are
    # backfilled before NOT NULL is applied.
    op.add_column("prediction_events", sa.Column("event_id", sa.BigInteger(), nullable=True))
    op.execute(
        "CREATE SEQUENCE IF NOT EXISTS prediction_events_event_id_seq "
        "OWNED BY prediction_events.event_id"
    )
    op.execute(
        "ALTER TABLE prediction_events ALTER COLUMN event_id "
        "SET DEFAULT nextval('prediction_events_event_id_seq')"
    )
    op.execute(
        "UPDATE prediction_events SET event_id = nextval('prediction_events_event_id_seq') "
        "WHERE event_id IS NULL"
    )
    op.execute("ALTER TABLE prediction_events ALTER COLUMN event_id SET NOT NULL")

    primary_key = sa.inspect(op.get_bind()).get_pk_constraint("prediction_events")["name"]
    if primary_key:
        op.drop_constraint(primary_key, "prediction_events", type_="primary")
    op.create_primary_key("pk_prediction_events", "prediction_events", ["event_id"])
    op.create_index(
        "ix_prediction_events_request_id",
        "prediction_events",
        ["request_id"],
        unique=False,
    )


def _sqlite_upgrade() -> None:
    # SQLite cannot add an autoincrement primary key in place; rebuild and copy.
    op.create_table(
        "prediction_events_new",
        sa.Column("event_id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=200), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.CheckConstraint("prediction IN (0, 1)", name="ck_prediction_events_binary"),
    )
    columns = ", ".join(EVENT_COLUMNS)
    op.execute(
        f"INSERT INTO prediction_events_new ({columns}) "
        f"SELECT {columns} FROM prediction_events ORDER BY timestamp"
    )
    op.drop_table("prediction_events")
    op.rename_table("prediction_events_new", "prediction_events")
    op.create_index("ix_prediction_events_request_id", "prediction_events", ["request_id"])
    op.create_index("ix_prediction_events_timestamp", "prediction_events", ["timestamp"])
    op.create_index("ix_prediction_events_model_version", "prediction_events", ["model_version"])


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        _postgres_upgrade()
    else:
        _sqlite_upgrade()


def _postgres_downgrade() -> None:
    op.drop_index("ix_prediction_events_request_id", table_name="prediction_events")
    op.drop_constraint("pk_prediction_events", "prediction_events", type_="primary")
    op.drop_column("prediction_events", "event_id")
    op.create_primary_key("prediction_events_pkey", "prediction_events", ["request_id"])


def _sqlite_downgrade() -> None:
    op.create_table(
        "prediction_events_old",
        sa.Column("request_id", sa.String(length=200), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.CheckConstraint("prediction IN (0, 1)", name="ck_prediction_events_binary"),
    )
    columns = ", ".join(EVENT_COLUMNS)
    op.execute(
        f"INSERT INTO prediction_events_old ({columns}) "
        f"SELECT {columns} FROM prediction_events ORDER BY event_id"
    )
    op.drop_table("prediction_events")
    op.rename_table("prediction_events_old", "prediction_events")
    op.create_index("ix_prediction_events_timestamp", "prediction_events", ["timestamp"])
    op.create_index("ix_prediction_events_model_version", "prediction_events", ["model_version"])


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        _postgres_downgrade()
    else:
        _sqlite_downgrade()
