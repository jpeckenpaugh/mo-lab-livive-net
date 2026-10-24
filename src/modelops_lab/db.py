"""PostgreSQL schema and Alembic migration entry point for lifecycle state.

The active-model pointer, its append-only transition log, every promotion
decision, and best-effort prediction events live in PostgreSQL (ADR-003).
MLflow owns artifacts and run metadata;
the registry alias marks approved artifacts, but the pointer row is authoritative
for what the lifecycle considers active (ADR-006).

Unit tests use SQLite through :class:`PointerStore`; the advisory lock is a
no-op there and the expected-version check still guards the transaction.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"
MIGRATIONS_DIR = REPO_ROOT / "migrations"

# Arbitrary but stable application-wide key for pg_advisory_xact_lock.
ADVISORY_LOCK_KEY = 0x4D4F4C50  # "MOLP"


class DatabaseError(RuntimeError):
    """Raised when the database cannot be reached or migrated."""


class Base(DeclarativeBase):
    """Declarative base shared by the models and the migration metadata."""


class ActiveModelRow(Base):
    """Singleton pointer to the model version the lifecycle treats as active."""

    __tablename__ = "active_model"
    __table_args__ = (CheckConstraint("id = 1", name="ck_active_model_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    mlflow_run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    mlflow_model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    decision_id: Mapped[str] = mapped_column(String(128), nullable=False)
    promoted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ModelTransitionRow(Base):
    """Append-only history of bootstrap, promotion, and rollback operations."""

    __tablename__ = "model_transitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    from_model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    to_model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    decision_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actor: Mapped[str] = mapped_column(String(255), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PredictionEventRow(Base):
    """One served prediction, retained for monitoring and audit (ADR-003).

    Features are JSONB in PostgreSQL so W5 can query categorical levels and
    numeric values without a schema change. SQLite uses JSON for the unit tier.
    """

    __tablename__ = "prediction_events"
    __table_args__ = (CheckConstraint("prediction IN (0, 1)", name="ck_prediction_events_binary"),)

    # Request ids are client-supplied and deliberately not unique: a retry must
    # still produce an event. The surrogate key is the stable identity.
    event_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    request_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    features: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False
    )
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)


class PredictionLabelRow(Base):
    """Delayed ground truth for one prediction event (W5).

    ``event_id`` is both the primary key and a cascading foreign key, so a
    label can never outlive the event it describes. Relabeling is an upsert:
    the newest label wins and ``labeled_at`` records when it was supplied.
    """

    __tablename__ = "prediction_labels"
    __table_args__ = (CheckConstraint("label IN (0, 1)", name="ck_prediction_labels_binary"),)

    event_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("prediction_events.event_id", ondelete="CASCADE"),
        primary_key=True,
    )
    label: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    labeled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class PromotionDecisionRow(Base):
    """Auditable record of one gate evaluation and the verdict it produced."""

    __tablename__ = "promotion_decisions"

    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    candidate_run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    active_model_version_before: Mapped[str | None] = mapped_column(String(64), nullable=True)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    checks: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    rejection_reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    actor: Mapped[str] = mapped_column(String(255), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    evaluation_split: Mapped[str] = mapped_column(String(32), nullable=False)


def create_engine_for(database_url: str) -> Engine:
    """Build an engine that works for Postgres and the SQLite test tier."""
    if not database_url.strip():
        raise DatabaseError("database URL must not be empty")
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    try:
        return create_engine(
            database_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
    except (ValueError, TypeError) as exc:
        raise DatabaseError(f"invalid database URL: {exc}") from exc


def utc(value: datetime) -> datetime:
    """Return ``value`` as an aware UTC datetime (SQLite drops tzinfo)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def upgrade_database(database_url: str) -> None:
    """Apply every Alembic migration up to ``head``."""
    from alembic import command
    from alembic.config import Config

    if not ALEMBIC_INI.exists():  # pragma: no cover - repository invariant
        raise DatabaseError(f"alembic.ini not found at {ALEMBIC_INI}")
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    # ``%`` is an interpolation character in Alembic option values.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    try:
        command.upgrade(config, "head")
    except Exception as exc:  # alembic raises its own util.CommandError
        raise DatabaseError(f"cannot upgrade database schema: {exc}") from exc


def _session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


__all__ = [
    "ADVISORY_LOCK_KEY",
    "ALEMBIC_INI",
    "ActiveModelRow",
    "Base",
    "DatabaseError",
    "ModelTransitionRow",
    "MIGRATIONS_DIR",
    "PredictionEventRow",
    "PredictionLabelRow",
    "PromotionDecisionRow",
    "create_engine_for",
    "upgrade_database",
    "utc",
]
