"""Alembic environment for the ModelOps Lab promotion-state schema."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from modelops_lab.db import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    configured = (config.get_main_option("sqlalchemy.url") or "").strip()
    if configured:
        return configured
    environment = os.environ.get("MODELOPS_DATABASE_URL", "").strip()
    if environment:
        return environment
    raise RuntimeError(
        "no database URL: pass --database-url, set MODELOPS_DATABASE_URL, "
        "or set sqlalchemy.url in alembic.ini"
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        version_table="modelops_alembic_version",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_database_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            version_table="modelops_alembic_version",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
