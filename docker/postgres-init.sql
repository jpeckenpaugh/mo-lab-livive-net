-- Integration tests truncate lifecycle tables between cases. They must never
-- run against the demo database, so a fresh Postgres volume creates a separate
-- `modelops_test` database owned by the configured (current) user.
SELECT format('CREATE DATABASE modelops_test OWNER %I', current_user)
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'modelops_test')\gexec
