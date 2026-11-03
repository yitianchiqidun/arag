"""Prepare the local cluster for the agent: the sample database and a read-only account.

The read-only account ``agent_ro`` gets SELECT on the sample database and PROCESS globally
(the diagnosis tables ``data_lock_waits`` and ``cluster_tidb_trx`` need it to show other
sessions' transactions). It gets no privilege that allows writing, so the agent's read-only
tools cannot change anything even if the statement check were bypassed;
:func:`verify_readonly` proves it on the live cluster.
"""

from __future__ import annotations

import os

from research_agent.tidb.db import (
    Database,
    DatabaseError,
    TiDBConfig,
    admin_database,
    readonly_database,
)
from research_agent.tidb.readonly import check_identifier
from research_agent.tidb.sample_data import DataSize, load

# Statements the read-only account must not be able to run.
WRITE_PROBES = [
    "CREATE TABLE {db}.ro_probe (id INT PRIMARY KEY)",
    "INSERT INTO {db}.users (id, name, phone, city, created_at) VALUES (-1, 'p', '0', 'p', NOW())",
    "UPDATE {db}.orders SET status = 'probe' WHERE id = 1",
    "DELETE FROM {db}.order_items WHERE id = 1",
    "SET GLOBAL tidb_mem_quota_query = 1073741825",
    "ANALYZE TABLE {db}.orders",
    "CREATE INDEX ro_probe_idx ON {db}.orders (status)",
    "DROP TABLE {db}.products",
]


def readonly_grants(cfg: TiDBConfig) -> list[str]:
    db = check_identifier(cfg.database)
    user = f"'{check_identifier(cfg.readonly_user)}'@'%'"
    return [f"GRANT SELECT ON {db}.* TO {user}", f"GRANT PROCESS ON *.* TO {user}"]


def create_readonly_user(admin: Database, cfg: TiDBConfig, password: str) -> None:
    user = f"'{check_identifier(cfg.readonly_user)}'@'%'"
    admin.execute(f"CREATE USER IF NOT EXISTS {user} IDENTIFIED BY %s", [password])
    admin.execute(f"ALTER USER {user} IDENTIFIED BY %s", [password])
    for grant in readonly_grants(cfg):
        admin.execute(grant)


def setup(cfg: TiDBConfig, size: DataSize | None = None, seed: int = 2026) -> dict[str, int]:
    """Create the database, load the sample data if the database is new, and create the
    read-only account (password from ``cfg.readonly_password_env``)."""
    password = os.environ.get(cfg.readonly_password_env)
    if not password:
        raise RuntimeError(f"set {cfg.readonly_password_env} in .env first")
    server = admin_database(cfg, database="")
    db = check_identifier(cfg.database)
    exists = server.query("SHOW DATABASES LIKE %s", [cfg.database]).rows
    server.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
    counts: dict[str, int] = {}
    if not exists:
        counts = load(admin_database(cfg), size, seed)
    create_readonly_user(server, cfg, password)
    return counts


def verify_readonly(cfg: TiDBConfig) -> dict[str, str | None]:
    """Try every write probe with the read-only account; statement -> the error, or None
    if the statement was accepted (which would be a failure)."""
    ro = readonly_database(cfg)
    out: dict[str, str | None] = {}
    for probe in WRITE_PROBES:
        sql = probe.format(db=cfg.database)
        try:
            ro.execute(sql)
            out[sql] = None
        except DatabaseError as e:
            out[sql] = str(e)
    return out
