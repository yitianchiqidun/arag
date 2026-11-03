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


# Error codes of a statement refused for lack of privileges (database, table, column,
# global/dynamic).
# Anything else, e.g. a failed login (1045), proves nothing about what the account may do.
PRIVILEGE_ERRORS = frozenset({1044, 1142, 1143, 1227, 8121})


def readonly_grants(cfg: TiDBConfig) -> list[str]:
    db = check_identifier(cfg.database)
    user = f"'{check_identifier(cfg.readonly_user)}'@'%'"
    return [
        f"GRANT SELECT ON {db}.* TO {user}",
        f"GRANT PROCESS ON *.* TO {user}",
        # SHOW STATS_META reads it; the rest of the mysql schema (users, password hashes) stays
        # out of reach, so SHOW STATS_HEALTHY is computed instead (ReadOnlyTools.table_stats).
        f"GRANT SELECT ON mysql.stats_meta TO {user}",
    ]


def create_readonly_user(admin: Database, cfg: TiDBConfig, password: str) -> None:
    # User and host are parameters too: the host wildcard '%' in the statement text would be
    # read as a format character by the driver's client-side parameter substitution.
    user = [check_identifier(cfg.readonly_user), "%"]
    admin.execute("CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s", [*user, password])
    admin.execute("ALTER USER %s@%s IDENTIFIED BY %s", [*user, password])
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
    """Try every write probe with the read-only account; statement -> the privilege error
    that refused it, or None if it was not refused for lack of privileges (a failure).

    Raises if the account cannot log in or read: a refused login would make every probe fail
    without saying anything about the privileges.
    """
    ro = readonly_database(cfg)
    try:
        ro.query(f"SELECT COUNT(*) FROM {check_identifier(cfg.database)}.orders")
    except DatabaseError as e:
        raise RuntimeError(f"the read-only account cannot log in and read: {e}") from e
    out: dict[str, str | None] = {}
    for probe in WRITE_PROBES:
        sql = probe.format(db=cfg.database)
        try:
            ro.execute(sql)
            out[sql] = None
        except DatabaseError as e:
            out[sql] = str(e) if e.code in PRIVILEGE_ERRORS else None
    return out
