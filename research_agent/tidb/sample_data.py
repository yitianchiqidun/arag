"""A small e-commerce database for the fault scenarios (stage 8), generated with a fixed seed.

Four tables, about 100,000 rows: users 10,000, products 2,000, orders 30,000, order_items
about 60,000. Some indexes are left out on purpose, as the scenarios need them missing:
``orders`` has no index on ``status`` or ``created_at`` (a missing index slows down a
frequent filter), and ``users.phone`` is a string column with an index (comparing it with a
number makes the index unusable). Everything is deterministic for a given seed.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel

from research_agent.tidb.db import Database

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
    id BIGINT PRIMARY KEY,
    name VARCHAR(64) NOT NULL,
    phone VARCHAR(20) NOT NULL,
    city VARCHAR(32) NOT NULL,
    created_at DATETIME NOT NULL,
    KEY idx_phone (phone)
)""",
    """CREATE TABLE IF NOT EXISTS products (
    id BIGINT PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    category VARCHAR(32) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    stock INT NOT NULL
)""",
    """CREATE TABLE IF NOT EXISTS orders (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    status VARCHAR(16) NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    created_at DATETIME NOT NULL,
    KEY idx_user (user_id)
)""",
    """CREATE TABLE IF NOT EXISTS order_items (
    id BIGINT PRIMARY KEY,
    order_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    quantity INT NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    KEY idx_order (order_id)
)""",
]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "西安", "苏州"]
CATEGORIES = ["手机", "电脑", "家电", "服饰", "图书", "食品", "美妆", "运动", "母婴", "家居"]
STATUSES = ["paid", "shipped", "delivered", "cancelled", "refunded"]
STATUS_WEIGHTS = [0.15, 0.2, 0.55, 0.07, 0.03]
START = datetime(2026, 1, 1)


class DataSize(BaseModel):
    users: int = 10_000
    products: int = 2_000
    orders: int = 30_000
    max_items_per_order: int = 3  # 1..max, about 2 on average


def generate(size: DataSize, seed: int = 2026) -> dict[str, list[tuple[Any, ...]]]:
    """Rows per table, in insertion order."""
    rng = random.Random(seed)
    users = [
        (
            i,
            f"用户{i:05d}",
            f"1{rng.randint(30, 99)}{rng.randint(0, 99_999_999):08d}",
            rng.choice(CITIES),
            START + timedelta(minutes=rng.randint(0, 300 * 24 * 60)),
        )
        for i in range(1, size.users + 1)
    ]
    products = [
        (
            i,
            f"商品{i:04d}",
            rng.choice(CATEGORIES),
            round(rng.uniform(5, 5000), 2),
            rng.randint(0, 1000),
        )
        for i in range(1, size.products + 1)
    ]
    prices = {p[0]: p[3] for p in products}
    orders: list[tuple[Any, ...]] = []
    items: list[tuple[Any, ...]] = []
    for i in range(1, size.orders + 1):
        lines = []
        for _ in range(rng.randint(1, size.max_items_per_order)):
            product = rng.randint(1, size.products)
            lines.append((product, rng.randint(1, 5), prices[product]))
        amount = round(sum(q * p for _, q, p in lines), 2)
        created = START + timedelta(minutes=rng.randint(0, 300 * 24 * 60))
        status = rng.choices(STATUSES, STATUS_WEIGHTS)[0]
        orders.append((i, rng.randint(1, size.users), status, amount, created))
        for product, quantity, price in lines:
            items.append((len(items) + 1, i, product, quantity, price))
    return {"users": users, "products": products, "orders": orders, "order_items": items}


COLUMNS = {
    "users": "(id, name, phone, city, created_at)",
    "products": "(id, name, category, price, stock)",
    "orders": "(id, user_id, status, amount, created_at)",
    "order_items": "(id, order_id, product_id, quantity, price)",
}


def _batches(rows: list[tuple[Any, ...]], size: int) -> Iterator[list[tuple[Any, ...]]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def insert_statements(
    tables: dict[str, list[tuple[Any, ...]]], batch: int = 1000
) -> Iterator[tuple[str, list[Any]]]:
    """Multi-row INSERT statements with their parameters."""
    for table, rows in tables.items():
        for chunk in _batches(rows, batch):
            width = len(chunk[0])
            values = ", ".join(["(" + ", ".join(["%s"] * width) + ")"] * len(chunk))
            yield (
                f"INSERT INTO {table} {COLUMNS[table]} VALUES {values}",
                [v for row in chunk for v in row],
            )


def load(db: Database, size: DataSize | None = None, seed: int = 2026) -> dict[str, int]:
    """Create the tables, insert the rows (the tables must be empty) and collect statistics;
    returns rows per table."""
    tables = generate(size or DataSize(), seed)
    for ddl in SCHEMA:
        db.execute(ddl)
    for sql, params in insert_statements(tables):
        db.execute(sql, params)
    for table in tables:
        db.execute(f"ANALYZE TABLE {table}")
    return {t: len(rows) for t, rows in tables.items()}
