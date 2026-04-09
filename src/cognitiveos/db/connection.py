from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import sqlite_vec


DEFAULT_SQLITE_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 5000
DEFAULT_SQLITE_JOURNAL_MODE = "WAL"
DEFAULT_SQLITE_SYNCHRONOUS = "NORMAL"
DEFAULT_SQLITE_TEMP_STORE = "MEMORY"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    cleaned = raw.strip()
    return cleaned or default


def open_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        db_path,
        timeout=_env_float(
            "COGNITIVEOS_SQLITE_TIMEOUT_SECONDS",
            DEFAULT_SQLITE_CONNECT_TIMEOUT_SECONDS,
        ),
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.execute(
        f"PRAGMA busy_timeout = {_env_int('COGNITIVEOS_SQLITE_BUSY_TIMEOUT_MS', DEFAULT_SQLITE_BUSY_TIMEOUT_MS)};"
    )
    connection.execute(
        f"PRAGMA journal_mode = {_env_str('COGNITIVEOS_SQLITE_JOURNAL_MODE', DEFAULT_SQLITE_JOURNAL_MODE)};"
    )
    connection.execute(
        f"PRAGMA synchronous = {_env_str('COGNITIVEOS_SQLITE_SYNCHRONOUS', DEFAULT_SQLITE_SYNCHRONOUS)};"
    )
    connection.execute(
        f"PRAGMA temp_store = {_env_str('COGNITIVEOS_SQLITE_TEMP_STORE', DEFAULT_SQLITE_TEMP_STORE)};"
    )
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection
