from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_events (
    prediction_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    batch_id TEXT,
    claim_reference TEXT,
    model_version TEXT NOT NULL,
    fraud_probability REAL NOT NULL,
    decision TEXT NOT NULL,
    input_json TEXT NOT NULL,
    audit_json TEXT,
    explanation_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_prediction_created_at ON prediction_events(created_at);
CREATE INDEX IF NOT EXISTS idx_prediction_batch_id ON prediction_events(batch_id);
CREATE TABLE IF NOT EXISTS evidence_files (
    evidence_id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    mime_type TEXT,
    content BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(prediction_id) REFERENCES prediction_events(prediction_id)
);
"""


def connect(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection


def save_prediction(
    database_path: str | Path,
    *,
    source: str,
    model_version: str,
    probability: float,
    decision: str,
    inputs: dict[str, Any],
    audit: dict[str, Any] | None = None,
    explanation: list[dict[str, Any]] | None = None,
    claim_reference: str | None = None,
    batch_id: str | None = None,
    evidence: Iterable[tuple[str, str | None, bytes]] = (),
) -> str:
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with connect(database_path) as connection:
        connection.execute(
            """INSERT INTO prediction_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                created_at,
                source,
                batch_id,
                claim_reference,
                model_version,
                float(probability),
                decision,
                json.dumps(inputs, default=str),
                json.dumps(audit, default=str) if audit else None,
                json.dumps(explanation, default=str) if explanation else None,
            ),
        )
        for filename, mime_type, content in evidence:
            connection.execute(
                "INSERT INTO evidence_files VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), prediction_id, filename, mime_type, content, created_at),
            )
    return prediction_id


def load_history(database_path: str | Path, limit: int = 1000) -> pd.DataFrame:
    with connect(database_path) as connection:
        return pd.read_sql_query(
            """SELECT prediction_id, created_at, source, batch_id, claim_reference,
                      model_version, fraud_probability, decision
               FROM prediction_events ORDER BY created_at DESC LIMIT ?""",
            connection,
            params=(int(limit),),
        )


def save_batch_predictions(
    database_path: str | Path,
    *,
    model_version: str,
    rows: pd.DataFrame,
    input_columns: list[str],
    audit_columns: list[str],
    id_column: str = "PolicyNumber",
) -> str:
    batch_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    records = []
    for _, row in rows.iterrows():
        inputs = {column: row.get(column) for column in input_columns}
        audit = {column: row.get(column) for column in audit_columns if column in rows.columns}
        records.append(
            (
                str(uuid.uuid4()),
                created_at,
                "batch",
                batch_id,
                str(row.get(id_column, "")) or None,
                model_version,
                float(row["fraud_probability"]),
                str(row["decision"]),
                json.dumps(inputs, default=str),
                json.dumps(audit, default=str) if audit else None,
                None,
            )
        )
    with connect(database_path) as connection:
        connection.executemany(
            "INSERT INTO prediction_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )
    return batch_id
