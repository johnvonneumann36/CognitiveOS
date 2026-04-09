from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import sqlite_vec

from cognitiveos.db.connection import open_connection
from cognitiveos.db.schema import SCHEMA_SQL
from cognitiveos.exceptions import NodeNotFoundError
from cognitiveos.metadata_shapes import (
    normalize_edge_metadata,
    normalize_node_metadata,
)
from cognitiveos.models import EdgeRecord, LinkedNode, NodeRecord, ReadNodeResult, SearchResult


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _json_loads(payload: str | None, default: Any) -> Any:
    if not payload:
        return default
    return json.loads(payload)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _serialize_embedding(embedding: list[float] | None) -> bytes | None:
    if embedding is None:
        return None
    return json.dumps(embedding, separators=(",", ":")).encode("utf-8")


def _deserialize_embedding(payload: bytes | str | None) -> list[float] | None:
    if payload is None:
        return None
    if isinstance(payload, bytes):
        return json.loads(payload.decode("utf-8"))
    return json.loads(payload)


class SQLiteRepository:
    VECTOR_TABLE_NAME = "node_embeddings_vec"
    STATE_KEY_EMBEDDING_DIMENSION = "embedding_dimension"

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        with open_connection(self.db_path) as connection:
            connection.executescript(SCHEMA_SQL)
            self._run_schema_migrations(connection)
            self._rebuild_search_index(connection)
            dimension = self.get_embedding_dimension(connection=connection)
            if dimension is not None:
                self._ensure_vector_table(connection, dimension)
                self._rebuild_vector_index(connection, dimension=dimension)

    def _run_schema_migrations(self, connection: sqlite3.Connection) -> None:
        self._migrate_nodes_description_limit(connection)
        self._migrate_nodes_content_limit(connection)
        self._migrate_nodes_remove_stability_score(connection)
        self._migrate_edges_remove_weight(connection)
        self._ensure_column(connection, "nodes", "node_type", "TEXT NOT NULL DEFAULT 'memory'")
        self._ensure_column(connection, "nodes", "durability", "TEXT NOT NULL DEFAULT 'working'")
        self._ensure_column(connection, "nodes", "last_reinforced_at", "DATETIME")
        self._ensure_column(connection, "edges", "strength_score", "REAL NOT NULL DEFAULT 1.0")
        self._ensure_column(connection, "edges", "durability", "TEXT NOT NULL DEFAULT 'durable'")
        self._ensure_column(connection, "edges", "status", "TEXT NOT NULL DEFAULT 'active'")
        self._ensure_column(connection, "edges", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "edges", "last_reinforced_at", "DATETIME")
        connection.execute(
            """
            UPDATE nodes
            SET node_type = COALESCE(NULLIF(node_type, ''), 'memory')
            """
        )
        connection.execute(
            """
            UPDATE nodes
            SET durability = COALESCE(NULLIF(durability, ''), 'working')
            """
        )
        connection.execute(
            """
            UPDATE edges
            SET status = COALESCE(NULLIF(status, ''), 'active'),
                durability = COALESCE(NULLIF(durability, ''), 'durable'),
                strength_score = CASE
                    WHEN strength_score IS NULL OR strength_score <= 0 THEN 1.0
                    ELSE strength_score
                END,
                metadata_json = COALESCE(NULLIF(metadata_json, ''), '{}')
            """
        )

    @staticmethod
    def _migrate_nodes_description_limit(connection: sqlite3.Connection) -> None:
        row = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = 'nodes'
            """
        ).fetchone()
        create_sql = (row["sql"] or "") if row else ""
        if "LENGTH(description) <= 500" not in create_sql:
            return

        connection.executescript(
            """
            ALTER TABLE nodes RENAME TO nodes_legacy_description_limit;

            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT NOT NULL CHECK(LENGTH(description) <= 800),
                content TEXT NOT NULL CHECK(LENGTH(content) <= 65535),
                embedding BLOB,
                tags_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                node_type TEXT NOT NULL DEFAULT 'memory',
                durability TEXT NOT NULL DEFAULT 'working',
                last_reinforced_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            INSERT INTO nodes (
                id,
                name,
                description,
                content,
                embedding,
                tags_json,
                metadata_json,
                node_type,
                durability,
                last_reinforced_at,
                created_at,
                updated_at
            )
            SELECT
                id,
                name,
                description,
                content,
                embedding,
                tags_json,
                metadata_json,
                node_type,
                durability,
                last_reinforced_at,
                created_at,
                updated_at
            FROM nodes_legacy_description_limit;

            DROP TABLE nodes_legacy_description_limit;
            """
        )

    @staticmethod
    def _migrate_nodes_content_limit(connection: sqlite3.Connection) -> None:
        row = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = 'nodes'
            """
        ).fetchone()
        create_sql = (row["sql"] or "") if row else ""
        if "LENGTH(content) <= 12800" not in create_sql:
            return

        connection.executescript(
            """
            ALTER TABLE nodes RENAME TO nodes_legacy_content_limit;

            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT NOT NULL CHECK(LENGTH(description) <= 800),
                content TEXT NOT NULL CHECK(LENGTH(content) <= 65535),
                embedding BLOB,
                tags_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                node_type TEXT NOT NULL DEFAULT 'memory',
                durability TEXT NOT NULL DEFAULT 'working',
                last_reinforced_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            INSERT INTO nodes (
                id,
                name,
                description,
                content,
                embedding,
                tags_json,
                metadata_json,
                node_type,
                durability,
                last_reinforced_at,
                created_at,
                updated_at
            )
            SELECT
                id,
                name,
                description,
                content,
                embedding,
                tags_json,
                metadata_json,
                node_type,
                durability,
                last_reinforced_at,
                created_at,
                updated_at
            FROM nodes_legacy_content_limit;

            DROP TABLE nodes_legacy_content_limit;
            """
        )

    def _migrate_nodes_remove_stability_score(self, connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "stability_score" not in columns:
            return
        connection.executescript(
            """
            ALTER TABLE nodes RENAME TO nodes_legacy_stability_score;

            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT NOT NULL CHECK(LENGTH(description) <= 800),
                content TEXT NOT NULL CHECK(LENGTH(content) <= 65535),
                embedding BLOB,
                tags_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                node_type TEXT NOT NULL DEFAULT 'memory',
                durability TEXT NOT NULL DEFAULT 'working',
                last_reinforced_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            INSERT INTO nodes (
                id,
                name,
                description,
                content,
                embedding,
                tags_json,
                metadata_json,
                node_type,
                durability,
                last_reinforced_at,
                created_at,
                updated_at
            )
            SELECT
                id,
                name,
                description,
                content,
                embedding,
                tags_json,
                metadata_json,
                node_type,
                durability,
                last_reinforced_at,
                created_at,
                updated_at
            FROM nodes_legacy_stability_score;

            DROP TABLE nodes_legacy_stability_score;
            """
        )

    def _migrate_edges_remove_weight(self, connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(edges)").fetchall()
        }
        if "weight" not in columns:
            return
        connection.executescript(
            """
            ALTER TABLE edges RENAME TO edges_legacy_weight;

            CREATE TABLE edges (
                src_id TEXT NOT NULL,
                dst_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                strength_score REAL NOT NULL DEFAULT 1.0,
                durability TEXT NOT NULL DEFAULT 'durable',
                status TEXT NOT NULL DEFAULT 'active',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_reinforced_at DATETIME,
                PRIMARY KEY (src_id, dst_id, relation),
                FOREIGN KEY (src_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (dst_id) REFERENCES nodes(id) ON DELETE CASCADE
            );

            INSERT INTO edges (
                src_id,
                dst_id,
                relation,
                strength_score,
                durability,
                status,
                metadata_json,
                created_at,
                last_reinforced_at
            )
            SELECT
                src_id,
                dst_id,
                relation,
                CASE
                    WHEN strength_score IS NULL OR strength_score <= 0 THEN 1.0
                    ELSE strength_score
                END,
                COALESCE(NULLIF(durability, ''), 'durable'),
                COALESCE(NULLIF(status, ''), 'active'),
                COALESCE(NULLIF(metadata_json, ''), '{}'),
                created_at,
                last_reinforced_at
            FROM edges_legacy_weight;

            DROP TABLE edges_legacy_weight;

            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
            """
        )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        definition: str,
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in columns:
            return
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _rebuild_search_index(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM nodes_fts")
        rows = connection.execute("SELECT * FROM nodes").fetchall()
        for row in rows:
            self._sync_node_fts_from_row(connection, row)

    def _rebuild_vector_index(self, connection: sqlite3.Connection, *, dimension: int) -> None:
        connection.execute(f"DELETE FROM {self.VECTOR_TABLE_NAME}")
        rows = connection.execute(
            "SELECT rowid, embedding FROM nodes WHERE embedding IS NOT NULL"
        ).fetchall()
        for row in rows:
            embedding = _deserialize_embedding(row["embedding"])
            if embedding is None or len(embedding) != dimension:
                continue
            connection.execute(
                f"""
                INSERT INTO {self.VECTOR_TABLE_NAME}(rowid, embedding)
                VALUES (?, ?)
                """,
                (int(row["rowid"]), sqlite_vec.serialize_float32(embedding)),
            )

    def _sync_node_fts_from_row(self, connection: sqlite3.Connection, row: sqlite3.Row) -> None:
        search_text = self._build_search_text(
            name=row["name"],
            description=row["description"],
            content=row["content"],
            node_type=row["node_type"],
            tags=_json_loads(row["tags_json"], []),
            metadata=_json_loads(row["metadata_json"], {}),
        )
        connection.execute("DELETE FROM nodes_fts WHERE id = ?", (row["id"],))
        connection.execute(
            "INSERT INTO nodes_fts(id, search_text) VALUES (?, ?)",
            (row["id"], search_text),
        )

    def _sync_node_fts(
        self,
        connection: sqlite3.Connection,
        *,
        node_id: str,
        name: str | None,
        description: str,
        content: str,
        node_type: str,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> None:
        search_text = self._build_search_text(
            name=name,
            description=description,
            content=content,
            node_type=node_type,
            tags=tags,
            metadata=metadata,
        )
        connection.execute("DELETE FROM nodes_fts WHERE id = ?", (node_id,))
        connection.execute(
            "INSERT INTO nodes_fts(id, search_text) VALUES (?, ?)",
            (node_id, search_text),
        )

    def _build_search_text(
        self,
        *,
        name: str | None,
        description: str,
        content: str,
        node_type: str,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> str:
        normalized_metadata = normalize_node_metadata(metadata)
        parts: list[str] = []
        text_candidates = [name, description]
        if node_type != "source_document":
            text_candidates.append(content)
        for candidate in text_candidates:
            if candidate:
                parts.append(candidate)
        parts.extend(tag for tag in tags if tag)
        parts.extend(self._extract_search_metadata_terms(normalized_metadata))
        return "\n".join(parts)

    def _extract_search_metadata_terms(self, metadata: dict[str, Any]) -> list[str]:
        ignored_keys = {
            "hash",
            "content_length",
            "token_estimate",
            "dream_source_node_ids",
            "dream_cluster_size",
            "dream_run_id",
            "dream_task_id",
        }
        ignored_suffixes = ("_id", "_ids", "_at")

        def walk(value: Any, *, key: str | None = None) -> list[str]:
            if key in ignored_keys:
                return []
            if key and key.endswith(ignored_suffixes):
                return []
            if isinstance(value, dict):
                terms: list[str] = []
                for child_key, child_value in value.items():
                    terms.extend(walk(child_value, key=child_key))
                return terms
            if isinstance(value, list):
                terms: list[str] = []
                for item in value:
                    terms.extend(walk(item, key=key))
                return terms
            if isinstance(value, str):
                stripped = value.strip()
                return [stripped] if stripped else []
            if isinstance(value, int | float) and key in {"index", "count"}:
                return [str(value)]
            return []

        return walk(metadata)

    def create_node(self, node: NodeRecord, *, actor: str, action_type: str) -> str:
        audit_log_id = _new_id("log")
        normalized_metadata = normalize_node_metadata(node.metadata)
        with open_connection(self.db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO nodes (
                    id,
                    name,
                    description,
                    content,
                    embedding,
                    tags_json,
                    metadata_json,
                    node_type,
                    durability,
                    last_reinforced_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.name,
                    node.description,
                    node.content,
                    _serialize_embedding(node.embedding),
                    _json_dumps(node.tags),
                    _json_dumps(normalized_metadata),
                    node.node_type,
                    node.durability,
                    node.last_reinforced_at,
                ),
            )
            rowid = int(cursor.lastrowid)
            if node.embedding is not None:
                self._upsert_vector(connection, rowid=rowid, embedding=node.embedding)
            self._sync_node_fts(
                connection,
                node_id=node.id,
                name=node.name,
                description=node.description,
                content=node.content,
                node_type=node.node_type,
                tags=node.tags,
                metadata=normalized_metadata,
            )
            connection.execute(
                """
                INSERT INTO audit_logs (log_id, node_id, action_type, previous_content, actor)
                VALUES (?, ?, ?, NULL, ?)
                """,
                (audit_log_id, node.id, action_type, actor),
            )
            self._record_memory_event(
                connection,
                event_type=f"node_{action_type}",
                ref_id=node.id,
                metadata={"actor": actor},
            )
        return audit_log_id

    def update_node(
        self,
        node_id: str,
        *,
        content: str,
        description: str,
        embedding: list[float] | None,
        tags: list[str] | None,
        durability: str | None,
        actor: str,
    ) -> str:
        existing = self.get_node_record(node_id)
        audit_log_id = _new_id("log")
        next_tags = tags if tags is not None else _json_loads(existing["tags_json"], [])
        next_durability = durability or existing["durability"]
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO audit_logs (log_id, node_id, action_type, previous_content, actor)
                VALUES (?, ?, 'update', ?, ?)
                """,
                (audit_log_id, node_id, existing["content"], actor),
            )
            connection.execute(
                """
                UPDATE nodes
                SET content = ?,
                    description = ?,
                    embedding = ?,
                    tags_json = ?,
                    durability = ?,
                    last_reinforced_at = COALESCE(last_reinforced_at, CURRENT_TIMESTAMP),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    content,
                    description,
                    _serialize_embedding(embedding),
                    _json_dumps(next_tags),
                    next_durability,
                    node_id,
                ),
            )
            if embedding is not None:
                self._upsert_vector(connection, rowid=int(existing["rowid"]), embedding=embedding)
            self._sync_node_fts(
                connection,
                node_id=node_id,
                name=existing["name"],
                description=description,
                content=content,
                node_type=existing["node_type"],
                tags=next_tags,
                metadata=_json_loads(existing["metadata_json"], {}),
            )
            self._record_memory_event(
                connection,
                event_type="node_update",
                ref_id=node_id,
                metadata={"actor": actor},
            )
        return audit_log_id

    def overwrite_node(self, node: NodeRecord, *, actor: str, action_type: str = "update") -> str:
        existing = self.get_node_record(node.id)
        audit_log_id = _new_id("log")
        normalized_metadata = normalize_node_metadata(node.metadata)
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO audit_logs (log_id, node_id, action_type, previous_content, actor)
                VALUES (?, ?, ?, ?, ?)
                """,
                (audit_log_id, node.id, action_type, existing["content"], actor),
            )
            connection.execute(
                """
                UPDATE nodes
                SET name = ?,
                    description = ?,
                    content = ?,
                    embedding = ?,
                    tags_json = ?,
                    metadata_json = ?,
                    node_type = ?,
                    durability = ?,
                    last_reinforced_at = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    node.name,
                    node.description,
                    node.content,
                    _serialize_embedding(node.embedding),
                    _json_dumps(node.tags),
                    _json_dumps(normalized_metadata),
                    node.node_type,
                    node.durability,
                    node.last_reinforced_at,
                    node.id,
                ),
            )
            if node.embedding is not None:
                self._upsert_vector(
                    connection,
                    rowid=int(existing["rowid"]),
                    embedding=node.embedding,
                )
            self._sync_node_fts(
                connection,
                node_id=node.id,
                name=node.name,
                description=node.description,
                content=node.content,
                node_type=node.node_type,
                tags=node.tags,
                metadata=normalized_metadata,
            )
            self._record_memory_event(
                connection,
                event_type=f"node_{action_type}",
                ref_id=node.id,
                metadata={"actor": actor},
            )
        return audit_log_id

    def update_node_embedding(self, node_id: str, embedding: list[float]) -> None:
        existing = self.get_node_record(node_id)
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                UPDATE nodes
                SET embedding = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (_serialize_embedding(embedding), node_id),
            )
            self._upsert_vector(connection, rowid=int(existing["rowid"]), embedding=embedding)

    def record_access(self, node_ids: list[str], *, access_type: str) -> None:
        unique_node_ids = list(dict.fromkeys(node_ids))
        if not unique_node_ids:
            return
        with open_connection(self.db_path) as connection:
            connection.executemany(
                """
                INSERT INTO access_logs (log_id, node_id, access_type)
                VALUES (?, ?, ?)
                """,
                [
                    (_new_id("access"), node_id, access_type)
                    for node_id in unique_node_ids
                ],
            )

    def refresh_semantic_neighbors_for_node(
        self,
        node_id: str,
        *,
        top_k: int,
        exclude_node_types: tuple[str, ...] = (),
    ) -> int:
        with open_connection(self.db_path) as connection:
            connection.execute(
                "DELETE FROM semantic_neighbors WHERE node_id = ?",
                (node_id,),
            )
            if top_k <= 0:
                return 0

            row = connection.execute(
                """
                SELECT rowid, embedding
                FROM nodes
                WHERE id = ?
                """,
                (node_id,),
            ).fetchone()
            if row is None or row["embedding"] is None:
                return 0

            embedding = _deserialize_embedding(row["embedding"])
            if embedding is None:
                return 0

            search_k = max(top_k * 4, 32)
            filters = ["n.id != ?"]
            params: list[Any] = [json.dumps(embedding), search_k, node_id]
            if exclude_node_types:
                placeholders = ", ".join("?" for _ in exclude_node_types)
                filters.append(f"n.node_type NOT IN ({placeholders})")
                params.extend(exclude_node_types)

            rows = connection.execute(
                f"""
                SELECT n.id, {self.VECTOR_TABLE_NAME}.distance AS distance
                FROM {self.VECTOR_TABLE_NAME}
                JOIN nodes n ON n.rowid = {self.VECTOR_TABLE_NAME}.rowid
                WHERE {self.VECTOR_TABLE_NAME}.embedding MATCH ?
                  AND {self.VECTOR_TABLE_NAME}.k = ?
                  AND {' AND '.join(filters)}
                ORDER BY distance
                """,
                params,
            ).fetchall()

            inserts: list[tuple[str, str, float]] = []
            seen: set[str] = set()
            for candidate_row in rows:
                neighbor_id = candidate_row["id"]
                distance = candidate_row["distance"]
                if neighbor_id in seen or distance is None:
                    continue
                similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
                inserts.append((node_id, neighbor_id, similarity))
                seen.add(neighbor_id)
                if len(inserts) >= top_k:
                    break

            if inserts:
                connection.executemany(
                    """
                    INSERT INTO semantic_neighbors (
                        node_id,
                        neighbor_id,
                        similarity
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT(node_id, neighbor_id)
                    DO UPDATE SET similarity = excluded.similarity,
                                  computed_at = CURRENT_TIMESTAMP
                    """,
                    inserts,
                )
            return len(inserts)

    def list_semantic_neighbors(
        self,
        node_ids: list[str],
        *,
        min_similarity: float | None = None,
    ) -> list[tuple[str, str, float]]:
        unique_node_ids = list(dict.fromkeys(node_ids))
        if not unique_node_ids:
            return []
        placeholders = ", ".join("?" for _ in unique_node_ids)
        params: list[Any] = list(unique_node_ids)
        where_clauses = [f"node_id IN ({placeholders})"]
        if min_similarity is not None:
            where_clauses.append("similarity >= ?")
            params.append(min_similarity)
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT node_id, neighbor_id, similarity
                FROM semantic_neighbors
                WHERE {' AND '.join(where_clauses)}
                ORDER BY node_id ASC, similarity DESC, neighbor_id ASC
                """,
                params,
            ).fetchall()
        return [
            (row["node_id"], row["neighbor_id"], float(row["similarity"]))
            for row in rows
        ]

    def create_edge(self, edge: EdgeRecord) -> None:
        with open_connection(self.db_path) as connection:
            self._create_edges(connection, [edge])

    def create_edges(self, edges: list[EdgeRecord]) -> None:
        if not edges:
            return
        with open_connection(self.db_path) as connection:
            self._create_edges(connection, edges)

    def list_edges_for_nodes(self, node_ids: list[str]) -> list[EdgeRecord]:
        if not node_ids:
            return []
        placeholders = ", ".join("?" for _ in node_ids)
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT src_id, dst_id, relation, created_at
                     , strength_score, durability, status, metadata_json, last_reinforced_at
                FROM edges
                WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})
                ORDER BY created_at ASC
                """,
                node_ids + node_ids,
            ).fetchall()
        return [self._row_to_edge_record(row) for row in rows]

    def list_relationships(
        self,
        node_id: str,
        *,
        relation: str | None = None,
        status: str | None = None,
    ) -> list[EdgeRecord]:
        clauses = ["(src_id = ? OR dst_id = ?)"]
        params: list[Any] = [node_id, node_id]
        if relation is not None:
            clauses.append("relation = ?")
            params.append(relation)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where_sql = " AND ".join(clauses)
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT src_id, dst_id, relation, created_at,
                       strength_score, durability, status, metadata_json, last_reinforced_at
                FROM edges
                WHERE {where_sql}
                ORDER BY created_at ASC
                """,
                params,
            ).fetchall()
        return [self._row_to_edge_record(row) for row in rows]

    def get_edge(self, src_id: str, dst_id: str, relation: str) -> EdgeRecord | None:
        with open_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT src_id, dst_id, relation, created_at,
                       strength_score, durability, status, metadata_json, last_reinforced_at
                FROM edges
                WHERE src_id = ? AND dst_id = ? AND relation = ?
                """,
                (src_id, dst_id, relation),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_edge_record(row)

    def delete_edge(self, src_id: str, dst_id: str, *, relation: str | None = None) -> int:
        clauses = ["src_id = ?", "dst_id = ?"]
        params: list[Any] = [src_id, dst_id]
        if relation is not None:
            clauses.append("relation = ?")
            params.append(relation)
        where_sql = " AND ".join(clauses)
        with open_connection(self.db_path) as connection:
            existing = connection.execute(
                f"""
                SELECT src_id, dst_id, relation
                FROM edges
                WHERE {where_sql}
                """,
                params,
            ).fetchall()
            removed = connection.execute(
                f"DELETE FROM edges WHERE {where_sql}",
                params,
            ).rowcount
            for row in existing:
                self._record_memory_event(
                    connection,
                    event_type="edge_removed",
                    ref_id=f"{row['src_id']}:{row['relation']}:{row['dst_id']}",
                    metadata={"src_id": row["src_id"], "dst_id": row["dst_id"]},
                )
        return int(removed)

    def reinforce_edges_between_nodes(
        self,
        node_ids: list[str],
        *,
        delta: float,
        actor: str,
        reason: str,
    ) -> int:
        if delta <= 0 or len(node_ids) < 2:
            return 0
        unique_node_ids = list(dict.fromkeys(node_ids))
        placeholders = ", ".join("?" for _ in unique_node_ids)
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT src_id, dst_id, relation, strength_score
                FROM edges
                WHERE src_id IN ({placeholders})
                  AND dst_id IN ({placeholders})
                  AND status != 'archived'
                """,
                unique_node_ids + unique_node_ids,
            ).fetchall()
            if not rows:
                return 0
            for row in rows:
                next_strength = float(row["strength_score"] or 1.0) + delta
                connection.execute(
                    """
                    UPDATE edges
                    SET strength_score = ?,
                        status = 'active',
                        last_reinforced_at = CURRENT_TIMESTAMP
                    WHERE src_id = ? AND dst_id = ? AND relation = ?
                    """,
                    (next_strength, row["src_id"], row["dst_id"], row["relation"]),
                )
                self._record_memory_event(
                    connection,
                    event_type="edge_reinforced",
                    ref_id=f"{row['src_id']}:{row['relation']}:{row['dst_id']}",
                    metadata={
                        "delta": delta,
                        "reason": reason,
                        "actor": actor,
                        "strength_score": next_strength,
                    },
                )
        return len(rows)

    def reinforce_edge(
        self,
        src_id: str,
        dst_id: str,
        relation: str,
        *,
        delta: float,
    ) -> EdgeRecord:
        with open_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT src_id, dst_id, relation, created_at,
                       strength_score, durability, status, metadata_json, last_reinforced_at
                FROM edges
                WHERE src_id = ? AND dst_id = ? AND relation = ?
                """,
                (src_id, dst_id, relation),
            ).fetchone()
            if row is None:
                raise NodeNotFoundError(
                    f"Relationship '{src_id}:{relation}:{dst_id}' was not found."
                )
            next_strength = max(0.0, float(row["strength_score"] or 1.0) + delta)
            connection.execute(
                """
                UPDATE edges
                SET strength_score = ?,
                    status = 'active',
                    last_reinforced_at = CURRENT_TIMESTAMP
                WHERE src_id = ? AND dst_id = ? AND relation = ?
                """,
                (next_strength, src_id, dst_id, relation),
            )
            self._record_memory_event(
                connection,
                event_type="edge_reinforced",
                ref_id=f"{src_id}:{relation}:{dst_id}",
                metadata={"delta": delta, "strength_score": next_strength},
            )
            updated = connection.execute(
                """
                SELECT src_id, dst_id, relation, created_at,
                       strength_score, durability, status, metadata_json, last_reinforced_at
                FROM edges
                WHERE src_id = ? AND dst_id = ? AND relation = ?
                """,
                (src_id, dst_id, relation),
            ).fetchone()
        return self._row_to_edge_record(updated)

    def transition_relationship_states(
        self,
        *,
        weak_after_hours: int,
        stale_after_hours: int,
        weak_strength_threshold: float,
        stale_strength_threshold: float,
        weak_decay_delta: float,
        stale_decay_delta: float,
        node_id: str | None = None,
    ) -> dict[str, Any]:
        clauses = ["durability != 'pinned'"]
        params: list[Any] = []
        if node_id is not None:
            clauses.append("(src_id = ? OR dst_id = ?)")
            params.extend([node_id, node_id])
        where_sql = " AND ".join(clauses)
        now = datetime.now(UTC)
        transitions: list[dict[str, Any]] = []
        inspected = 0

        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT src_id, dst_id, relation, created_at,
                       strength_score, durability, status, metadata_json, last_reinforced_at
                FROM edges
                WHERE {where_sql}
                ORDER BY created_at ASC
                """,
                params,
            ).fetchall()
            inspected = len(rows)
            for row in rows:
                edge = self._row_to_edge_record(row)
                reference_timestamp = edge.last_reinforced_at or edge.created_at
                age_hours: float | None = None
                if reference_timestamp:
                    reference_dt = datetime.fromisoformat(
                        reference_timestamp.replace(" ", "T")
                    ).replace(tzinfo=UTC)
                    age_hours = (now - reference_dt).total_seconds() / 3600

                next_status = edge.status
                reason = "stable"
                if (
                    age_hours is not None
                    and age_hours >= stale_after_hours
                ) or edge.strength_score <= stale_strength_threshold:
                    next_status = "stale"
                    reason = (
                        "age"
                        if age_hours is not None and age_hours >= stale_after_hours
                        else "strength"
                    )
                elif (
                    age_hours is not None
                    and age_hours >= weak_after_hours
                ) or edge.strength_score <= weak_strength_threshold:
                    next_status = "weak"
                    reason = (
                        "age"
                        if age_hours is not None and age_hours >= weak_after_hours
                        else "strength"
                    )
                else:
                    next_status = "active"

                if next_status == edge.status:
                    continue

                next_strength = edge.strength_score
                if next_status == "weak" and edge.status == "active":
                    next_strength = max(0.0, next_strength - weak_decay_delta)
                elif next_status == "stale" and edge.status != "stale":
                    next_strength = max(0.0, next_strength - stale_decay_delta)

                connection.execute(
                    """
                    UPDATE edges
                    SET status = ?,
                        strength_score = ?
                    WHERE src_id = ? AND dst_id = ? AND relation = ?
                    """,
                    (next_status, next_strength, edge.src_id, edge.dst_id, edge.relation),
                )
                transition = {
                    "src_id": edge.src_id,
                    "dst_id": edge.dst_id,
                    "relation": edge.relation,
                    "from_status": edge.status,
                    "to_status": next_status,
                    "age_hours": round(age_hours, 2) if age_hours is not None else None,
                    "reason": reason,
                    "strength_score": round(next_strength, 4),
                }
                transitions.append(transition)
                self._record_memory_event(
                    connection,
                    event_type="edge_status_changed",
                    ref_id=f"{edge.src_id}:{edge.relation}:{edge.dst_id}",
                    metadata=transition,
                )

        status_counts: dict[str, int] = {"active": 0, "weak": 0, "stale": 0, "archived": 0}
        with open_connection(self.db_path) as connection:
            count_rows = connection.execute(
                f"""
                SELECT status, COUNT(*) AS count
                FROM edges
                WHERE {where_sql}
                GROUP BY status
                """,
                params,
            ).fetchall()
        for row in count_rows:
            status_counts[row["status"]] = int(row["count"])
        return {
            "inspected_count": inspected,
            "transition_count": len(transitions),
            "transitions": transitions,
            "status_counts": status_counts,
        }

    def prune_relationships(
        self,
        *,
        node_id: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        clauses = ["status IN ('weak', 'stale')", "durability != 'pinned'"]
        params: list[Any] = []
        if node_id is not None:
            clauses.append("(src_id = ? OR dst_id = ?)")
            params.extend([node_id, node_id])
        where_sql = " AND ".join(clauses)
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT src_id, dst_id, relation, created_at,
                       strength_score, durability, status, metadata_json, last_reinforced_at
                FROM edges
                WHERE {where_sql}
                ORDER BY created_at ASC
                """,
                params,
            ).fetchall()
            candidates = []
            now = datetime.now(UTC)
            for row in rows:
                edge = self._row_to_edge_record(row)
                reference_timestamp = edge.last_reinforced_at or edge.created_at
                age_hours = None
                if reference_timestamp:
                    reference_dt = datetime.fromisoformat(
                        reference_timestamp.replace(" ", "T")
                    ).replace(tzinfo=UTC)
                    age_hours = (now - reference_dt).total_seconds() / 3600
                candidates.append(
                    {
                        **edge.model_dump(),
                        "age_hours_since_reinforcement": (
                            round(age_hours, 2) if age_hours is not None else None
                        ),
                        "prune_reason": f"status={edge.status}",
                    }
                )
            removed = 0
            if not dry_run and rows:
                removed = connection.execute(
                    f"DELETE FROM edges WHERE {where_sql}",
                    params,
                ).rowcount
                for row in rows:
                    self._record_memory_event(
                        connection,
                        event_type="edge_removed",
                        ref_id=f"{row['src_id']}:{row['relation']}:{row['dst_id']}",
                        metadata={"pruned": True},
                    )
        return {
            "dry_run": dry_run,
            "scope_node_id": node_id,
            "candidate_count": len(candidates),
            "removed_count": removed,
            "candidates": candidates,
        }

    def delete_nodes(self, node_ids: list[str], *, actor: str = "agent") -> int:
        unique_node_ids = list(dict.fromkeys(node_ids))
        if not unique_node_ids:
            return 0
        placeholders = ", ".join("?" for _ in unique_node_ids)
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                f"SELECT rowid, id, name, node_type FROM nodes WHERE id IN ({placeholders})",
                unique_node_ids,
            ).fetchall()
            if not rows:
                return 0
            dimension = self.get_embedding_dimension(connection=connection)
            if dimension is not None:
                connection.executemany(
                    f"DELETE FROM {self.VECTOR_TABLE_NAME} WHERE rowid = ?",
                    [(int(row["rowid"]),) for row in rows],
                )
            for row in rows:
                self._record_memory_event(
                    connection,
                    event_type="node_deleted",
                    ref_id=row["id"],
                    metadata={
                        "actor": actor,
                        "node_type": row["node_type"],
                        "name": row["name"],
                    },
                )
            removed = connection.execute(
                f"DELETE FROM nodes WHERE id IN ({placeholders})",
                unique_node_ids,
            ).rowcount
        return int(removed)

    def set_node_durability(self, node_id: str, durability: str) -> NodeRecord:
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                UPDATE nodes
                SET durability = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (durability, node_id),
            )
            self._record_memory_event(
                connection,
                event_type="node_durability_changed",
                ref_id=node_id,
                metadata={"durability": durability},
            )
        return self.get_node(node_id)

    def set_node_pinned(self, node_id: str, *, pinned: bool) -> NodeRecord:
        event_type = "node_pinned" if pinned else "node_unpinned"
        durability = "pinned" if pinned else "durable"
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                UPDATE nodes
                SET durability = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (durability, node_id),
            )
            self._record_memory_event(
                connection,
                event_type=event_type,
                ref_id=node_id,
                metadata={"durability": durability},
            )
        return self.get_node(node_id)

    def delete_external_edges_for_nodes(self, node_ids: list[str]) -> None:
        if not node_ids:
            return
        placeholders = ", ".join("?" for _ in node_ids)
        with open_connection(self.db_path) as connection:
            connection.execute(
                f"""
                DELETE FROM edges
                WHERE ((src_id IN ({placeholders})) OR (dst_id IN ({placeholders})))
                  AND NOT ((src_id IN ({placeholders})) AND (dst_id IN ({placeholders})))
                """,
                node_ids + node_ids + node_ids + node_ids,
            )

    def get_node_record(self, node_id: str) -> sqlite3.Row:
        with open_connection(self.db_path) as connection:
            row = connection.execute(
                "SELECT rowid, * FROM nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
        if row is None:
            raise NodeNotFoundError(f"Node '{node_id}' was not found.")
        return row

    def get_embedding_dimension(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> int | None:
        if connection is None:
            with open_connection(self.db_path) as managed:
                return self.get_embedding_dimension(connection=managed)
        row = connection.execute(
            "SELECT value FROM app_state WHERE key = ?",
            (self.STATE_KEY_EMBEDDING_DIMENSION,),
        ).fetchone()
        if row is None:
            return None
        return int(row["value"])

    def set_embedding_dimension(self, dimension: int, *, connection: sqlite3.Connection) -> None:
        self.set_app_state_value(
            self.STATE_KEY_EMBEDDING_DIMENSION,
            str(dimension),
            connection=connection,
        )

    def get_app_state_value(
        self,
        key: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> str | None:
        if connection is None:
            with open_connection(self.db_path) as managed:
                return self.get_app_state_value(key, connection=managed)
        row = connection.execute(
            "SELECT value FROM app_state WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def set_app_state_value(
        self,
        key: str,
        value: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if connection is None:
            with open_connection(self.db_path) as managed:
                self.set_app_state_value(key, value, connection=managed)
                return
        connection.execute(
            """
            INSERT INTO app_state (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
            """,
            (key, value),
        )

    def get_app_state_json(
        self,
        key: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> Any:
        payload = self.get_app_state_value(key, connection=connection)
        if payload is None:
            return None
        return json.loads(payload)

    def set_app_state_json(
        self,
        key: str,
        value: Any,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self.set_app_state_value(
            key,
            _json_dumps(value),
            connection=connection,
        )

    def get_node_count(self) -> int:
        with open_connection(self.db_path) as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM nodes").fetchone()
        return int(row["count"])

    def get_vector_count(self) -> int:
        with open_connection(self.db_path) as connection:
            dimension = self.get_embedding_dimension(connection=connection)
            if dimension is None:
                return 0
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM {self.VECTOR_TABLE_NAME}"
            ).fetchone()
        return int(row["count"])

    def get_last_completed_dream_run(self) -> sqlite3.Row | None:
        with open_connection(self.db_path) as connection:
            return connection.execute(
                """
                SELECT *
                FROM dream_runs
                WHERE completed_at IS NOT NULL
                ORDER BY completed_at DESC
                LIMIT 1
                """
            ).fetchone()

    def get_dream_run(self, run_id: str) -> sqlite3.Row | None:
        with open_connection(self.db_path) as connection:
            return connection.execute(
                """
                SELECT *
                FROM dream_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

    def get_first_memory_event_time(self) -> str | None:
        with open_connection(self.db_path) as connection:
            row = connection.execute(
                "SELECT MIN(created_at) AS first_created_at FROM memory_events"
            ).fetchone()
        return row["first_created_at"] if row and row["first_created_at"] else None

    def count_memory_events_since(self, completed_at: str | None = None) -> int:
        with open_connection(self.db_path) as connection:
            if completed_at is None:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM memory_events"
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM memory_events
                    WHERE created_at > ?
                    """,
                    (completed_at,),
                ).fetchone()
        return int(row["count"])

    def start_dream_run(
        self,
        *,
        trigger_reason: str,
        auto_triggered: bool,
        requires_chat: bool,
        status: str = "running",
    ) -> str:
        run_id = _new_id("dream")
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO dream_runs (
                    run_id,
                    status,
                    trigger_reason,
                    auto_triggered,
                    requires_chat
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    status,
                    trigger_reason,
                    int(auto_triggered),
                    int(requires_chat),
                ),
            )
        return run_id

    def get_active_dream_run(self) -> sqlite3.Row | None:
        with open_connection(self.db_path) as connection:
            return connection.execute(
                """
                SELECT *
                FROM dream_runs
                WHERE status IN ('queued', 'running', 'awaiting_host_compaction')
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()

    def mark_dream_run_running(self, run_id: str) -> None:
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                UPDATE dream_runs
                SET status = 'running'
                WHERE run_id = ?
                """,
                (run_id,),
            )

    def fail_dream_run(self, run_id: str, *, error: str, notes: list[str] | None = None) -> None:
        merged_notes = list(notes or [])
        merged_notes.append(error)
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                UPDATE dream_runs
                SET status = 'failed',
                    notes_json = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
                """,
                (_json_dumps(merged_notes), run_id),
            )

    def complete_dream_run(
        self,
        run_id: str,
        *,
        status: str,
        candidate_count: int,
        clusters_created: int,
        memory_path: str | None,
        notes: list[str],
        mark_completed: bool = True,
    ) -> None:
        with open_connection(self.db_path) as connection:
            if mark_completed:
                connection.execute(
                    """
                    UPDATE dream_runs
                    SET status = ?,
                        candidate_count = ?,
                        clusters_created = ?,
                        memory_path = ?,
                        notes_json = ?,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE run_id = ?
                    """,
                    (
                        status,
                        candidate_count,
                        clusters_created,
                        memory_path,
                        _json_dumps(notes),
                        run_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE dream_runs
                    SET status = ?,
                        candidate_count = ?,
                        clusters_created = ?,
                        memory_path = ?,
                        notes_json = ?,
                        completed_at = NULL
                    WHERE run_id = ?
                    """,
                    (
                        status,
                        candidate_count,
                        clusters_created,
                        memory_path,
                        _json_dumps(notes),
                        run_id,
                    ),
                )

    def create_dream_compaction_task(
        self,
        *,
        run_id: str,
        requested_backend: str,
        fallback_backend: str,
        reason: str | None,
        suggested_title: str | None,
        suggested_description: str | None,
        prepared_content: str,
        prompt: str,
        source_nodes: list[dict[str, Any]],
        source_node_ids: list[str],
    ) -> str:
        task_id = _new_id("dreamtask")
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO dream_compaction_tasks (
                    task_id,
                    run_id,
                    status,
                    requested_backend,
                    fallback_backend,
                    reason,
                    suggested_title,
                    suggested_description,
                    prepared_content,
                    prompt,
                    source_nodes_json,
                    source_node_ids_json
                )
                VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    run_id,
                    requested_backend,
                    fallback_backend,
                    reason,
                    suggested_title,
                    suggested_description,
                    prepared_content,
                    prompt,
                    _json_dumps(source_nodes),
                    _json_dumps(source_node_ids),
                ),
            )
        return task_id

    def get_dream_compaction_task(self, task_id: str) -> sqlite3.Row | None:
        with open_connection(self.db_path) as connection:
            return connection.execute(
                """
                SELECT *
                FROM dream_compaction_tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()

    def list_dream_compaction_tasks(
        self,
        *,
        run_id: str | None = None,
        status: str | None = "pending",
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        where_clause = ""
        if clauses:
            where_clause = "WHERE " + " AND ".join(clauses)

        with open_connection(self.db_path) as connection:
            return connection.execute(
                f"""
                SELECT *
                FROM dream_compaction_tasks
                {where_clause}
                ORDER BY created_at ASC
                """,
                params,
            ).fetchall()

    def mark_dream_compaction_task_running(self, task_id: str) -> None:
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                UPDATE dream_compaction_tasks
                SET status = 'running'
                WHERE task_id = ?
                """,
                (task_id,),
            )

    def fail_dream_compaction_task(self, task_id: str, *, error: str) -> None:
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                UPDATE dream_compaction_tasks
                SET status = 'failed',
                    reason = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
                """,
                (error, task_id),
            )

    def count_pending_dream_compaction_tasks(self, run_id: str) -> int:
        with open_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM dream_compaction_tasks
                WHERE run_id = ? AND status IN ('pending', 'running')
                """,
                (run_id,),
            ).fetchone()
        return int(row["count"])

    def complete_dream_compaction_task(
        self,
        task_id: str,
        *,
        resolved_node_id: str,
        resolution_backend: str,
    ) -> None:
        with open_connection(self.db_path) as connection:
            connection.execute(
                """
                UPDATE dream_compaction_tasks
                SET status = 'completed',
                    resolved_node_id = ?,
                    resolution_backend = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
                """,
                (resolved_node_id, resolution_backend, task_id),
            )

    def list_dream_runs(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
    ) -> list[sqlite3.Row]:
        if status is None:
            query = """
                SELECT r.*,
                       (
                           SELECT COUNT(*)
                           FROM dream_compaction_tasks t
                           WHERE t.run_id = r.run_id
                             AND t.status IN ('pending', 'running')
                       ) AS pending_task_count
                FROM dream_runs r
                ORDER BY r.started_at DESC
                LIMIT ?
            """
            params: tuple[Any, ...] = (limit,)
        else:
            query = """
                SELECT r.*,
                       (
                           SELECT COUNT(*)
                           FROM dream_compaction_tasks t
                           WHERE t.run_id = r.run_id
                             AND t.status IN ('pending', 'running')
                       ) AS pending_task_count
                FROM dream_runs r
                WHERE r.status = ?
                ORDER BY r.started_at DESC
                LIMIT ?
            """
            params = (status, limit)

        with open_connection(self.db_path) as connection:
            return connection.execute(query, params).fetchall()

    def read_nodes(self, ids: list[str], *, include_content: bool) -> dict[str, ReadNodeResult]:
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        with open_connection(self.db_path) as connection:
            node_rows = connection.execute(
                f"SELECT rowid, * FROM nodes WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            edge_rows = connection.execute(
                f"""
                SELECT src_id, dst_id, relation, created_at
                     , strength_score, durability, status, metadata_json, last_reinforced_at
                FROM edges
                WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})
                ORDER BY created_at ASC
                """,
                ids + ids,
            ).fetchall()

        edges_by_node: dict[str, list[EdgeRecord]] = {node_id: [] for node_id in ids}
        for edge_row in edge_rows:
            edge = self._row_to_edge_record(edge_row)
            if edge.src_id in edges_by_node:
                edges_by_node[edge.src_id].append(edge)
            if edge.dst_id in edges_by_node and edge.dst_id != edge.src_id:
                edges_by_node[edge.dst_id].append(edge)

        results: dict[str, ReadNodeResult] = {}
        for row in node_rows:
            results[row["id"]] = ReadNodeResult(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                content=row["content"] if include_content else None,
                tags=_json_loads(row["tags_json"], []),
                metadata=normalize_node_metadata(_json_loads(row["metadata_json"], {})),
                node_type=row["node_type"],
                durability=row["durability"],
                edges=edges_by_node.get(row["id"], []),
                updated_at=row["updated_at"],
                created_at=row["created_at"],
                last_reinforced_at=row["last_reinforced_at"],
            )
        return results

    def get_node(self, node_id: str) -> NodeRecord:
        row = self.get_node_record(node_id)
        return self._row_to_node_record(row)

    def list_all_nodes(self) -> list[NodeRecord]:
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                "SELECT rowid, * FROM nodes ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._row_to_node_record(row) for row in rows]

    def list_nodes_by_type(self, *, node_type: str) -> list[NodeRecord]:
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT rowid, *
                FROM nodes
                WHERE node_type = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (node_type,),
            ).fetchall()
        return [self._row_to_node_record(row) for row in rows]

    def list_nodes_missing_embeddings(self) -> list[NodeRecord]:
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT rowid, *
                FROM nodes
                WHERE embedding IS NULL
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
        return [self._row_to_node_record(row) for row in rows]

    def list_recent_or_frequent_nodes(
        self,
        *,
        window_hours: int,
        min_accesses: int,
        limit: int,
    ) -> list[NodeRecord]:
        cutoff = (datetime.now(UTC) - timedelta(hours=window_hours)).strftime("%Y-%m-%d %H:%M:%S")
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                """
                WITH recent_nodes AS (
                    SELECT DISTINCT node_id
                    FROM audit_logs
                    WHERE created_at >= ?
                ),
                frequent_nodes AS (
                    SELECT node_id
                    FROM access_logs
                    WHERE created_at >= ?
                    GROUP BY node_id
                    HAVING COUNT(*) >= ?
                ),
                candidate_nodes AS (
                    SELECT node_id FROM recent_nodes
                    UNION
                    SELECT node_id FROM frequent_nodes
                )
                SELECT n.*
                , n.rowid AS rowid
                FROM nodes n
                JOIN candidate_nodes c ON c.node_id = n.id
                ORDER BY n.updated_at DESC, n.created_at DESC
                LIMIT ?
                """,
                (cutoff, cutoff, min_accesses, limit),
            ).fetchall()
        return [self._row_to_node_record(row) for row in rows]

    def search_keyword_matches(self, *, keyword: str, top_k: int) -> list[tuple[str, float]]:
        fts_query = self._normalize_fts_query(keyword)
        if not fts_query:
            return []
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT n.id, bm25(nodes_fts) AS score
                FROM nodes_fts
                JOIN nodes n ON n.id = nodes_fts.id
                WHERE nodes_fts MATCH ?
                ORDER BY bm25(nodes_fts)
                LIMIT ?
                """,
                (fts_query, top_k),
            ).fetchall()
        return [(row["id"], float(row["score"])) for row in rows]

    def search_semantic_matches(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        with open_connection(self.db_path) as connection:
            dimension = self.get_embedding_dimension(connection=connection)
            if dimension is None:
                return []
            if dimension != len(query_embedding):
                raise ValueError(
                    "Query embedding dimension "
                    f"{len(query_embedding)} does not match index dimension {dimension}."
                )
            rows = connection.execute(
                f"""
                SELECT n.id, {self.VECTOR_TABLE_NAME}.distance AS distance
                FROM {self.VECTOR_TABLE_NAME}
                JOIN nodes n ON n.rowid = {self.VECTOR_TABLE_NAME}.rowid
                WHERE {self.VECTOR_TABLE_NAME}.embedding MATCH ?
                  AND {self.VECTOR_TABLE_NAME}.k = ?
                ORDER BY distance
                """,
                (json.dumps(query_embedding), top_k),
            ).fetchall()
        return [
            (row["id"], float(row["distance"]))
            for row in rows
            if row["distance"] is not None
        ]

    def build_search_results(
        self,
        node_ids: list[str],
        *,
        include_neighbors: int,
        scores: dict[str, dict[str, float]] | None = None,
    ) -> list[SearchResult]:
        if not node_ids:
            return []
        placeholders = ", ".join("?" for _ in node_ids)
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT id, name, description, tags_json
                     , metadata_json, node_type, durability
                FROM nodes
                WHERE id IN ({placeholders})
                """,
                node_ids,
            ).fetchall()

        neighbors_by_root = self._load_neighbors_for_roots(node_ids, include_neighbors)
        rows_by_id = {row["id"]: row for row in rows}
        results: list[SearchResult] = []
        for node_id in node_ids:
            row = rows_by_id.get(node_id)
            if row is None:
                continue
            results.append(
                SearchResult(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    tags=_json_loads(row["tags_json"], []),
                    metadata=normalize_node_metadata(_json_loads(row["metadata_json"], {})),
                    node_type=row["node_type"],
                    durability=row["durability"],
                    score=(scores or {}).get(row["id"], {}).get("score"),
                    semantic_score=(scores or {}).get(row["id"], {}).get("semantic_score"),
                    keyword_score=(scores or {}).get(row["id"], {}).get("keyword_score"),
                    linked_nodes=neighbors_by_root.get(row["id"], []),
                )
            )
        return results

    def list_nodes_for_memory_projection(self) -> list[NodeRecord]:
        with open_connection(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT rowid, *
                FROM nodes
                WHERE durability IN ('durable', 'pinned')
                ORDER BY
                    CASE durability WHEN 'pinned' THEN 0 ELSE 1 END,
                    CASE
                        WHEN json_extract(metadata_json, '$.profile.kind') = 'system' THEN 0
                        WHEN node_type = 'super_node' THEN 1
                        ELSE 2
                    END,
                    updated_at DESC,
                    created_at DESC
                """
            ).fetchall()
        return [self._row_to_node_record(row) for row in rows]

    def _load_neighbors_for_roots(
        self,
        root_ids: list[str],
        max_depth: int,
    ) -> dict[str, list[LinkedNode]]:
        if max_depth <= 0:
            return {root_id: [] for root_id in root_ids}
        ordered_root_ids = list(dict.fromkeys(root_ids))
        if not ordered_root_ids:
            return {}

        discovered_by_root: dict[str, dict[str, LinkedNode]] = {
            root_id: {} for root_id in ordered_root_ids
        }
        visited_by_root: dict[str, set[str]] = {
            root_id: {root_id} for root_id in ordered_root_ids
        }
        frontier_by_root: dict[str, set[str]] = {
            root_id: {root_id} for root_id in ordered_root_ids
        }

        with open_connection(self.db_path) as connection:
            for depth in range(max_depth):
                frontier = sorted(
                    {
                        node_id
                        for frontier_node_ids in frontier_by_root.values()
                        for node_id in frontier_node_ids
                    }
                )
                if not frontier:
                    break
                placeholders = ", ".join("?" for _ in frontier)
                edge_rows = connection.execute(
                    f"""
                    SELECT src_id, dst_id, relation, created_at,
                           strength_score, durability, status, metadata_json, last_reinforced_at
                    FROM edges
                    WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})
                    ORDER BY created_at ASC
                    """,
                    frontier + frontier,
                ).fetchall()
                edge_rows_by_node: dict[str, list[sqlite3.Row]] = defaultdict(list)
                for edge_row in edge_rows:
                    edge_rows_by_node[edge_row["src_id"]].append(edge_row)
                    if edge_row["dst_id"] != edge_row["src_id"]:
                        edge_rows_by_node[edge_row["dst_id"]].append(edge_row)

                pending_by_root: dict[str, list[tuple[str, str, str, int]]] = {
                    root_id: [] for root_id in ordered_root_ids
                }
                pending_neighbor_ids: set[str] = set()
                next_frontier_by_root: dict[str, set[str]] = {
                    root_id: set() for root_id in ordered_root_ids
                }

                for root_id in ordered_root_ids:
                    for current_id in sorted(frontier_by_root[root_id]):
                        for edge_row in edge_rows_by_node.get(current_id, []):
                            if edge_row["src_id"] == current_id:
                                neighbor_id = edge_row["dst_id"]
                                direction = "outbound"
                            else:
                                neighbor_id = edge_row["src_id"]
                                direction = "inbound"

                            if (
                                neighbor_id == root_id
                                or neighbor_id in discovered_by_root[root_id]
                                or neighbor_id in visited_by_root[root_id]
                            ):
                                continue

                            pending_by_root[root_id].append(
                                (neighbor_id, edge_row["relation"], direction, depth + 1)
                            )
                            pending_neighbor_ids.add(neighbor_id)

                if not pending_neighbor_ids:
                    frontier_by_root = next_frontier_by_root
                    continue

                neighbor_placeholders = ", ".join("?" for _ in pending_neighbor_ids)
                node_rows = connection.execute(
                    f"""
                    SELECT id, name, description, length(content) AS content_length
                    FROM nodes
                    WHERE id IN ({neighbor_placeholders})
                    """,
                    list(pending_neighbor_ids),
                ).fetchall()
                node_rows_by_id = {row["id"]: row for row in node_rows}

                for root_id in ordered_root_ids:
                    for neighbor_id, relation, direction, hop in pending_by_root[root_id]:
                        if neighbor_id in discovered_by_root[root_id]:
                            continue
                        node_row = node_rows_by_id.get(neighbor_id)
                        if node_row is None:
                            continue
                        discovered_by_root[root_id][neighbor_id] = LinkedNode(
                            id=node_row["id"],
                            name=node_row["name"],
                            description=node_row["description"],
                            relation=relation,
                            direction=direction,
                            content_length=node_row["content_length"],
                            hop=hop,
                        )
                        visited_by_root[root_id].add(neighbor_id)
                        next_frontier_by_root[root_id].add(neighbor_id)

                frontier_by_root = next_frontier_by_root

        return {
            root_id: list(discovered.values())
            for root_id, discovered in discovered_by_root.items()
        }

    @staticmethod
    def _normalize_fts_query(keyword: str) -> str:
        stripped = keyword.strip()
        if not stripped:
            return ""
        if any(token in stripped for token in ('"', "AND", "OR", "NOT", "*", "NEAR")):
            return stripped
        if " " in stripped:
            return f'"{stripped}"'
        return stripped

    def _ensure_vector_table(self, connection: sqlite3.Connection, dimension: int) -> None:
        connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.VECTOR_TABLE_NAME}
            USING vec0(embedding float[{dimension}] distance_metric=cosine)
            """
        )

    def _record_memory_event(
        self,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        ref_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_events (event_id, event_type, ref_id, metadata_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                _new_id("event"),
                event_type,
                ref_id,
                _json_dumps(metadata or {}),
            ),
        )

    def _create_edges(
        self,
        connection: sqlite3.Connection,
        edges: list[EdgeRecord],
    ) -> None:
        normalized_payloads = [
            (
                edge.src_id,
                edge.dst_id,
                edge.relation,
                edge.strength_score,
                edge.durability,
                edge.status,
                _json_dumps(normalize_edge_metadata(edge.metadata)),
                edge.last_reinforced_at,
            )
            for edge in edges
        ]
        connection.executemany(
            """
            INSERT INTO edges (
                src_id,
                dst_id,
                relation,
                strength_score,
                durability,
                status,
                metadata_json,
                last_reinforced_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(src_id, dst_id, relation)
            DO UPDATE SET strength_score = excluded.strength_score,
                          durability = excluded.durability,
                          status = excluded.status,
                          metadata_json = excluded.metadata_json,
                          last_reinforced_at = excluded.last_reinforced_at
            """,
            normalized_payloads,
        )
        for edge in edges:
            self._record_memory_event(
                connection,
                event_type="edge_create",
                ref_id=f"{edge.src_id}:{edge.relation}:{edge.dst_id}",
                metadata={
                    "strength_score": edge.strength_score,
                    "durability": edge.durability,
                    "status": edge.status,
                },
            )

    def _upsert_vector(
        self,
        connection: sqlite3.Connection,
        *,
        rowid: int,
        embedding: list[float],
    ) -> None:
        dimension = self.get_embedding_dimension(connection=connection)
        if dimension is None:
            dimension = len(embedding)
            self.set_embedding_dimension(dimension, connection=connection)
            self._ensure_vector_table(connection, dimension)
        elif dimension != len(embedding):
            raise ValueError(
                "Embedding dimension "
                f"{len(embedding)} does not match configured index dimension {dimension}."
            )
        else:
            self._ensure_vector_table(connection, dimension)

        connection.execute(
            f"DELETE FROM {self.VECTOR_TABLE_NAME} WHERE rowid = ?",
            (rowid,),
        )
        connection.execute(
            f"""
            INSERT INTO {self.VECTOR_TABLE_NAME}(rowid, embedding)
            VALUES (?, ?)
            """,
            (rowid, sqlite_vec.serialize_float32(embedding)),
        )

    @staticmethod
    def _row_to_node_record(row: sqlite3.Row) -> NodeRecord:
        return NodeRecord(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            content=row["content"],
            embedding=_deserialize_embedding(row["embedding"]),
            tags=_json_loads(row["tags_json"], []),
            metadata=normalize_node_metadata(_json_loads(row["metadata_json"], {})),
            node_type=row["node_type"],
            durability=row["durability"],
            last_reinforced_at=row["last_reinforced_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_edge_record(row: sqlite3.Row) -> EdgeRecord:
        return EdgeRecord(
            src_id=row["src_id"],
            dst_id=row["dst_id"],
            relation=row["relation"],
            strength_score=float(row["strength_score"] or 1.0),
            durability=row["durability"] or "durable",
            status=row["status"] or "active",
            metadata=normalize_edge_metadata(_json_loads(row["metadata_json"], {})),
            created_at=row["created_at"],
            last_reinforced_at=row["last_reinforced_at"],
        )
