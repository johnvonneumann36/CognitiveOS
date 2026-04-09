SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS nodes (
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

CREATE TABLE IF NOT EXISTS edges (
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

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);

CREATE TABLE IF NOT EXISTS audit_logs (
    log_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    previous_content TEXT,
    actor TEXT DEFAULT 'agent',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS access_logs (
    log_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    access_type TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_access_logs_node_id ON access_logs(node_id);
CREATE INDEX IF NOT EXISTS idx_access_logs_created_at ON access_logs(created_at);

CREATE TABLE IF NOT EXISTS semantic_neighbors (
    node_id TEXT NOT NULL,
    neighbor_id TEXT NOT NULL,
    similarity REAL NOT NULL,
    computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (node_id, neighbor_id),
    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (neighbor_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_semantic_neighbors_node_id ON semantic_neighbors(node_id);
CREATE INDEX IF NOT EXISTS idx_semantic_neighbors_neighbor_id ON semantic_neighbors(neighbor_id);

CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    ref_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memory_events_created_at ON memory_events(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_events_event_type ON memory_events(event_type);

CREATE TABLE IF NOT EXISTS dream_runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    trigger_reason TEXT,
    auto_triggered INTEGER NOT NULL DEFAULT 0,
    requires_chat INTEGER NOT NULL DEFAULT 0,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    clusters_created INTEGER NOT NULL DEFAULT 0,
    memory_path TEXT,
    notes_json TEXT NOT NULL DEFAULT '[]',
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_dream_runs_completed_at ON dream_runs(completed_at);

CREATE TABLE IF NOT EXISTS dream_compaction_tasks (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_backend TEXT NOT NULL,
    fallback_backend TEXT NOT NULL DEFAULT 'heuristic',
    reason TEXT,
    suggested_title TEXT,
    suggested_description TEXT,
    prepared_content TEXT NOT NULL,
    prompt TEXT NOT NULL,
    source_nodes_json TEXT NOT NULL DEFAULT '[]',
    source_node_ids_json TEXT NOT NULL DEFAULT '[]',
    resolved_node_id TEXT,
    resolution_backend TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    FOREIGN KEY (run_id) REFERENCES dream_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dream_compaction_tasks_run_id
ON dream_compaction_tasks(run_id);
CREATE INDEX IF NOT EXISTS idx_dream_compaction_tasks_status
ON dream_compaction_tasks(status);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    id UNINDEXED,
    search_text,
    tokenize = 'unicode61'
);

CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(id, search_text)
    VALUES (new.id, '');
END;

CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    DELETE FROM nodes_fts WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    DELETE FROM nodes_fts WHERE id = old.id;
    INSERT INTO nodes_fts(id, search_text)
    VALUES (new.id, '');
END;
"""
