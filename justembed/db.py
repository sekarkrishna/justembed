"""
DuckDB storage layer — one DB per knowledge base.
"""

import re
from pathlib import Path
from typing import Any, List, Dict, Tuple

import duckdb


def _valid_kb_name(name: str) -> bool:
    """KB name: alphanumeric + underscore only, max 64 chars."""
    return bool(re.match(r"^[a-zA-Z0-9_]{1,64}$", name))


def get_kb_path(workspace: Path, kb_name: str) -> Path:
    """Path to KB DuckDB file."""
    if not _valid_kb_name(kb_name):
        raise ValueError(
            f"Invalid KB name '{kb_name}': alphanumeric + underscore only, max 64 chars"
        )
    return workspace / "kb" / f"{kb_name}.duckdb"


def ensure_chunks_table(conn: duckdb.DuckDBPyConnection, embedding_dim: int = 384) -> None:
    """Create chunks table if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT,
            UNIQUE(file, chunk_index)
        )
    """)


def ensure_metadata_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create metadata table if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)


def set_kb_metadata(conn: duckdb.DuckDBPyConnection, key: str, value: str) -> None:
    """Set KB metadata key-value pair."""
    ensure_metadata_table(conn)
    conn.execute("""
        INSERT INTO metadata (key, value) VALUES (?, ?)
        ON CONFLICT (key) DO UPDATE SET value = excluded.value
    """, [key, value])


def get_kb_metadata(conn: duckdb.DuckDBPyConnection, key: str, default: str = None) -> str:
    """Get KB metadata value by key."""
    ensure_metadata_table(conn)
    result = conn.execute("SELECT value FROM metadata WHERE key = ?", [key]).fetchone()
    return result[0] if result else default


def ensure_upload_history_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create upload_history table if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS upload_history (
            id INTEGER PRIMARY KEY,
            type TEXT NOT NULL,
            filename TEXT NOT NULL,
            kb_name TEXT,
            model_name TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def list_kbs(workspace: Path) -> List[str]:
    """List knowledge base names from kb/ folder (excludes _history)."""
    kb_dir = workspace / "kb"
    if not kb_dir.exists():
        return []
    return [p.stem for p in kb_dir.glob("*.duckdb") if p.stem != "_history"]


def create_kb(workspace: Path, kb_name: str, model_type: str = "e5", model_name: str = "e5-small") -> None:
    """Create a new knowledge base (DuckDB file) with model metadata."""
    path = get_kb_path(workspace, kb_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    ensure_chunks_table(conn)
    ensure_metadata_table(conn)
    
    # Store model information
    set_kb_metadata(conn, "model_type", model_type)
    set_kb_metadata(conn, "model_name", model_name)
    
    conn.close()


def delete_kb(workspace: Path, kb_name: str) -> None:
    """Delete a knowledge base (DuckDB file)."""
    path = get_kb_path(workspace, kb_name)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base '{kb_name}' does not exist")
    path.unlink()  # Delete the file


def get_connection(workspace: Path, kb_name: str) -> duckdb.DuckDBPyConnection:
    """Open DuckDB connection for KB."""
    path = get_kb_path(workspace, kb_name)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base '{kb_name}' does not exist")
    return duckdb.connect(str(path), read_only=False)


def insert_chunks(
    conn: duckdb.DuckDBPyConnection,
    file: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    """Insert chunks and embeddings into KB.
    
    If chunks for this file already exist, they will be replaced.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")

    # Delete existing chunks for this file (allows re-uploading)
    conn.execute("DELETE FROM chunks WHERE file = ?", [file])

    import json
    base_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0]
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        emb_json = json.dumps([float(x) for x in emb])
        conn.execute(
            """
            INSERT INTO chunks (id, file, chunk_index, text, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            [base_id + i + 1, file, chunk["chunk_index"], chunk["text"], emb_json],
        )


def insert_chunks_without_embeddings(
    conn: duckdb.DuckDBPyConnection,
    file: str,
    chunks: List[Dict[str, Any]],
) -> None:
    """Insert chunks without embeddings (for preview workflow).
    
    If chunks for this file already exist, they will be replaced.
    """
    # Delete existing chunks for this file (allows re-uploading)
    conn.execute("DELETE FROM chunks WHERE file = ?", [file])
    
    base_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0]
    for i, chunk in enumerate(chunks):
        conn.execute(
            """
            INSERT INTO chunks (id, file, chunk_index, text, embedding)
            VALUES (?, ?, ?, ?, NULL)
            """,
            [base_id + i + 1, file, chunk["chunk_index"], chunk["text"]],
        )


def get_chunks_without_embeddings(
    conn: duckdb.DuckDBPyConnection,
    file: str,
) -> List[Dict[str, Any]]:
    """Get chunks that don't have embeddings yet."""
    rows = conn.execute(
        "SELECT id, file, chunk_index, text FROM chunks WHERE file = ? AND embedding IS NULL",
        [file],
    ).fetchall()
    return [
        {"id": r[0], "file": r[1], "chunk_index": r[2], "text": r[3]}
        for r in rows
    ]


def update_embeddings(
    conn: duckdb.DuckDBPyConnection,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    """Update chunks with embeddings."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    
    import json
    for chunk, emb in zip(chunks, embeddings):
        emb_json = json.dumps([float(x) for x in emb])
        conn.execute(
            "UPDATE chunks SET embedding = ? WHERE id = ?",
            [emb_json, chunk["id"]],
        )


def add_upload_history(
    workspace: Path,
    filename: str,
    upload_type: str,
    kb_name: str = None,
    model_name: str = None,
    max_entries: int = 50
) -> int:
    """
    Add to upload history; purge old entries if over max.
    
    Args:
        workspace: Workspace path
        filename: Name of uploaded file
        upload_type: "kb_upload" or "model_training"
        kb_name: KB name (for kb_upload type)
        model_name: Model name (for model_training type)
        max_entries: Maximum history entries to keep
    
    Returns:
        Number of entries purged (0 if none)
    """
    history_path = workspace / "kb" / "_history.duckdb"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(history_path))
    ensure_upload_history_table(conn)
    
    # Insert new entry
    next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM upload_history").fetchone()[0]
    conn.execute(
        """INSERT INTO upload_history (id, type, filename, kb_name, model_name, uploaded_at) 
           VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
        [next_id, upload_type, filename, kb_name, model_name],
    )
    
    # Check if we need to purge
    count = conn.execute("SELECT COUNT(*) FROM upload_history").fetchone()[0]
    purged = 0
    
    if count > max_entries:
        to_remove = count - max_entries
        ids_to_remove = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM upload_history ORDER BY uploaded_at ASC LIMIT ?",
                [to_remove],
            ).fetchall()
        ]
        for iid in ids_to_remove:
            conn.execute("DELETE FROM upload_history WHERE id = ?", [iid])
        purged = to_remove
    
    conn.close()
    return purged


def get_upload_history(workspace: Path, limit: int = 50) -> Tuple[List[Dict], int]:
    """
    Get upload history.
    
    Returns:
        Tuple of (history_list, total_purged_count)
    """
    history_path = workspace / "kb" / "_history.duckdb"
    if not history_path.exists():
        return ([], 0)
    
    conn = duckdb.connect(str(history_path))
    ensure_upload_history_table(conn)
    
    # Get history entries
    rows = conn.execute(
        """SELECT type, filename, kb_name, model_name, uploaded_at 
           FROM upload_history 
           ORDER BY uploaded_at DESC 
           LIMIT ?""",
        [limit],
    ).fetchall()
    
    history = []
    for r in rows:
        history.append({
            "type": r[0],
            "filename": r[1],
            "kb_name": r[2],
            "model_name": r[3],
            "uploaded_at": r[4],
        })
    
    # Calculate total purged (rough estimate based on max ID vs current count)
    max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM upload_history").fetchone()[0]
    current_count = conn.execute("SELECT COUNT(*) FROM upload_history").fetchone()[0]
    total_purged = max(0, max_id - current_count)
    
    conn.close()
    return (history, total_purged)


def query_chunks(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: List[float],
    top_k: int = 5,
    kb_name: str = "",
) -> List[Dict[str, Any]]:
    """Return top-k chunks by cosine similarity (embeddings are L2-normalized)."""
    import json
    import numpy as np

    q = np.array(query_embedding, dtype=np.float32)
    rows = conn.execute("SELECT id, file, chunk_index, text, embedding FROM chunks").fetchall()
    scored = []
    for r in rows:
        raw = r[4]
        emb = json.loads(raw) if isinstance(raw, str) else (list(raw) if raw else [])
        if len(emb) != len(q):
            continue
        score = float(np.dot(q, np.array(emb, dtype=np.float32)))
        scored.append((r[1], r[2], r[3], score, emb))  # Include embedding
    scored.sort(key=lambda x: x[3], reverse=True)
    return [
        {"file": r[0], "chunk_index": r[1], "text": r[2], "score": r[3], "kb": kb_name, "embedding": r[4]}
        for r in scored[:top_k]
    ]
