"""
Job management system for background tasks.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, List, Dict, Any


class JobManager:
    """Manages background jobs with SQLite persistence."""
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.db_path = workspace / "jobs.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize job database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER DEFAULT 0,
                message TEXT,
                params TEXT,
                result TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status 
            ON jobs(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_created 
            ON jobs(created_at DESC)
        """)
        conn.commit()
        conn.close()
    
    def create_job(self, job_type: str, params: Dict[str, Any]) -> str:
        """Create a new job and return job_id."""
        import uuid
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO jobs (job_id, job_type, status, params, created_at, updated_at, message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [job_id, job_type, "queued", json.dumps(params), now, now, "Job queued"])
        conn.commit()
        conn.close()
        
        return job_id
    
    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """Update job status and progress."""
        conn = sqlite3.connect(self.db_path)
        updates = []
        values = []
        
        if status is not None:
            updates.append("status = ?")
            values.append(status)
            
            # Set completed_at when job finishes
            if status in ["complete", "failed"]:
                updates.append("completed_at = ?")
                values.append(datetime.utcnow().isoformat())
        
        if progress is not None:
            updates.append("progress = ?")
            values.append(progress)
        
        if message is not None:
            updates.append("message = ?")
            values.append(message)
        
        if result is not None:
            updates.append("result = ?")
            values.append(json.dumps(result))
        
        if error is not None:
            updates.append("error = ?")
            values.append(error)
        
        updates.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(job_id)
        
        conn.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?", values)
        conn.commit()
        conn.close()
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", [job_id]).fetchone()
        conn.close()
        
        if not row:
            return None
        
        job = dict(row)
        
        # Parse JSON fields
        if job.get("params"):
            try:
                job["params"] = json.loads(job["params"])
            except:
                pass
        
        if job.get("result"):
            try:
                job["result"] = json.loads(job["result"])
            except:
                pass
        
        return job
    
    def list_jobs(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List jobs with optional status filter."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        if status:
            rows = conn.execute("""
                SELECT * FROM jobs 
                WHERE status = ?
                ORDER BY created_at DESC 
                LIMIT ?
            """, [status, limit]).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM jobs 
                ORDER BY created_at DESC 
                LIMIT ?
            """, [limit]).fetchall()
        
        conn.close()
        
        jobs = []
        for row in rows:
            job = dict(row)
            
            # Parse JSON fields
            if job.get("params"):
                try:
                    job["params"] = json.loads(job["params"])
                except:
                    pass
            
            if job.get("result"):
                try:
                    job["result"] = json.loads(job["result"])
                except:
                    pass
            
            jobs.append(job)
        
        return jobs
    
    def get_next_queued_job(self) -> Optional[Dict[str, Any]]:
        """Get the next queued job to process."""
        jobs = self.list_jobs(limit=100, status="queued")
        return jobs[0] if jobs else None
    
    def delete_old_jobs(self, days: int = 7):
        """Delete completed/failed jobs older than specified days."""
        cutoff = datetime.utcnow().timestamp() - (days * 86400)
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            DELETE FROM jobs 
            WHERE status IN ('complete', 'failed') 
            AND completed_at < ?
        """, [cutoff_iso])
        conn.commit()
        deleted = conn.total_changes
        conn.close()
        
        return deleted
    
    def get_job_stats(self) -> Dict[str, int]:
        """Get job statistics."""
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        for status in ["queued", "running", "complete", "failed"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?", [status]
            ).fetchone()[0]
            stats[status] = count
        
        conn.close()
        return stats
