"""
Background worker for processing jobs.
"""

import asyncio
import traceback
from pathlib import Path
from typing import Optional

from justembed.jobs import JobManager
from justembed.training.trainer import CustomModelTrainer
from justembed.embedder import E5Embedder, CustomEmbedder
from justembed import db


class BackgroundWorker:
    """Process background jobs from the job queue."""
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.job_manager = JobManager(workspace)
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the background worker."""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._process_loop())
        print(f"Background worker started for workspace: {self.workspace}")
    
    async def stop(self):
        """Stop the background worker."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("Background worker stopped")
    
    async def _process_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                # Get next queued job
                job = self.job_manager.get_next_queued_job()
                
                if job:
                    await self._process_job(job)
                else:
                    # No jobs, sleep for a bit
                    await asyncio.sleep(1)
            
            except Exception as e:
                print(f"Worker error: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)
    
    async def _process_job(self, job: dict):
        """Process a single job."""
        job_id = job["job_id"]
        job_type = job["job_type"]
        
        print(f"Processing job {job_id} ({job_type})")
        
        try:
            self.job_manager.update_job(
                job_id,
                status="running",
                progress=0,
                message="Starting job..."
            )
            
            if job_type == "train_model":
                await self._process_train_model(job)
            elif job_type == "embed_chunks":
                await self._process_embed_chunks(job)
            else:
                raise ValueError(f"Unknown job type: {job_type}")
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"Job {job_id} failed: {error_msg}")
            traceback.print_exc()
            
            self.job_manager.update_job(
                job_id,
                status="failed",
                message="Job failed",
                error=error_msg,
            )
    
    async def _process_train_model(self, job: dict):
        """Process model training job."""
        job_id = job["job_id"]
        params = job["params"]
        
        # Extract parameters
        corpus = params["corpus"]
        model_name = params["model_name"]
        embedding_dim = params.get("embedding_dim", 128)
        max_features = params.get("max_features", 5000)
        
        # Progress callback
        def progress_callback(percent: int, message: str):
            self.job_manager.update_job(
                job_id,
                progress=percent,
                message=message,
            )
        
        # Train model (run in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        
        def train_sync():
            trainer = CustomModelTrainer()
            return trainer.train(
                corpus=corpus,
                model_name=model_name,
                embedding_dim=embedding_dim,
                max_features=max_features,
                progress_callback=progress_callback,
            )
        
        model_dir = await loop.run_in_executor(None, train_sync)
        
        # Mark complete
        self.job_manager.update_job(
            job_id,
            status="complete",
            progress=100,
            message="Training complete!",
            result={
                "model_dir": str(model_dir),
                "model_name": model_name,
            },
        )
        
        print(f"Job {job_id} completed: model saved to {model_dir}")
    
    async def _process_embed_chunks(self, job: dict):
        """Process chunk embedding job."""
        job_id = job["job_id"]
        params = job["params"]
        
        # Extract parameters
        kb_name = params["kb_name"]
        filename = params["filename"]
        model_type = params.get("model_type", "e5")
        model_name = params.get("model_name", "e5-small")
        
        # Get chunks to embed
        conn = db.get_connection(self.workspace, kb_name)
        chunks_to_embed = db.get_chunks_without_embeddings(conn, filename)
        total_chunks = len(chunks_to_embed)
        
        if total_chunks == 0:
            self.job_manager.update_job(
                job_id,
                status="complete",
                progress=100,
                message="No chunks to embed",
            )
            conn.close()
            return
        
        # Get embedder
        if model_type == "e5":
            embedder = E5Embedder()
        else:
            embedder = CustomEmbedder(model_name)
        
        # Embed in batches
        batch_size = 10
        for i in range(0, total_chunks, batch_size):
            batch = chunks_to_embed[i:i + batch_size]
            
            # Update progress
            progress = int((i / total_chunks) * 100)
            self.job_manager.update_job(
                job_id,
                progress=progress,
                message=f"Embedding chunks {i + 1}-{min(i + batch_size, total_chunks)} of {total_chunks}...",
            )
            
            # Embed batch (run in executor)
            loop = asyncio.get_event_loop()
            texts = [c["text"] for c in batch]
            embeddings = await loop.run_in_executor(None, embedder.embed, texts)
            
            # Update database
            db.update_embeddings(conn, batch, embeddings)
            
            # Small delay to avoid blocking
            await asyncio.sleep(0.01)
        
        conn.close()
        
        # Mark complete
        self.job_manager.update_job(
            job_id,
            status="complete",
            progress=100,
            message=f"Embedded {total_chunks} chunks successfully!",
            result={
                "kb_name": kb_name,
                "filename": filename,
                "total_chunks": total_chunks,
            },
        )
        
        print(f"Job {job_id} completed: embedded {total_chunks} chunks")


# Global worker instance
_worker: Optional[BackgroundWorker] = None


def get_worker(workspace: Path) -> BackgroundWorker:
    """Get or create the global worker instance."""
    global _worker
    if _worker is None or _worker.workspace != workspace:
        _worker = BackgroundWorker(workspace)
    return _worker
