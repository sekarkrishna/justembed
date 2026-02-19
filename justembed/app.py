"""
FastAPI app — JustEmbed server.
"""

import re
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import FastAPI, Form, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import FormData

from justembed.config import get_workspace, register_workspace, deregister_workspace, list_registered_workspaces, ensure_workspace_structure as ensure_ws
from justembed import db
from justembed.chunker import chunk_text
from justembed.embedder import E5Embedder, CustomEmbedder
from justembed.training.trainer import CustomModelTrainer
from justembed.jobs import JobManager
from justembed.worker import get_worker

# Lazy embedder
_embedder: Optional[E5Embedder] = None
_custom_embedders: Dict[str, CustomEmbedder] = {}


def get_embedder(model_type: str = "e5", model_name: str = "e5-small"):
    """Get embedder by type and name."""
    if model_type == "e5":
        global _embedder
        if _embedder is None:
            _embedder = E5Embedder()
        return _embedder
    elif model_type == "custom":
        if model_name not in _custom_embedders:
            _custom_embedders[model_name] = CustomEmbedder(model_name)
        return _custom_embedders[model_name]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# File size limit (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Temporary storage for upload content (in-memory cache)
_upload_cache: Dict[str, Dict] = {}


def list_custom_models() -> list[dict]:
    """List available custom models."""
    from pathlib import Path
    import json
    from justembed.config import get_custom_models_dir
    
    try:
        models_dir = get_custom_models_dir()
    except FileNotFoundError:
        # Workspace not configured yet
        return []
    
    if not models_dir.exists():
        return []
    
    models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            config_path = model_dir / "config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    models.append({
                        "name": model_dir.name,
                        "embedding_dim": config.get("embedding_dim", 128),
                        "created_at": config.get("created_at", ""),
                        "num_chunks": config.get("corpus_stats", {}).get("num_chunks", 0),
                    })
                except Exception:
                    pass
    return models


def create_app() -> FastAPI:
    app = FastAPI(title="JustEmbed", version="0.1.1a7")

    # Templates and static
    pkg_root = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(pkg_root / "templates"))
    if (pkg_root / "static").exists():
        app.mount("/static", StaticFiles(directory=str(pkg_root / "static")), name="static")

    # Startup: Start background worker
    @app.on_event("startup")
    async def startup_event():
        try:
            workspace = get_workspace()
            worker = get_worker(workspace)
            await worker.start()
        except Exception as e:
            print(f"Could not start background worker: {e}")
            # Don't fail startup if no workspace configured yet
    
    # Shutdown: Stop background worker
    @app.on_event("shutdown")
    async def shutdown_event():
        try:
            workspace = get_workspace()
            worker = get_worker(workspace)
            await worker.stop()
        except Exception:
            pass

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        try:
            workspace = get_workspace()
        except Exception:
            return templates.TemplateResponse("setup.html", {"request": request})
        
        # Get list of all registered workspaces
        registered_workspaces = list_registered_workspaces()
        
        # Get KB list with model info and size
        kb_names = db.list_kbs(workspace)
        kbs = []
        for kb_name in kb_names:
            try:
                conn = db.get_connection(workspace, kb_name)
                model_type = db.get_kb_metadata(conn, "model_type", "e5")
                model_name = db.get_kb_metadata(conn, "model_name", "e5-small")
                
                # Get KB size (count chunks)
                chunk_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
                conn.close()
                
                kbs.append({
                    "name": kb_name,
                    "model_type": model_type,
                    "model_name": model_name,
                    "chunk_count": chunk_count,
                })
            except Exception:
                # If we can't read metadata, assume default
                kbs.append({
                    "name": kb_name,
                    "model_type": "e5",
                    "model_name": "e5-small",
                    "chunk_count": 0,
                })
        
        custom_models = list_custom_models()
        
        # Get upload history
        history, total_purged = db.get_upload_history(workspace)
        
        # Check if we should highlight a newly created KB
        new_kb = request.query_params.get("new_kb")
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "workspace": str(workspace),
                "registered_workspaces": registered_workspaces,
                "kbs": kbs,
                "custom_models": custom_models,
                "upload_history": history,
                "total_purged": total_purged,
                "new_kb": new_kb,
            },
        )

    @app.post("/set-workspace")
    async def api_set_workspace(workspace: str = Form(...)):
        p = Path(workspace).resolve()
        register_workspace(p)
        return RedirectResponse(url="/", status_code=303)

    @app.post("/register-workspace")
    async def api_register_workspace(workspace: str = Form(...)):
        """Register a new workspace (doesn't delete data)."""
        p = Path(workspace).resolve()
        register_workspace(p)
        return RedirectResponse(url="/", status_code=303)

    @app.post("/deregister-workspace")
    async def api_deregister_workspace(confirm_path: str = Form(...)):
        """Deregister workspace (data stays on disk)."""
        try:
            current_ws = get_workspace()
        except Exception:
            raise HTTPException(400, "No workspace configured")
        
        # Verify confirmation path matches
        if Path(confirm_path).resolve() != current_ws:
            raise HTTPException(400, "Confirmation path does not match current workspace")
        
        # Deregister the workspace (data stays on disk)
        deregister_workspace(current_ws)
        
        # Redirect to setup page
        return RedirectResponse(url="/", status_code=303)

    @app.post("/create-kb")
    async def api_create_kb(
        name: str = Form(...),
        model_type: str = Form("e5"),
        model_name: str = Form("e5-small"),
    ):
        ws = get_workspace()
        
        # Validate KB name
        if not name or not name.strip():
            raise HTTPException(400, "KB name cannot be empty")
        
        name = name.strip()
        if not re.match(r"^[a-zA-Z0-9_]{1,64}$", name):
            raise HTTPException(
                400, 
                "Invalid KB name. Use only alphanumeric characters and underscores (max 64 chars)"
            )
        
        # Check if KB already exists
        if name in db.list_kbs(ws):
            raise HTTPException(400, f"Knowledge base '{name}' already exists")
        
        try:
            db.create_kb(ws, name, model_type=model_type, model_name=model_name)
        except ValueError as e:
            raise HTTPException(400, str(e))
        
        # Redirect to home with new_kb parameter to stay on KB tab and highlight
        return RedirectResponse(url=f"/?tab=kbs&new_kb={name}", status_code=303)

    @app.post("/delete-kb")
    async def api_delete_kb(name: str = Form(...)):
        """Delete a knowledge base."""
        ws = get_workspace()
        
        try:
            db.delete_kb(ws, name)
        except FileNotFoundError:
            raise HTTPException(404, f"Knowledge base '{name}' not found")
        except Exception as e:
            raise HTTPException(500, f"Failed to delete KB: {str(e)}")
        
        return RedirectResponse(url="/?tab=kbs", status_code=303)

    @app.post("/upload")
    async def api_upload(
        file: UploadFile = ...,
        kb: str = Form(...),
    ):
        """Upload file and redirect to chunk preview."""
        if not file.filename or not file.filename.lower().endswith((".txt", ".md")):
            raise HTTPException(400, "Only .txt and .md files allowed")

        # Read file content
        content_bytes = await file.read()
        
        # Check file size
        if len(content_bytes) > MAX_FILE_SIZE:
            raise HTTPException(400, f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB")
        
        if len(content_bytes) == 0:
            raise HTTPException(400, "File is empty")
        
        try:
            content = content_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(400, f"Failed to decode file: {str(e)}")
        
        ws = get_workspace()
        ensure_ws(ws)

        # Validate KB exists
        try:
            db.get_connection(ws, kb).close()
        except FileNotFoundError:
            raise HTTPException(404, f"Knowledge base '{kb}' not found")

        # Store in cache with unique key
        import uuid
        cache_key = str(uuid.uuid4())
        _upload_cache[cache_key] = {
            "content": content,
            "filename": file.filename or "upload.txt",
            "kb": kb,
        }

        # Redirect to preview
        return RedirectResponse(url=f"/preview-chunks-ui?cache_key={cache_key}", status_code=303)

    @app.get("/upload-to-kb", response_class=HTMLResponse)
    async def ui_upload_to_kb(request: Request, kb: str):
        """Show upload form for a specific KB."""
        ws = get_workspace()
        
        # Validate KB exists
        try:
            conn = db.get_connection(ws, kb)
            model_type = db.get_kb_metadata(conn, "model_type", "e5")
            model_name = db.get_kb_metadata(conn, "model_name", "e5-small")
            conn.close()
        except FileNotFoundError:
            raise HTTPException(404, f"Knowledge base '{kb}' not found")
        
        return templates.TemplateResponse(
            "upload_to_kb.html",
            {
                "request": request,
                "kb_name": kb,
                "model_type": model_type,
                "model_name": model_name,
            },
        )

    @app.post("/preview-chunks")
    async def api_preview_chunks(
        content: str = Form(...),
        filename: str = Form("upload.txt"),
        max_tokens: int = Form(300),
        merge_threshold: int = Form(50),
    ):
        """API endpoint for chunk preview (returns JSON)."""
        chunks = chunk_text(
            content,
            file=filename,
            max_tokens=max(10, min(800, max_tokens)),
            merge_threshold=max(10, merge_threshold),
        )
        return {"chunks": chunks, "count": len(chunks)}

    @app.get("/preview-chunks-ui", response_class=HTMLResponse)
    @app.post("/preview-chunks-ui", response_class=HTMLResponse)
    async def ui_preview_chunks(
        request: Request,
        cache_key: str = Form(None),
        content: str = Form(None),
        filename: str = Form(None),
        kb: str = Form(None),
        max_tokens: int = Form(300),
        merge_threshold: int = Form(50),
        split_by_headings: str = Form("true"),
        split_by_paragraphs: str = Form("true"),
    ):
        """UI endpoint for chunk preview."""
        # Handle GET with cache_key
        if request.method == "GET":
            cache_key = request.query_params.get("cache_key")
            if cache_key and cache_key in _upload_cache:
                cached = _upload_cache[cache_key]
                content = cached["content"]
                filename = cached["filename"]
                kb = cached["kb"]
            else:
                raise HTTPException(400, "Invalid or expired upload session")
            max_tokens = int(request.query_params.get("max_tokens", 300))
            merge_threshold = int(request.query_params.get("merge_threshold", 50))
            split_by_headings = request.query_params.get("split_by_headings", "true")
            split_by_paragraphs = request.query_params.get("split_by_paragraphs", "true")
        else:
            # POST from re-preview form
            if cache_key and cache_key in _upload_cache:
                cached = _upload_cache[cache_key]
                if not content:
                    content = cached["content"]
                if not filename:
                    filename = cached["filename"]
                if not kb:
                    kb = cached["kb"]

        if not content:
            raise HTTPException(400, "No content provided")

        # Convert string bools
        split_headings = split_by_headings.lower() in ("true", "1", "yes", "on")
        split_paras = split_by_paragraphs.lower() in ("true", "1", "yes", "on")

        # Generate chunks
        chunks = chunk_text(
            content,
            file=filename,
            max_tokens=max(10, min(800, max_tokens)),
            merge_threshold=max(10, merge_threshold),
            split_by_headings=split_headings,
            split_by_paragraphs=split_paras,
        )

        # Add token counts
        from justembed.chunker import _word_tokens
        for chunk in chunks:
            chunk["token_count"] = len(_word_tokens(chunk["text"]))

        # Check if file already exists in KB
        ws = get_workspace()
        file_exists = False
        existing_chunk_count = 0
        try:
            conn = db.get_connection(ws, kb)
            result = conn.execute("SELECT COUNT(*) FROM chunks WHERE file = ?", [filename]).fetchone()
            existing_chunk_count = result[0] if result else 0
            file_exists = existing_chunk_count > 0
            conn.close()
        except Exception:
            pass

        return templates.TemplateResponse(
            "chunk_preview.html",
            {
                "request": request,
                "chunks": chunks,
                "filename": filename,
                "kb": kb,
                "cache_key": cache_key,
                "max_tokens": max_tokens,
                "merge_threshold": merge_threshold,
                "split_by_headings": split_headings,
                "split_by_paragraphs": split_paras,
                "file_exists": file_exists,
                "existing_chunk_count": existing_chunk_count,
            },
        )

    @app.post("/apply-chunking")
    async def api_apply_chunking(
        cache_key: str = Form(...),
        filename: str = Form(...),
        kb: str = Form(...),
        max_tokens: int = Form(300),
        merge_threshold: int = Form(50),
        split_by_headings: str = Form("true"),
        split_by_paragraphs: str = Form("true"),
    ):
        """Apply chunking and save to KB (without embeddings)."""
        # Get content from cache
        if cache_key not in _upload_cache:
            raise HTTPException(400, "Upload session expired. Please upload the file again.")
        
        content = _upload_cache[cache_key]["content"]
        ws = get_workspace()
        
        # Convert string bools
        split_headings = split_by_headings.lower() in ("true", "1", "yes", "on")
        split_paras = split_by_paragraphs.lower() in ("true", "1", "yes", "on")

        # Generate chunks
        chunks = chunk_text(
            content,
            file=filename,
            max_tokens=max(10, min(800, max_tokens)),
            merge_threshold=max(10, merge_threshold),
            split_by_headings=split_headings,
            split_by_paragraphs=split_paras,
        )

        if not chunks:
            raise HTTPException(400, "No chunks produced")

        # Save chunks without embeddings
        try:
            conn = db.get_connection(ws, kb)
        except FileNotFoundError:
            raise HTTPException(404, f"Knowledge base '{kb}' not found")

        db.insert_chunks_without_embeddings(conn, filename, chunks)
        conn.close()

        # Clean up cache
        del _upload_cache[cache_key]

        # Redirect to embed step
        from urllib.parse import urlencode
        params = urlencode({"filename": filename, "kb": kb})
        return RedirectResponse(url=f"/embed?{params}", status_code=303)

    @app.get("/embed", response_class=HTMLResponse)
    async def ui_embed(
        request: Request,
        filename: str,
        kb: str,
    ):
        """Embed chunks (creates background job)."""
        ws = get_workspace()
        
        try:
            conn = db.get_connection(ws, kb)
        except FileNotFoundError:
            raise HTTPException(404, f"Knowledge base '{kb}' not found")

        # Get KB model info
        model_type = db.get_kb_metadata(conn, "model_type", "e5")
        model_name = db.get_kb_metadata(conn, "model_name", "e5-small")

        # Check if there are chunks to embed
        chunks_to_embed = db.get_chunks_without_embeddings(conn, filename)
        conn.close()
        
        if not chunks_to_embed:
            # No chunks to embed, redirect to home
            return RedirectResponse(url="/?tab=kbs", status_code=303)
        
        # Create background job for embedding
        job_manager = JobManager(ws)
        job_id = job_manager.create_job(
            job_type="embed_chunks",
            params={
                "kb_name": kb,
                "filename": filename,
                "model_type": model_type,
                "model_name": model_name,
            },
        )
        
        # Add to upload history
        db.add_upload_history(ws, filename, "kb_upload", kb_name=kb)
        
        # Redirect to job status page
        return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)

    @app.post("/query", response_class=HTMLResponse)
    async def api_query(
        request: Request,
        query: str = Form(...),
        kbs: str = Form("all"),
        top_k: int = Form(5),
        mode: str = Form("retrieve"),
    ):
        if not query.strip():
            raise HTTPException(400, "Query required")

        ws = get_workspace()
        kb_list = [k.strip() for k in kbs.split(",") if k.strip()] if kbs else []
        if not kb_list or (kbs or "").strip().lower() == "all":
            kb_list = db.list_kbs(ws)

        results: List[Dict] = []
        by_kb: List[Dict] = []
        
        for kb_name in kb_list:
            try:
                conn = db.get_connection(ws, kb_name)
            except FileNotFoundError:
                continue
            
            # Get KB model info
            model_type = db.get_kb_metadata(conn, "model_type", "e5")
            model_name = db.get_kb_metadata(conn, "model_name", "e5-small")
            
            # Get appropriate embedder
            emb = get_embedder(model_type=model_type, model_name=model_name)
            q_emb = emb.embed_query(query)
            
            if mode == "count":
                all_matches = db.query_chunks(conn, q_emb, top_k=1000, kb_name=kb_name)
                by_kb.append({"kb": kb_name, "count": len(all_matches), "model": f"{model_type}:{model_name}"})
            else:
                rows = db.query_chunks(conn, q_emb, top_k=top_k, kb_name=kb_name)
                
                # Add explanation data to each result
                for row in rows:
                    row["model"] = f"{model_type}:{model_name}"
                    
                    # Get chunk embedding from database
                    chunk_emb = row.get("embedding", [])
                    
                    # Calculate dimension-level differences for full embedding
                    dim_diffs = []
                    if len(q_emb) == len(chunk_emb):
                        for i, (q_val, c_val) in enumerate(zip(q_emb, chunk_emb)):
                            diff = abs(q_val - c_val)
                            same_sign = (q_val >= 0 and c_val >= 0) or (q_val < 0 and c_val < 0)
                            dim_diffs.append({
                                "dim": i,
                                "query_val": q_val,
                                "chunk_val": c_val,
                                "diff": diff,
                                "same_sign": same_sign
                            })
                        
                        # Sort by difference (descending) to find most divergent
                        sorted_by_diff = sorted(dim_diffs, key=lambda x: x["diff"], reverse=True)
                        most_divergent = sorted_by_diff[:3]
                        
                        # Sort by difference (ascending) to find most aligned
                        most_aligned = sorted(dim_diffs, key=lambda x: x["diff"])[:3]
                    else:
                        most_divergent = []
                        most_aligned = []
                    
                    # Build explanation
                    explanation = {
                        "score": row.get("score", 0.0),
                        "query_embedding_preview": q_emb[:3] if len(q_emb) >= 3 else q_emb,
                        "chunk_embedding_preview": chunk_emb[:3] if len(chunk_emb) >= 3 else chunk_emb,
                        "query_embedding_full": q_emb,
                        "chunk_embedding_full": chunk_emb,
                        "embedding_dim": len(q_emb),
                        "model_type": model_type,
                        "model_name": model_name,
                        "most_divergent": most_divergent,
                        "most_aligned": most_aligned,
                    }
                    
                    # For custom models, add term analysis and vocabulary
                    if model_type == "custom" and hasattr(emb, 'get_top_terms'):
                        explanation["query_terms"] = emb.get_top_terms(query, top_k=3)
                        explanation["chunk_terms"] = emb.get_top_terms(row["text"], top_k=3)
                        
                        # Add vocabulary (feature names) for dimension browser
                        explanation["vocabulary"] = None
                        if hasattr(emb, '_vectorizer') and emb._vectorizer is not None:
                            try:
                                feature_names = emb._vectorizer.get_feature_names_out()
                                if feature_names is not None and len(feature_names) > 0:
                                    explanation["vocabulary"] = feature_names.tolist()
                                    print(f"DEBUG: Loaded {len(explanation['vocabulary'])} vocabulary terms")
                            except Exception as e:
                                print(f"DEBUG: Failed to load vocabulary: {e}")
                                explanation["vocabulary"] = None
                        else:
                            print(f"DEBUG: No vectorizer found for custom model")
                    else:
                        explanation["vocabulary"] = None
                    
                    row["explanation"] = explanation
                
                results.extend(rows)
            conn.close()

        if mode == "count":
            total = sum(r["count"] for r in by_kb)
            return templates.TemplateResponse(
                "query_results.html",
                {
                    "request": request,
                    "mode": "count",
                    "query": query,
                    "total": total,
                    "by_kb": by_kb,
                },
            )
        
        # Sort results by score (highest first) across all KBs
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return templates.TemplateResponse(
            "query_results.html",
            {
                "request": request,
                "mode": "retrieve",
                "query": query,
                "results": results,
                "count": len(results),
            },
        )

    @app.get("/status")
    async def api_status():
        try:
            ws = get_workspace()
        except Exception:
            return {"workspace": None, "kbs": []}
        return {"workspace": str(ws), "kbs": db.list_kbs(ws)}

    @app.get("/train-model", response_class=HTMLResponse)
    async def ui_train_model(request: Request):
        """UI for training custom models."""
        return templates.TemplateResponse("train_model.html", {"request": request})

    @app.post("/train-model")
    async def api_train_model(
        model_name: str = Form(...),
        file: UploadFile = ...,
        embedding_dim: int = Form(128),
        max_features: int = Form(5000),
    ):
        """Train custom model from uploaded file (creates background job)."""
        # Validate model name
        if not model_name or not model_name.replace("_", "").isalnum():
            raise HTTPException(400, "Invalid model name. Use alphanumeric and underscore only.")
        
        # Validate file
        if not file.filename or not file.filename.lower().endswith((".txt", ".md")):
            raise HTTPException(400, "Only .txt and .md files allowed")
        
        # Read file content
        content_bytes = await file.read()
        
        if len(content_bytes) > MAX_FILE_SIZE:
            raise HTTPException(400, f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB")
        
        if len(content_bytes) == 0:
            raise HTTPException(400, "File is empty")
        
        try:
            content = content_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(400, f"Failed to decode file: {str(e)}")
        
        # Split into paragraphs (simple chunking for training)
        # Try double newline first, then single newline, then sentence-based chunking
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        # If no double-newline paragraphs, try single newlines
        if len(paragraphs) < 3:
            paragraphs = [p.strip() for p in content.split("\n") if p.strip() and len(p.strip()) > 50]
        
        # If still too few, split by sentences (periods followed by space/newline)
        if len(paragraphs) < 3:
            import re
            # Split on period + space/newline, keep sentences with 20+ words
            sentences = re.split(r'\.\s+', content)
            paragraphs = [s.strip() + '.' for s in sentences if s.strip() and len(s.split()) >= 20]
        
        # If still too few, chunk by word count (every ~100 words)
        if len(paragraphs) < 3:
            words = content.split()
            chunk_size = max(100, len(words) // 10)  # At least 100 words per chunk, or 10 chunks
            paragraphs = []
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                if chunk.strip():
                    paragraphs.append(chunk.strip())
        
        if len(paragraphs) < 3:
            raise HTTPException(400, f"Could not create enough chunks for training. Got {len(paragraphs)} chunks, need at least 3. Try adding paragraph breaks or providing more text.")
        
        # Create background job
        ws = get_workspace()
        job_manager = JobManager(ws)
        
        job_id = job_manager.create_job(
            job_type="train_model",
            params={
                "corpus": paragraphs,
                "model_name": model_name,
                "embedding_dim": embedding_dim,
                "max_features": max_features,
                "filename": file.filename,
            },
        )
        
        # Add to upload history
        db.add_upload_history(ws, file.filename, "model_training", model_name=model_name)
        
        # Redirect to job status page
        return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)

    @app.get("/train-success", response_class=HTMLResponse)
    async def ui_train_success(
        request: Request,
        model_name: str,
        model_dir: str,
    ):
        """Show training success page."""
        import json
        from pathlib import Path
        
        # Load model config
        config_path = Path(model_dir) / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        return templates.TemplateResponse(
            "train_success.html",
            {
                "request": request,
                "model_name": model_name,
                "model_dir": model_dir,
                "config": config,
            },
        )

    # Job Management Endpoints
    
    @app.get("/jobs", response_class=HTMLResponse)
    async def ui_jobs(request: Request):
        """Show all jobs."""
        ws = get_workspace()
        job_manager = JobManager(ws)
        jobs = job_manager.list_jobs(limit=100)
        stats = job_manager.get_job_stats()
        
        return templates.TemplateResponse(
            "jobs.html",
            {
                "request": request,
                "jobs": jobs,
                "stats": stats,
            },
        )
    
    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def ui_job_status(request: Request, job_id: str):
        """Show job status with auto-refresh."""
        ws = get_workspace()
        job_manager = JobManager(ws)
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(404, "Job not found")
        
        return templates.TemplateResponse(
            "job_status.html",
            {"request": request, "job": job},
        )
    
    @app.get("/api/jobs/{job_id}")
    async def api_job_status(job_id: str):
        """API endpoint for job status (for polling)."""
        ws = get_workspace()
        job_manager = JobManager(ws)
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(404, "Job not found")
        
        return job

    return app
