"""
JustEmbed Python API — Programmatic interface for scripts and notebooks.

Usage:
    import justembed as je
    
    je.begin(workspace="/path/to/workspace")
    je.train_model(name="my_model", training_data="/path/to/texts")
    je.create_kb(name="my_kb", model="my_model")
    je.add(kb="my_kb", path="/path/to/docs")
    results = je.query("search query", kb="my_kb")
"""

import json
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress indicator
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')
            self.n = 0
            self.disable = kwargs.get('disable', False)
        
        def update(self, n=1):
            self.n += n
            if not self.disable and self.total > 0:
                pct = int(100 * self.n / self.total)
                print(f"\r{self.desc}: {self.n}/{self.total} ({pct}%)", end='', flush=True)
        
        def close(self):
            if not self.disable:
                print()
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()


# Global state for server management
_active_servers: Dict[int, dict] = {}  # port -> {thread, stop_event, workspace}


def _print_table(headers: List[str], rows: List[List[str]], title: str = None):
    """Print a simple ASCII table."""
    if title:
        print(f"\n{title}")
    
    if not rows:
        print("  (empty)")
        return
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    header_line = "  " + " │ ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("  " + "─" * (len(header_line) - 2))
    
    # Print rows
    for row in rows:
        print("  " + " │ ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    print()


def _find_process_on_port(port: int) -> Optional[int]:
    """Find process ID using the given port."""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port and conn.status == 'LISTEN':
                        return proc.pid
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
    except ImportError:
        pass
    return None


def _kill_process(pid: int) -> bool:
    """Kill process by PID."""
    try:
        import psutil
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=3)
        return True
    except Exception:
        try:
            import os
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            return True
        except Exception:
            return False


def _is_port_available(port: int) -> bool:
    """Check if port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def begin(
    workspace: Union[str, Path],
    port: int = 5424,
    host: str = "127.0.0.1",
    open_browser: bool = False,
    background: bool = False,
) -> int:
    """
    Start the JustEmbed server.
    
    Args:
        workspace: Path to workspace directory (required)
        port: Server port (default 5424)
        host: Bind address (default 127.0.0.1)
        open_browser: Auto-open browser (default False)
        background: Run in background (default False, blocks in foreground)
    
    Returns:
        int: The actual port number used (may differ from requested if port was in use)
    
    Examples:
        >>> import justembed as je
        >>> je.begin(workspace="~/my_docs")
        >>> port = je.begin(workspace="/data/kb", port=8080, background=True)
        >>> print(f"Server running on port {port}")
    """
    from justembed.config import register_workspace, ensure_workspace_structure
    import uvicorn
    
    # Register workspace (creates structure if needed, doesn't delete data)
    ws_path = Path(workspace).expanduser().resolve()
    register_workspace(ws_path)
    
    # Handle port conflicts
    original_port = port
    while not _is_port_available(port):
        pid = _find_process_on_port(port)
        if pid:
            # Try to kill our own process
            try:
                import psutil
                proc = psutil.Process(pid)
                if 'justembed' in proc.name().lower() or 'python' in proc.name().lower():
                    print(f"Port {port} in use by PID {pid}, attempting to reclaim...")
                    if _kill_process(pid):
                        time.sleep(1)
                        if _is_port_available(port):
                            print(f"✓ Reclaimed port {port}")
                            break
            except Exception:
                pass
        
        # Port still in use, try next one
        port += 1
        if port > original_port + 10:
            raise RuntimeError(f"Could not find available port (tried {original_port}-{port})")
    
    if port != original_port:
        print(f"Port {original_port} in use, using port {port} instead")
    
    # Open browser if requested
    if open_browser:
        import webbrowser
        import threading
        def open_after_delay():
            time.sleep(1.5)
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=open_after_delay, daemon=True).start()
    
    # Start server
    from justembed.app import create_app
    app = create_app()
    
    print(f"\n✓ JustEmbed server starting...")
    print(f"  Workspace: {ws_path}")
    print(f"  URL: http://{host}:{port}")
    print(f"  Press Ctrl+C to stop\n")
    
    if background:
        # Background mode (for notebooks)
        import threading as thread_module
        stop_event = thread_module.Event()
        
        def run_server():
            config = uvicorn.Config(app, host=host, port=port, log_level="warning")
            server = uvicorn.Server(config)
            
            # Store server reference
            _active_servers[port]['server'] = server
            
            # Run until stop event is set
            try:
                server.run()
            except Exception as e:
                if not stop_event.is_set():
                    print(f"Server error: {e}")
        
        thread = thread_module.Thread(target=run_server, daemon=True)
        
        # Store server info
        _active_servers[port] = {
            'thread': thread,
            'stop_event': stop_event,
            'workspace': str(ws_path),
            'host': host,
        }
        
        thread.start()
        time.sleep(2)  # Give server time to start
        print(f"✓ Server running in background on port {port}")
        return port
    else:
        # Foreground mode (blocks)
        uvicorn.run(app, host=host, port=port)
        return port


def terminate(port: Optional[int] = None) -> None:
    """
    Terminate JustEmbed server(s).
    
    Args:
        port: Specific port to terminate. If None, terminates all active servers.
    
    Examples:
        >>> import justembed as je
        >>> port = je.begin(workspace="~/docs", background=True)
        >>> # Later...
        >>> je.terminate(port)  # Terminate specific server
        >>> 
        >>> # Or terminate all servers
        >>> je.terminate()
    """
    global _active_servers
    
    if port is not None:
        # Terminate specific server
        if port not in _active_servers:
            print(f"⚠️  No active server found on port {port}")
            return
        
        server_info = _active_servers[port]
        
        # Set stop event to signal server to stop
        if 'stop_event' in server_info:
            server_info['stop_event'].set()
        
        # If we have a server object, try to shut it down gracefully
        if 'server' in server_info:
            try:
                server_info['server'].should_exit = True
            except Exception:
                pass
        
        # Remove from active servers
        del _active_servers[port]
        print(f"✓ Server on port {port} terminated")
    
    else:
        # Terminate all servers
        if not _active_servers:
            print("⚠️  No active servers to terminate")
            return
        
        ports = list(_active_servers.keys())
        for p in ports:
            terminate(p)
        
        if len(ports) > 1:
            print(f"✓ All servers terminated ({len(ports)} total)")


def list_servers(silent: bool = False) -> List[Dict[str, Any]]:
    """
    List all active JustEmbed servers.
    
    Args:
        silent: Disable table output (default False)
    
    Returns:
        List of server dicts with: {port, workspace, host, status}
    
    Example:
        >>> import justembed as je
        >>> servers = je.list_servers()
    """
    servers = []
    
    for port, info in _active_servers.items():
        status = "running" if info['thread'].is_alive() else "stopped"
        servers.append({
            "port": port,
            "workspace": info['workspace'],
            "host": info['host'],
            "status": status,
        })
    
    if not silent:
        if servers:
            rows = []
            for s in servers:
                url = f"http://{s['host']}:{s['port']}"
                rows.append([str(s['port']), s['workspace'], url, s['status']])
            _print_table(["Port", "Workspace", "URL", "Status"], rows, title="Active Servers:")
        else:
            print("\nNo active servers")
    
    return servers


def train_model(
    name: str,
    training_data: Union[str, Path, List[str]],
    embedding_dim: int = 128,
    max_features: int = 5000,
    overwrite: bool = False,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Train a custom embedding model.
    
    Args:
        name: Model name (alphanumeric and underscore only)
        training_data: Path to file/folder, or list of texts
        embedding_dim: Embedding dimension (default 128)
        max_features: Max TF-IDF features (default 5000)
        overwrite: Overwrite if model exists (default False)
        silent: Disable progress output (default False)
    
    Returns:
        Dict with model info: {name, path, embedding_dim, num_chunks, created_at}
    
    Examples:
        >>> je.train_model("medical", "/data/medical_texts")
        >>> je.train_model("legal", ["doc1 text...", "doc2 text..."])
    """
    from justembed.config import get_workspace, get_custom_models_dir
    from justembed.training.trainer import CustomModelTrainer
    from justembed.chunker import chunk_text
    
    # Validate model name
    if not name or not name.replace("_", "").isalnum():
        raise ValueError("Model name must be alphanumeric with underscores only")
    
    # Check if model exists
    models_dir = get_custom_models_dir()
    model_path = models_dir / name
    if model_path.exists() and not overwrite:
        raise ValueError(f"Model '{name}' already exists. Use overwrite=True to replace.")
    
    # Prepare training corpus
    if not silent:
        print(f"\n📚 Preparing training data for model '{name}'...")
    
    corpus = []
    
    if isinstance(training_data, (str, Path)):
        path = Path(training_data).expanduser().resolve()
        
        if path.is_file():
            # Single file
            content = path.read_text(encoding='utf-8', errors='replace')
            # Chunk the content
            chunks = chunk_text(content, file=path.name, max_tokens=300)
            corpus = [c['text'] for c in chunks]
            if not silent:
                print(f"  ✓ Loaded {len(corpus)} chunks from {path.name}")
        
        elif path.is_dir():
            # Directory of files
            files = list(path.glob("*.txt")) + list(path.glob("*.md"))
            if not files:
                raise ValueError(f"No .txt or .md files found in {path}")
            
            pbar = tqdm(files, desc="  Loading files", disable=silent)
            for file in pbar:
                content = file.read_text(encoding='utf-8', errors='replace')
                chunks = chunk_text(content, file=file.name, max_tokens=300)
                corpus.extend([c['text'] for c in chunks])
                pbar.set_postfix({"chunks": len(corpus)})
            pbar.close()
            
            if not silent:
                print(f"  ✓ Loaded {len(corpus)} chunks from {len(files)} files")
        else:
            raise ValueError(f"Path not found: {path}")
    
    elif isinstance(training_data, list):
        # List of texts
        if not silent:
            print(f"  ✓ Using {len(training_data)} provided texts")
        corpus = training_data
    
    else:
        raise TypeError("training_data must be path (str/Path) or list of texts")
    
    # Validate corpus size
    if len(corpus) < 3:
        raise ValueError(f"Need at least 3 chunks for training, got {len(corpus)}")
    
    # Train model
    if not silent:
        print(f"\n🔧 Training model...")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Max features: {max_features}")
        print(f"  Corpus size: {len(corpus)} chunks")
    
    trainer = CustomModelTrainer()
    
    try:
        model_dir = trainer.train(
            corpus=corpus,
            model_name=name,
            embedding_dim=embedding_dim,
            max_features=max_features,
        )
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")
    
    # Load config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    if not silent:
        print(f"\n✓ Model '{name}' trained successfully!")
        print(f"  Location: {model_dir}")
        print(f"  Embedding dim: {config['embedding_dim']}")
        print(f"  Training chunks: {config['corpus_stats']['num_chunks']}")
    
    return {
        "name": name,
        "path": str(model_dir),
        "embedding_dim": config['embedding_dim'],
        "num_chunks": config['corpus_stats']['num_chunks'],
        "created_at": config.get('created_at', ''),
    }


def create_kb(
    name: str,
    model: str = "default",
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Create a new knowledge base.
    
    Args:
        name: KB name (alphanumeric and underscore only, max 64 chars)
        model: Model name - "default" (e5-small) or custom model name
        silent: Disable output (default False)
    
    Returns:
        Dict with KB info: {name, model_type, model_name}
    
    Examples:
        >>> je.create_kb("docs")  # Uses default model
        >>> je.create_kb("medical_kb", model="medical_model")
    """
    from justembed.config import get_workspace
    from justembed import db
    
    ws = get_workspace()
    
    # Validate KB name
    if not name or not name.replace("_", "").isalnum() or len(name) > 64:
        raise ValueError("KB name must be alphanumeric with underscores, max 64 chars")
    
    # Check if KB exists
    if name in db.list_kbs(ws):
        raise ValueError(f"Knowledge base '{name}' already exists")
    
    # Determine model type
    if model in ("default", "e5-small"):
        model_type = "e5"
        model_name = "e5-small"
    else:
        # Check if custom model exists
        from justembed.config import get_custom_models_dir
        models_dir = get_custom_models_dir()
        if not (models_dir / model).exists():
            raise ValueError(f"Custom model '{model}' not found. Train it first with train_model()")
        model_type = "custom"
        model_name = model
    
    # Create KB
    db.create_kb(ws, name, model_type=model_type, model_name=model_name)
    
    if not silent:
        model_display = "default (e5-small)" if model_type == "e5" else f"custom:{model_name}"
        print(f"✓ Knowledge base '{name}' created with model '{model_display}'")
    
    return {
        "name": name,
        "model_type": model_type,
        "model_name": model_name,
    }


def add(
    kb: str,
    path: Union[str, Path] = None,
    text: str = None,
    paths: List[Union[str, Path]] = None,
    documents: List[Dict[str, str]] = None,
    max_tokens: int = 300,
    merge_threshold: int = 50,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Add documents to a knowledge base.
    
    Args:
        kb: Knowledge base name
        path: Single file or folder path
        text: Direct text content
        paths: List of file paths
        documents: List of dicts with 'text' and optional 'filename'
        max_tokens: Max tokens per chunk (default 300)
        merge_threshold: Merge threshold (default 50)
        silent: Disable progress output (default False)
    
    Returns:
        Dict with stats: {kb, files_added, chunks_added, model}
    
    Examples:
        >>> je.add(kb="docs", path="/data/manual.txt")
        >>> je.add(kb="docs", path="/data/docs_folder")
        >>> je.add(kb="docs", text="Direct text content...")
        >>> je.add(kb="docs", paths=["file1.txt", "file2.md"])
    """
    from justembed.config import get_workspace
    from justembed import db
    from justembed.chunker import chunk_text
    from justembed.embedder import E5Embedder, CustomEmbedder
    
    ws = get_workspace()
    
    # Validate KB exists
    try:
        conn = db.get_connection(ws, kb)
    except FileNotFoundError:
        raise ValueError(f"Knowledge base '{kb}' not found. Create it first with create_kb()")
    
    # Get KB model info
    model_type = db.get_kb_metadata(conn, "model_type", "e5")
    model_name = db.get_kb_metadata(conn, "model_name", "e5-small")
    
    # Get embedder
    if model_type == "e5":
        embedder = E5Embedder()
    else:
        embedder = CustomEmbedder(model_name)
    
    # Collect files to process
    files_to_process = []
    
    if path:
        p = Path(path).expanduser().resolve()
        if p.is_file():
            files_to_process.append(p)
        elif p.is_dir():
            files_to_process.extend(p.glob("*.txt"))
            files_to_process.extend(p.glob("*.md"))
        else:
            raise ValueError(f"Path not found: {p}")
    
    elif paths:
        for p in paths:
            p = Path(p).expanduser().resolve()
            if not p.exists():
                raise ValueError(f"File not found: {p}")
            files_to_process.append(p)
    
    elif text:
        # Direct text - create temporary entry
        files_to_process = [("direct_text", text)]
    
    elif documents:
        # List of documents with text and filename
        for doc in documents:
            if 'text' not in doc:
                raise ValueError("Each document must have 'text' field")
            filename = doc.get('filename', f'doc_{len(files_to_process)}.txt')
            files_to_process.append((filename, doc['text']))
    
    else:
        raise ValueError("Must provide one of: path, text, paths, or documents")
    
    # Process files
    total_chunks = 0
    total_files = 0
    
    if not silent:
        print(f"\n📄 Adding documents to '{kb}'...")
    
    pbar = tqdm(files_to_process, desc="  Processing files", disable=silent)
    
    for item in pbar:
        if isinstance(item, tuple):
            # Direct text
            filename, content = item
        else:
            # File path
            filename = item.name
            content = item.read_text(encoding='utf-8', errors='replace')
        
        # Chunk text
        chunks = chunk_text(
            content,
            file=filename,
            max_tokens=max_tokens,
            merge_threshold=merge_threshold,
        )
        
        if not chunks:
            continue
        
        # Embed chunks
        texts = [c['text'] for c in chunks]
        embeddings = embedder.embed(texts)
        
        # Insert into DB
        db.insert_chunks(conn, filename, chunks, embeddings)
        
        total_chunks += len(chunks)
        total_files += 1
        pbar.set_postfix({"chunks": total_chunks})
    
    pbar.close()
    conn.close()
    
    # Add to history
    if total_files > 0:
        db.add_upload_history(ws, f"{total_files} files", "kb_upload", kb_name=kb)
    
    if not silent:
        model_display = "default" if model_type == "e5" else f"custom:{model_name}"
        print(f"✓ Added {total_chunks} chunks from {total_files} files to '{kb}'")
        print(f"  Model: {model_display}")
    
    return {
        "kb": kb,
        "files_added": total_files,
        "chunks_added": total_chunks,
        "model": f"{model_type}:{model_name}",
    }


def query(
    text: str,
    kb: Union[str, List[str]] = "all",
    top_k: int = 5,
    min_score: float = 0.0,
    silent: bool = False,
) -> List[Dict[str, Any]]:
    """
    Query knowledge bases.
    
    Args:
        text: Query text
        kb: KB name, list of KB names, or "all" (default "all")
        top_k: Number of results per KB (default 5)
        min_score: Minimum similarity score (default 0.0)
        silent: Disable output (default False)
    
    Returns:
        List of result dicts with: {text, score, kb, file, chunk_index, model}
        Results are sorted by score (highest first) across all KBs
    
    Examples:
        >>> results = je.query("machine learning")
        >>> results = je.query("python tutorial", kb="docs")
        >>> results = je.query("medical terms", kb=["medical_kb", "research_kb"])
    """
    from justembed.config import get_workspace
    from justembed import db
    from justembed.embedder import E5Embedder, CustomEmbedder
    
    if not text.strip():
        raise ValueError("Query text cannot be empty")
    
    ws = get_workspace()
    
    # Determine KB list
    if kb == "all" or not kb:
        kb_list = db.list_kbs(ws)
    elif isinstance(kb, str):
        kb_list = [kb]
    else:
        kb_list = list(kb)
    
    if not kb_list:
        if not silent:
            print("⚠️  No knowledge bases found")
        return []
    
    # Query each KB
    all_results = []
    
    for kb_name in kb_list:
        try:
            conn = db.get_connection(ws, kb_name)
        except FileNotFoundError:
            if not silent:
                print(f"⚠️  KB '{kb_name}' not found, skipping")
            continue
        
        # Get KB model info
        model_type = db.get_kb_metadata(conn, "model_type", "e5")
        model_name = db.get_kb_metadata(conn, "model_name", "e5-small")
        
        # Get embedder
        if model_type == "e5":
            embedder = E5Embedder()
        else:
            embedder = CustomEmbedder(model_name)
        
        # Embed query
        q_emb = embedder.embed_query(text)
        
        # Query chunks
        results = db.query_chunks(conn, q_emb, top_k=top_k, kb_name=kb_name)
        
        # Add model info and filter by score
        for r in results:
            r['model'] = f"{model_type}:{model_name}"
            if r['score'] >= min_score:
                all_results.append(r)
        
        conn.close()
    
    # Sort by score (highest first) across all KBs
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Print results
    if not silent:
        print(f"\n🔍 Query: \"{text}\"")
        print(f"   Found {len(all_results)} results (sorted by relevance)")
        
        if all_results:
            print()
            for i, r in enumerate(all_results[:top_k], 1):
                print(f"{i}. [{r['kb']}] {r['file']} (score: {r['score']:.3f}) {r['model']}")
                preview = r['text'][:150].replace('\n', ' ')
                if len(r['text']) > 150:
                    preview += "..."
                print(f"   {preview}\n")
    
    return all_results


def list_kbs(silent: bool = False) -> List[Dict[str, Any]]:
    """
    List all knowledge bases.
    
    Args:
        silent: Disable table output (default False)
    
    Returns:
        List of KB dicts with: {name, model_type, model_name, chunks}
    
    Example:
        >>> kbs = je.list_kbs()
    """
    from justembed.config import get_workspace
    from justembed import db
    
    ws = get_workspace()
    kb_names = db.list_kbs(ws)
    
    kbs = []
    for kb_name in kb_names:
        try:
            conn = db.get_connection(ws, kb_name)
            model_type = db.get_kb_metadata(conn, "model_type", "e5")
            model_name = db.get_kb_metadata(conn, "model_name", "e5-small")
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
            conn.close()
            
            kbs.append({
                "name": kb_name,
                "model_type": model_type,
                "model_name": model_name,
                "chunks": chunk_count,
            })
        except Exception:
            continue
    
    if not silent:
        if kbs:
            rows = []
            for kb in kbs:
                model_display = "default" if kb['model_type'] == "e5" else f"custom:{kb['model_name']}"
                rows.append([kb['name'], model_display, str(kb['chunks'])])
            _print_table(["Name", "Model", "Chunks"], rows, title="Knowledge Bases:")
        else:
            print("\nNo knowledge bases found")
    
    return kbs


def list_models(silent: bool = False) -> List[Dict[str, Any]]:
    """
    List all models (inbuilt + custom).
    
    Args:
        silent: Disable table output (default False)
    
    Returns:
        List of model dicts with: {name, type, embedding_dim, ...}
    
    Example:
        >>> models = je.list_models()
    """
    from justembed.config import get_custom_models_dir
    
    models = []
    
    # Add inbuilt model
    models.append({
        "name": "default",
        "type": "inbuilt",
        "embedding_dim": 384,
        "model": "e5-small",
    })
    
    # Add custom models
    try:
        models_dir = get_custom_models_dir()
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                config_path = model_dir / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        models.append({
                            "name": model_dir.name,
                            "type": "custom",
                            "embedding_dim": config.get("embedding_dim", 128),
                            "num_chunks": config.get("corpus_stats", {}).get("num_chunks", 0),
                            "created_at": config.get("created_at", ""),
                        })
                    except Exception:
                        pass
    except Exception:
        pass
    
    if not silent:
        if models:
            rows = []
            for m in models:
                if m['type'] == 'inbuilt':
                    rows.append([m['name'], m['type'], str(m['embedding_dim']), "e5-small"])
                else:
                    rows.append([m['name'], m['type'], str(m['embedding_dim']), str(m.get('num_chunks', 0))])
            _print_table(["Name", "Type", "Dim", "Info"], rows, title="Models:")
        else:
            print("\nNo models found")
    
    return models


def list_history(silent: bool = False) -> List[Dict[str, Any]]:
    """
    List upload history.
    
    Args:
        silent: Disable table output (default False)
    
    Returns:
        List of history dicts with: {type, filename, kb_name, model_name, uploaded_at}
    
    Example:
        >>> history = je.list_history()
    """
    from justembed.config import get_workspace
    from justembed import db
    
    ws = get_workspace()
    history, total_purged = db.get_upload_history(ws)
    
    if not silent:
        if history:
            rows = []
            for h in history:
                type_display = "KB Upload" if h['type'] == 'kb_upload' else "Model Training"
                target = h.get('kb_name') or h.get('model_name') or '-'
                timestamp = str(h['uploaded_at'])[:19]  # Trim microseconds
                rows.append([type_display, h['filename'], target, timestamp])
            _print_table(["Type", "File", "Target", "Timestamp"], rows, title="Upload History:")
            
            if total_purged > 0:
                print(f"  ({total_purged} older entries purged)\n")
        else:
            print("\nNo upload history")
    
    return history


def delete_kb(name: str, confirm: bool = False) -> None:
    """
    Delete a knowledge base.
    
    Args:
        name: KB name to delete
        confirm: Must be True to confirm deletion
    
    Example:
        >>> je.delete_kb("old_kb", confirm=True)
    """
    if not confirm:
        raise ValueError("Must set confirm=True to delete knowledge base")
    
    from justembed.config import get_workspace
    from justembed import db
    
    ws = get_workspace()
    
    try:
        db.delete_kb(ws, name)
        print(f"✓ Knowledge base '{name}' deleted")
    except FileNotFoundError:
        raise ValueError(f"Knowledge base '{name}' not found")


def delete_model(name: str, confirm: bool = False) -> None:
    """
    Delete a custom model.
    
    Args:
        name: Model name to delete
        confirm: Must be True to confirm deletion
    
    Example:
        >>> je.delete_model("old_model", confirm=True)
    """
    if not confirm:
        raise ValueError("Must set confirm=True to delete model")
    
    if name in ("default", "e5-small"):
        raise ValueError("Cannot delete inbuilt model")
    
    from justembed.config import get_custom_models_dir
    import shutil
    
    models_dir = get_custom_models_dir()
    model_path = models_dir / name
    
    if not model_path.exists():
        raise ValueError(f"Model '{name}' not found")
    
    shutil.rmtree(model_path)
    print(f"✓ Model '{name}' deleted")


def register_workspace(
    path: Union[str, Path],
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Register a workspace for access. Does NOT delete any existing data.
    Creates folder structure if needed. User can zip/share and re-register later.
    
    Args:
        path: Path to workspace directory
        silent: Disable output (default False)
    
    Returns:
        Dict with workspace info: {path, registered}
    
    Examples:
        >>> je.register_workspace("~/my_docs")
        >>> je.register_workspace("/shared/team_kb")  # After unzipping shared folder
    """
    from justembed.config import register_workspace as _register
    
    ws_path = Path(path).expanduser().resolve()
    _register(ws_path)
    
    if not silent:
        print(f"✓ Workspace registered: {ws_path}")
        print(f"  You can now use this workspace with begin()")
    
    return {
        "path": str(ws_path),
        "registered": True,
    }


def deregister_workspace(
    path: Union[str, Path],
    confirm: bool = False,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Deregister a workspace. Does NOT delete any data on disk.
    Just removes from registry. User can zip/share folder and re-register later.
    
    Args:
        path: Path to workspace directory
        confirm: Must be True to confirm deregistration
        silent: Disable output (default False)
    
    Returns:
        Dict with workspace info: {path, deregistered}
    
    Examples:
        >>> je.deregister_workspace("~/old_docs", confirm=True)
        >>> # Data still exists on disk, just not accessible via API
        >>> # Can zip and share: tar -czf my_kb.tar.gz ~/old_docs
        >>> # Recipient can unzip and: je.register_workspace("~/received_kb")
    """
    if not confirm:
        raise ValueError("Must set confirm=True to deregister workspace")
    
    from justembed.config import deregister_workspace as _deregister
    
    ws_path = Path(path).expanduser().resolve()
    _deregister(ws_path)
    
    if not silent:
        print(f"✓ Workspace deregistered: {ws_path}")
        print(f"  Data still exists on disk - not deleted")
        print(f"  You can zip/share this folder and re-register it later")
    
    return {
        "path": str(ws_path),
        "deregistered": True,
    }


def list_workspaces(silent: bool = False) -> List[Dict[str, Any]]:
    """
    List all registered workspaces.
    
    Args:
        silent: Disable table output (default False)
    
    Returns:
        List of workspace dicts with: {path, exists, current}
    
    Example:
        >>> workspaces = je.list_workspaces()
    """
    from justembed.config import list_registered_workspaces, get_workspace
    
    registered = list_registered_workspaces()
    
    # Get current workspace
    try:
        current_ws = str(get_workspace())
    except FileNotFoundError:
        current_ws = None
    
    workspaces = []
    for ws_path in registered:
        p = Path(ws_path)
        workspaces.append({
            "path": ws_path,
            "exists": p.exists(),
            "current": ws_path == current_ws,
        })
    
    if not silent:
        if workspaces:
            rows = []
            for ws in workspaces:
                status = "✓" if ws['exists'] else "✗"
                current = "→" if ws['current'] else " "
                rows.append([current, ws['path'], status])
            _print_table(["", "Path", "Exists"], rows, title="Registered Workspaces:")
        else:
            print("\nNo registered workspaces")
    
    return workspaces
