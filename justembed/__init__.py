"""
JustEmbed 0.1.1 — LENS (Language Embedder With No Synthesizer)

Offline-first semantic search for everyday laptops.

Usage:
    import justembed as je
    
    # Register workspace (doesn't delete data, just makes it accessible)
    je.register_workspace("/path/to/workspace")
    
    # Start server
    je.begin(workspace="/path/to/workspace")
    
    # Train custom model
    je.train_model(name="my_model", training_data="/path/to/texts")
    
    # Create knowledge base
    je.create_kb(name="my_kb", model="my_model")
    
    # Add documents
    je.add(kb="my_kb", path="/path/to/docs")
    
    # Query
    results = je.query("search query", kb="my_kb")
    
    # List resources
    je.list_kbs()
    je.list_models()
    je.list_workspaces()
"""

__version__ = "0.1.1a7"

# Import API functions
from justembed.api import (
    begin,
    terminate,
    list_servers,
    train_model,
    create_kb,
    add,
    query,
    list_kbs,
    list_models,
    list_history,
    delete_kb,
    delete_model,
    register_workspace,
    deregister_workspace,
    list_workspaces,
)

# Import config utilities (for advanced users)
from justembed.config import get_workspace

# Import app factory (for advanced users)
from justembed.app import create_app

__all__ = [
    # Core API
    "begin",
    "terminate",
    "list_servers",
    "train_model",
    "create_kb",
    "add",
    "query",
    "list_kbs",
    "list_models",
    "list_history",
    "delete_kb",
    "delete_model",
    # Workspace management
    "register_workspace",
    "deregister_workspace",
    "list_workspaces",
    # Advanced
    "get_workspace",
    "create_app",
]
