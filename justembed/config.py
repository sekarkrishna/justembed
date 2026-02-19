"""
Configuration and workspace handling.
"""

import json
from pathlib import Path
from typing import List, Union

_CONFIG_PATH = Path.home() / ".justembed" / "config.json"


def _load_config() -> dict:
    """Load config file, return empty dict if not exists."""
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, KeyError):
            return {}
    return {}


def _save_config(config: dict) -> None:
    """Save config to file."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2))


def get_workspace() -> Path:
    """Get current workspace path. Raises if not set."""
    config = _load_config()
    workspace_str = config.get("workspace", "")
    if workspace_str and workspace_str.strip():  # Check for non-empty string
        p = Path(workspace_str)
        if p.exists():
            return p
    # No workspace configured - will trigger setup page
    raise FileNotFoundError("Workspace not configured")


def get_custom_models_dir() -> Path:
    """Get custom models directory within workspace."""
    workspace = get_workspace()
    models_dir = workspace / "custom_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_dropin_models_dir() -> Path:
    """Get drop-in models directory within workspace."""
    workspace = get_workspace()
    models_dir = workspace / "drop-in_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def set_workspace(path: Union[str, Path]) -> None:
    """Set workspace path and persist to config."""
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    config = _load_config()
    config["workspace"] = str(p)
    _save_config(config)


def register_workspace(path: Union[str, Path]) -> None:
    """
    Register a workspace for access. Creates structure if needed.
    Does NOT delete any existing data - just makes it accessible.
    """
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    
    # Add to registered workspaces list
    config = _load_config()
    registered = config.get("registered_workspaces", [])
    
    path_str = str(p)
    if path_str not in registered:
        registered.append(path_str)
        config["registered_workspaces"] = registered
    
    # Set as current workspace
    config["workspace"] = path_str
    _save_config(config)
    
    # Ensure structure exists
    ensure_workspace_structure(p)


def deregister_workspace(path: Union[str, Path]) -> None:
    """
    Deregister a workspace. Does NOT delete any data on disk.
    Just removes it from the registry so it's not accessible.
    User can zip/share the folder and re-register it later.
    """
    p = Path(path).expanduser().resolve()
    path_str = str(p)
    
    config = _load_config()
    registered = config.get("registered_workspaces", [])
    
    if path_str in registered:
        registered.remove(path_str)
        config["registered_workspaces"] = registered
        
        # If this was the current workspace, clear it completely
        if config.get("workspace") == path_str:
            config.pop("workspace", None)  # Remove the key entirely
        
        _save_config(config)


def list_registered_workspaces() -> List[str]:
    """List all registered workspace paths."""
    config = _load_config()
    return config.get("registered_workspaces", [])


def ensure_workspace_structure(workspace: Path) -> None:
    """Create kb, custom_models, and drop-in_models subfolders in workspace."""
    (workspace / "kb").mkdir(parents=True, exist_ok=True)
    (workspace / "custom_models").mkdir(parents=True, exist_ok=True)
    (workspace / "drop-in_models").mkdir(parents=True, exist_ok=True)


def _config_file() -> Path:
    """Get config file path."""
    return _CONFIG_PATH


def ensure_workspace_structure(workspace: Path) -> None:
    """Create kb, custom_models, and drop-in_models subfolders in workspace."""
    (workspace / "kb").mkdir(parents=True, exist_ok=True)
    (workspace / "custom_models").mkdir(parents=True, exist_ok=True)
    (workspace / "drop-in_models").mkdir(parents=True, exist_ok=True)


def _config_file() -> Path:
    """Get config file path."""
    return _CONFIG_PATH
