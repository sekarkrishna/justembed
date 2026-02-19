"""
E5 model loading — extract from bundled tar.gz, cache in ~/.cache/justembed.
"""

import tarfile
from pathlib import Path


def get_cache_dir() -> Path:
    return Path.home() / ".cache" / "justembed"


def get_bundled_path() -> Path:
    return Path(__file__).parent / "e5-small-int8.tar.gz"


def get_model_path(force: bool = False) -> Path:
    """Extract model if needed; return path to e5-small-int8.onnx."""
    cache = get_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    model_path = cache / "e5-small-int8.onnx"

    if not force and model_path.exists():
        return model_path

    bundled = get_bundled_path()
    if not bundled.exists():
        raise FileNotFoundError(
            f"Bundled model not found at {bundled}. Package may be corrupted."
        )

    with tarfile.open(bundled, "r:gz") as tar:
        tar.extractall(path=cache)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model extraction failed. Expected {model_path} after extraction."
        )

    return model_path
