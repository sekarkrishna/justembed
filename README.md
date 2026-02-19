# JustEmbed

**Your First Step Into Semantic Search**

Experience embeddings hands-on. No cloud accounts, no setup complexity, no commitment. Just your laptop and your curiosity.

[![PyPI version](https://badge.fury.io/py/justembed.svg)](https://pypi.org/project/justembed/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author**: Krishnamoorthy Sankaran  
**Email**: krishnamoorthy.sankaran@sekrad.org  
**GitHub**: https://github.com/sekarkrishna/justembed  
**PyPI**: https://pypi.org/project/justembed/

---

## What is JustEmbed?

JustEmbed is a focused tool for **semantic search** - understanding meaning, not just matching keywords. It's designed as your entry point into the embedding ecosystem, letting you experience how semantic search works before committing to cloud platforms or production tools.

### For Non-Technical Users

Upload your documents through a web interface and search by meaning. No coding required, no technical knowledge needed. See exactly how your text is processed and understand what's happening at each step.

### For Developers

A simple Python API (`import justembed as je`) that lets you experiment with embeddings locally. Build confidence with semantic search concepts before moving to production vector databases.

---

## Quick Start

### Installation

```bash
pip install justembed
```

### Web Interface

```bash
justembed begin --workspace ~/my_documents
```

Open http://localhost:5424 in your browser.

### Python API

```python
import justembed as je

je.begin(workspace="~/docs")
je.create_kb("my_kb")
je.add(kb="my_kb", path="document.txt")
results = je.query("search term", kb="my_kb")
```

---

## Understanding Semantic Search

Traditional keyword search looks for exact word matches. Semantic search understands meaning.

**Example**: Imagine a document with these paragraphs:

1. "Volcanoes erupt with molten lava at temperatures exceeding 1000°C..."
2. "Industrial smelting uses high-temperature furnaces above 800°C..."
3. "Igloos are dome-shaped shelters built from compressed snow..."
4. "Icebergs float in cold ocean waters at sub-zero temperatures..."

Search for **"hot"**:
- Traditional search: No results (word "hot" doesn't appear)
- Semantic search: Returns paragraphs 1 & 2 (understands heat/temperature relationship)

This is what JustEmbed lets you experience.

---

## Core Concepts

### 1. Chunking
Documents are broken into smaller pieces (chunks) for efficient searching. JustEmbed's UI shows you exactly how your text will be chunked before processing.

### 2. Embedding
Each chunk is converted to a list of numbers (an embedding) that represents its meaning. Similar meanings have similar numbers.

### 3. Searching
When you search, your query is converted to an embedding and compared to all chunk embeddings. Results are ranked by similarity (0.0-1.0 score).

---

## Complete API Reference

### Workspace Management

```python
# Start workspace
je.begin(workspace="~/my_docs", port=5424)

# Register existing workspace
je.register_workspace("~/shared_workspace")

# List workspaces
workspaces = je.list_workspaces()

# Deregister (data stays on disk)
je.deregister_workspace("~/old_workspace", confirm=True)

# Stop server
je.terminate()
```

### Knowledge Bases

```python
# Create with default model
je.create_kb("general_kb")

# Create with custom model
je.create_kb("medical_kb", model="medical_v1")

# List all KBs
kbs = je.list_kbs()

# Delete KB
je.delete_kb("old_kb", confirm=True)
```

### Adding Documents

```python
# From file
je.add(kb="my_kb", path="document.txt")

# From text
je.add(kb="my_kb", text="Your content...")

# With chunking options
je.add(
    kb="my_kb",
    path="document.txt",
    max_tokens=300,
    merge_threshold=50,
)
```

### Searching

```python
# Basic search
results = je.query("search term", kb="my_kb")

# Search all KBs
results = je.query("search term", kb="all")

# Advanced options
results = je.query(
    text="search term",
    kb="my_kb",
    top_k=10,
    min_score=0.5
)

# Results structure
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text']}")
    print(f"File: {result['file']}")
    print(f"KB: {result['kb']}")
```

### Custom Model Training

```python
# Train from file
je.train_model(
    name="medical_v1",
    training_data="medical_textbook.txt",
    embedding_dim=128,
    max_features=5000
)

# Train from text
je.train_model(
    name="legal_v1",
    training_data=["Your training corpus..."],
    embedding_dim=128
)

# List models
models = je.list_models()
```

---

## Key Features

### Domain-Specific Models

Train models that understand your domain's vocabulary:

```python
# Medical domain
medical_text = """
Pyrexia, commonly known as fever, is elevated body temperature.
Renal function refers to kidney performance.
A UTI affects the bladder and kidneys.
"""

je.train_model("medical_v1", training_data=[medical_text])
je.create_kb("medical_kb", model="medical_v1")

# Now "fever" finds "pyrexia", "kidney" finds "renal"
```

### Multiple Knowledge Bases

Organize by topic, each with its own model:

```python
je.create_kb("medical_kb", model="medical_v1")
je.create_kb("legal_kb", model="legal_v1")
je.create_kb("general_kb")  # Uses default E5-Small model
```

### Workspace Sharing

Share by zipping the workspace folder:

```python
# Create and populate
je.begin(workspace="~/shared_kb")
je.create_kb("team_kb")
je.add(kb="team_kb", path="docs.txt")

# Zip ~/shared_kb and share

# Recipient registers and uses
je.register_workspace("~/received_kb")
je.begin(workspace="~/received_kb")
results = je.query("search", kb="team_kb")
```

---

## Architecture

```
User Interface (Web UI / Python API)
           ↓
    FastAPI Server
           ↓
Embedder Layer (E5-Small / Custom Models)
           ↓
Storage Layer (DuckDB / File System)
```

### Design Decisions

**Offline-First**: Everything runs locally. No API keys, no cloud dependencies, no internet after installation.

**ONNX Models**: Portable, CPU-friendly, small size (~8-15 MB). Works on any platform.

**DuckDB Storage**: Embedded database, no separate server. Fast columnar storage.

**Deterministic Chunking**: Rule-based, predictable. Same input always produces same chunks.

**Privacy**: Your data never leaves your machine. No telemetry, no tracking.

---

## Requirements

- Python 3.8+
- 500 MB disk space
- 1 GB RAM
- CPU (no GPU required)
- No internet (after installation)

---

## Guarantees

**Technical**:
- Deterministic (same input → same output)
- No hallucinations (only returns your text)
- Offline (works without internet)
- Private (data never leaves your machine)
- No tracking or telemetry

**File System**:
- Writes only to workspace and `~/.cache/justembed/`
- Reads only files you upload
- Never deletes files outside workspace

---

## License

MIT License

---

## Author

**Krishnamoorthy Sankaran**

- Email: krishnamoorthy.sankaran@sekrad.org
- GitHub: https://github.com/sekarkrishna/justembed
- PyPI: https://pypi.org/project/justembed/

---

## Support

- Issues: https://github.com/sekarkrishna/justembed/issues
- Discussions: https://github.com/sekarkrishna/justembed/discussions
- Email: krishnamoorthy.sankaran@sekrad.org

---

## Citation

```bibtex
@software{justembed2026,
  title = {JustEmbed: Your First Step Into Semantic Search},
  author = {Sankaran, Krishnamoorthy},
  year = {2026},
  url = {https://github.com/sekarkrishna/justembed}
}
```

---

## Acknowledgments

- E5-Small model: Microsoft Research
- ONNX Runtime: Microsoft
- FastAPI: Sebastián Ramírez
- DuckDB: DuckDB Labs
- scikit-learn: scikit-learn developers

---

**JustEmbed** - Start here. Build confidence. Graduate to production tools when ready.
