# Medical Embedding Demo

This folder contains example text for demonstrating how JustEmbed chunking and embeddings work, especially with **custom models** and **E5**.

## Two files: training vs KB content

| File | Use for | Chunks (defaults) |
|------|---------|-------------------|
| **medical_embedding_demo.txt** | **Training** the custom ONNX model (vocabulary and concepts) | 7 chunks |
| **medical_kb_documents.txt** | **Adding to a KB** (documents to search over) | 4 chunks |

**Workflow:** Train a custom model on `medical_embedding_demo.txt` → Create a KB with that model → Add `medical_kb_documents.txt` to the KB → Run queries.

---

## File: `medical_embedding_demo.txt` (training only)

- **Chunking defaults:** Max tokens 300, Merge threshold 50, Split by headings and paragraphs.
- **Result:** 7 chunks (each between ~70 and ~110 tokens). Headings are kept with their first paragraph.
- **Concepts:** Fever, pyrexia, hyperthermia, neutropenia, febrile neutropenia, paracetamol, acetaminophen, antipyretic, analgesia.

Use this file **only to train** the custom model (Web UI: Train Custom Model, or API: `je.train_model("medical_demo", "path/to/medical_embedding_demo.txt")`). Do not add it to the KB if you want a clear split between training data and searchable documents.

---

## File: `medical_kb_documents.txt` (add to KB)

- **Result:** 4 chunks (patient case, febrile neutropenia protocol, discharge summary, pharmacy note).
- Same chunking defaults; uses overlapping medical terms (fever, paracetamol, neutropenia, UTI, antipyretic) so the custom model retrieves well.

Use this file to **add to a KB** after creating it (with E5 or with the custom model trained on `medical_embedding_demo.txt`). This is the content users search over.

---

## Suggested Test Queries

### Queries that work well with **custom model** (terms appear in the text)

| Query | Expected chunk(s) | Why |
|-------|-------------------|-----|
| `fever` | Chunks 1, 2, 4, 5, 6, 7 | Central topic |
| `pyrexia` | Chunks 1, 7 | Synonym for fever in text |
| `paracetamol` | Chunks 1, 2, 4, 5, 6, 7 | Drug name repeated |
| `acetaminophen` | Chunks 1, 4, 5, 6, 7 | Same drug, different name |
| `neutropenia` | Chunks 3, 4, 7 | Main topic of section |
| `febrile neutropenia` | Chunks 3, 4, 7 | Key phrase |
| `antipyretic` | Chunks 1, 2, 5, 7 | Fever-reducing drug |
| `hot` | Chunk 1 | Word appears in “feel hot, flushed” |

### Queries that work better with **E5** (semantic)

| Query | Expected | Why |
|-------|----------|-----|
| `temperature regulation` | Chunk 2 | E5 understands “body temperature” / “hypothalamus” |
| `low white blood cells` | Chunks 3, 4 | E5 maps to “neutrophils” / “neutropenia” |
| `pain relief medicine` | Chunks 5, 6 | E5 maps to “analgesic” / “paracetamol” |
| `drug for fever` | Chunks 1, 2, 5, 6 | Semantic link to antipyretics |

---

## How to run the demo

### Option A: Web UI

1. Start: `justembed begin --workspace ~/justembed_demo`
2. **Train** a custom model: Upload `medical_embedding_demo.txt` → Train (e.g. name “medical_demo”).
3. Create a KB (e.g. “medical_kb”) with the custom model “medical_demo” (or use E5).
4. **Add to KB:** Upload `medical_kb_documents.txt` → Preview chunks (4 chunks) → Apply chunking → Embed.
5. In Query, try: `fever`, `paracetamol`, `neutropenia`, `febrile neutropenia`, `UTI`.

### Option B: Python API

```python
import justembed as je

je.register_workspace("~/justembed_demo")
je.begin(workspace="~/justembed_demo", background=True)

# Train custom model on the training text (vocabulary)
je.train_model("medical_demo", "path/to/justembed/examples/medical_embedding_demo.txt")

# Create KB and add the documents you want to search (not the training file)
je.create_kb("medical_kb", model="medical_demo")
je.add(kb="medical_kb", path="path/to/justembed/examples/medical_kb_documents.txt")

# Query
results = je.query("paracetamol", kb="medical_kb", top_k=5)
for r in results:
    print(f"{r['score']:.3f} | {r['text'][:80]}...")
```

---

## Chunking check (defaults)

With **max_tokens=300**, **merge_threshold=50**, and **split by headings and paragraphs**:

- Short blocks (e.g. a heading alone) are merged with the next block until the combined size is ≥ 50 tokens or would exceed 300.
- Paragraphs under 300 tokens stay as one chunk.
- You get 7 chunks, each well under 300 tokens, so the preview matches how the document is indexed.

This structure makes it easy to show which chunk answers which query.
