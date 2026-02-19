"""
Deterministic chunking engine — word tokens, structure-aware.
"""

import re
from typing import Any, List, Dict

# Word token: split on whitespace and punctuation
_WORD_TOKEN_RE = re.compile(r"\b\w+\b|[^\w\s]")


def _word_tokens(text: str) -> List[str]:
    """Split text into word tokens (deterministic)."""
    return _WORD_TOKEN_RE.findall(text)


def chunk_text(
    text: str,
    file: str = "",
    max_tokens: int = 300,
    merge_threshold: int = 50,
    split_by_headings: bool = True,
    split_by_paragraphs: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chunk text deterministically.

    Args:
        text: Raw text
        file: Source filename
        max_tokens: Max tokens per chunk (10-800)
        merge_threshold: Merge chunks smaller than this
        split_by_headings: Split on # headings
        split_by_paragraphs: Split on blank lines

    Returns:
        List of {"file", "chunk_index", "text"}
    """
    max_tokens = max(10, min(800, max_tokens))
    merge_threshold = max(10, merge_threshold)

    if not text or not text.strip():
        return []

    blocks: List[str] = []

    if split_by_headings or split_by_paragraphs:
        # Split by paragraphs (blank lines)
        paras = re.split(r"\n\s*\n", text)
        for p in paras:
            p = p.strip()
            if not p:
                continue
            if split_by_headings and p.startswith("#"):
                # Split headings into separate blocks
                parts = re.split(r"(?=^#{1,6}\s)", p, flags=re.MULTILINE)
                for part in parts:
                    part = part.strip()
                    if part:
                        blocks.append(part)
            else:
                blocks.append(p)
    else:
        blocks = [text] if text.strip() else []

    if not blocks:
        blocks = [text]

    chunks: List[str] = []
    for block in blocks:
        tokens = _word_tokens(block)
        if not tokens:
            continue
        if len(tokens) <= max_tokens:
            chunks.append(block)
            continue
        # Split by max_tokens
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text_str = _reconstruct(chunk_tokens, block)
            chunks.append(chunk_text_str)
            start = end

    # Merge small chunks
    merged: List[str] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        current_tokens = _word_tokens(current)
        while i + 1 < len(chunks) and len(current_tokens) < merge_threshold:
            next_chunk = chunks[i + 1]
            next_tokens = _word_tokens(next_chunk)
            combined = (current + "\n\n" + next_chunk).strip()
            combined_tokens = _word_tokens(combined)
            if len(combined_tokens) <= max_tokens:
                current = combined
                current_tokens = combined_tokens
                i += 1
            else:
                break
        merged.append(current)
        i += 1

    return [
        {"file": file, "chunk_index": idx, "text": t}
        for idx, t in enumerate(merged)
    ][:500]  # Cap chunks per file


def _reconstruct(tokens: List[str], original: str) -> str:
    """Reconstruct text from tokens (approximate)."""
    return " ".join(tokens)
