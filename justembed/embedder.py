"""
Embedding engine — pluggable interface, E5 implementation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, List

import numpy as np


class Embedder(ABC):
    """Pluggable embedder interface."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return L2-normalized embeddings for texts."""
        ...


class E5Embedder(Embedder):
    """E5-small ONNX embedder (384-dim)."""

    def __init__(self):
        import onnxruntime as ort

        from justembed.models.core.model_loader import get_model_path

        model_path = get_model_path()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            import psutil
            num_cores = psutil.cpu_count(logical=False) or 4
        except ImportError:
            num_cores = 4

        sess_options.intra_op_num_threads = num_cores
        sess_options.inter_op_num_threads = num_cores

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._max_length = 512
        self._tokenizer = None
        self._use_real_tokenizer = False

        # Try tokenizer (check cache dir first, then package)
        tokenizer_paths = [
            Path.home() / ".cache" / "justembed" / "tokenizer.json",
            Path(__file__).parent / "models" / "core" / "tokenizer.json",
            Path(__file__).resolve().parent.parent / "models" / "core" / "tokenizer.json",
        ]
        
        for tokenizer_path in tokenizer_paths:
            if tokenizer_path.exists():
                try:
                    from tokenizers import Tokenizer
                    self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
                    self._use_real_tokenizer = True
                    break
                except Exception:
                    continue

    def _tokenize(self, text: str, prefix: str = "passage: ") -> dict[str, Any]:
        text = prefix + text

        if self._use_real_tokenizer and self._tokenizer:
            enc = self._tokenizer.encode(text, add_special_tokens=True)
            enc.truncate(self._max_length)
            enc.pad(self._max_length, pad_id=0)
            return {
                "input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            }

        # Placeholder tokenization
        if len(text) > self._max_length:
            text = text[: self._max_length]
        tokens = [101]
        for char in text[: self._max_length - 2]:
            tokens.append(ord(char) % 1000 + 1000)
        tokens.append(102)
        attention_mask = [1] * len(tokens)
        while len(tokens) < self._max_length:
            tokens.append(0)
            attention_mask.append(0)
        return {
            "input_ids": np.array([tokens], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
        }

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed document chunks (passage: prefix)."""
        return self._embed_with_prefix(texts, "passage: ")

    def embed_query(self, text: str) -> List[float]:
        """Embed query (query: prefix). Returns single 384-dim vector."""
        return self._embed_with_prefix([text], "query: ")[0]

    def _embed_with_prefix(self, texts: List[str], prefix: str) -> List[List[float]]:
        result: List[List[float]] = []
        for t in texts:
            inp = self._tokenize(t, prefix=prefix)
            out = self._session.run(
                None,
                {
                    "input_ids": inp["input_ids"],
                    "attention_mask": inp["attention_mask"],
                },
            )
            # E5: last_hidden_state shape (1, seq_len, 384) -> mean pool over seq
            last_hidden = np.array(out[0], dtype=np.float32).squeeze(0)  # (seq_len, 384)
            mask = np.array(inp["attention_mask"], dtype=np.float32).squeeze(0)  # (seq_len,)
            mask_exp = np.expand_dims(mask, -1)
            masked = last_hidden * mask_exp
            sum_emb = np.sum(masked, axis=0)
            sum_mask = max(np.sum(mask), 1e-9)
            emb = sum_emb / sum_mask  # (384,)
            emb = self._normalize(emb)
            result.append(emb.tolist())
        return result


class CustomEmbedder(Embedder):
    """Custom TF-IDF→MLP embedder (ONNX inference)."""
    
    def __init__(self, model_name: str, models_dir: Optional[Path] = None):
        """
        Initialize custom embedder.
        
        Args:
            model_name: Name of the custom model to load
            models_dir: Directory containing models. If None, uses workspace/custom_models.
        
        Raises:
            FileNotFoundError: If model not found
            ValueError: If model config is invalid
        """
        import onnxruntime as ort
        import json
        import pickle
        
        # Locate model directory
        if models_dir is None:
            from justembed.config import get_custom_models_dir
            models_dir = get_custom_models_dir()
        
        model_dir = Path(models_dir) / model_name
        model_path = model_dir / "model.onnx"
        config_path = model_dir / "config.json"
        vectorizer_path = model_dir / "vectorizer.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Custom model not found: {model_name}\n"
                f"Expected at: {model_path}\n"
                f"Train a model first using CustomModelTrainer."
            )
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {config_path}\n"
                f"Model may be corrupted. Try retraining."
            )
        
        # Load metadata
        with open(config_path) as f:
            self._config = json.load(f)
        
        self._model_name = model_name
        self._embedding_dim = self._config["embedding_dim"]
        
        # Load vectorizer if available (for term analysis)
        self._vectorizer = None
        if vectorizer_path.exists():
            try:
                with open(vectorizer_path, 'rb') as f:
                    self._vectorizer = pickle.load(f)
            except Exception:
                pass  # Vectorizer is optional
        
        # Load ONNX model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            import psutil
            num_cores = psutil.cpu_count(logical=False) or 4
        except ImportError:
            num_cores = 4
        
        sess_options.intra_op_num_threads = num_cores
        sess_options.inter_op_num_threads = num_cores
        
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
    
    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        """L2 normalize embedding."""
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts using custom model.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors (each is embedding_dim floats)
        """
        result = []
        for text in texts:
            # Run ONNX inference
            # Input: string tensor shape (1, 1)
            inputs = {"input": np.array([[text]])}
            outputs = self._session.run(None, inputs)
            
            # Extract embedding from output
            # ONNX output shape is (embedding_dim, 1) due to sklearn-onnx quirk
            # We need to reshape to (embedding_dim,)
            embedding = outputs[0].flatten()  # Flatten to 1D array
            
            # Verify shape
            if len(embedding) != self._embedding_dim:
                raise ValueError(
                    f"Unexpected embedding dimension: got {len(embedding)}, "
                    f"expected {self._embedding_dim}"
                )
            
            # L2 normalize
            embedding = self._normalize(embedding)
            
            result.append(embedding.tolist())
        
        return result
    
    def embed_query(self, text: str) -> list[float]:
        """
        Embed query (same as document for custom models).
        
        Args:
            text: Query text
        
        Returns:
            Single embedding vector (embedding_dim floats)
        """
        return self.embed([text])[0]
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name
    
    @property
    def config(self) -> dict:
        """Return model configuration."""
        return self._config.copy()
    
    def get_top_terms(self, text: str, top_k: int = 3) -> List[tuple]:
        """
        Get top contributing TF-IDF terms for a text.
        
        Args:
            text: Input text to analyze
            top_k: Number of top terms to return
        
        Returns:
            List of (term, score) tuples, sorted by score descending
        """
        if not self._vectorizer:
            return []
        
        try:
            # Transform text to TF-IDF vector
            tfidf_matrix = self._vectorizer.transform([text])
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get feature names
            feature_names = self._vectorizer.get_feature_names_out()
            
            # Get top terms
            top_indices = tfidf_scores.argsort()[-top_k:][::-1]
            return [(feature_names[i], float(tfidf_scores[i])) 
                    for i in top_indices if tfidf_scores[i] > 0]
        except Exception:
            return []
