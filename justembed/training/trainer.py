"""
Custom model trainer — TF-IDF→MLP pipeline.
"""

from pathlib import Path
from typing import Any, Optional, List, Callable
import json
from datetime import datetime


class CustomModelTrainer:
    """Train custom TF-IDF→MLP models from text corpus."""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize trainer.
        
        Args:
            models_dir: Directory to save models. If None, uses workspace/custom_models.
        """
        if models_dir is None:
            from justembed.config import get_custom_models_dir
            self._cache_dir = get_custom_models_dir()
        else:
            self._cache_dir = Path(models_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        corpus: List[str],
        model_name: str,
        embedding_dim: int = 128,
        max_features: int = 5000,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        """
        Train custom model and export to ONNX.
        
        Args:
            corpus: List of text chunks for training
            model_name: Name for the custom model
            embedding_dim: Output embedding dimension (64-256)
            max_features: Max TF-IDF features (1000-10000)
            progress_callback: Optional callback(progress_percent, message) for progress updates
        
        Returns:
            Path to saved model directory
        
        Raises:
            ValueError: If corpus is too small or parameters invalid
        """
        def update_progress(percent: int, message: str):
            if progress_callback:
                progress_callback(percent, message)
        
        # Validate inputs
        update_progress(5, "Validating inputs...")
        self._validate_inputs(corpus, model_name, embedding_dim, max_features)
        
        # Step 1: Build TF-IDF vectorizer
        update_progress(15, "Building TF-IDF vectorizer...")
        vectorizer = self._build_tfidf(corpus, max_features)
        
        # Step 2: Transform corpus to TF-IDF vectors
        update_progress(35, "Transforming corpus to TF-IDF vectors...")
        X = vectorizer.transform(corpus)
        
        # Step 3: Train MLP
        update_progress(50, "Training neural network (this may take a while)...")
        mlp = self._train_mlp(X, embedding_dim)
        
        # Step 4: Export to ONNX
        update_progress(85, "Exporting model to ONNX format...")
        model_dir = self._export_onnx(vectorizer, mlp, model_name)
        
        # Step 5: Save metadata
        update_progress(95, "Saving model metadata...")
        self._save_metadata(model_dir, corpus, model_name, embedding_dim, max_features)
        
        update_progress(100, "Training complete!")
        return model_dir
    
    def _validate_inputs(
        self,
        corpus: List[str],
        model_name: str,
        embedding_dim: int,
        max_features: int,
    ) -> None:
        """Validate training inputs."""
        # Check corpus size
        if len(corpus) < 3:
            raise ValueError(f"Corpus too small: {len(corpus)} chunks (minimum 3)")
        
        if len(corpus) > 1000:
            raise ValueError(f"Corpus too large: {len(corpus)} chunks (maximum 1000)")
        
        # Check word count - reduced minimum for better usability
        total_words = sum(len(text.split()) for text in corpus)
        if total_words < 500:
            raise ValueError(f"Corpus too small: {total_words} words (minimum 500)")
        
        # Check model name
        if not model_name or not model_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid model name: {model_name} (use alphanumeric + underscore)")
        
        # Check embedding dimension
        if not 64 <= embedding_dim <= 256:
            raise ValueError(f"Invalid embedding_dim: {embedding_dim} (must be 64-256)")
        
        # Check max features
        if not 1000 <= max_features <= 10000:
            raise ValueError(f"Invalid max_features: {max_features} (must be 1000-10000)")
    
    def _build_tfidf(self, corpus: List[str], max_features: int):
        """Build TF-IDF vectorizer from corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams for synonym learning
            min_df=1,  # Minimum document frequency (keep rare terms)
            max_df=0.95,  # Maximum document frequency (remove common terms)
            sublinear_tf=True,  # Use log(1 + tf) scaling
            norm='l2',  # L2 normalization
            lowercase=True,  # Convert to lowercase
            strip_accents=None,  # Required for ONNX conversion
            stop_words=None,  # Keep all words (domain-specific)
        )
        
        # Fit vectorizer on corpus
        vectorizer.fit(corpus)
        
        return vectorizer
    
    def _train_mlp(self, X, embedding_dim: int):
        """Train MLP to compress TF-IDF to dense embeddings."""
        from sklearn.neural_network import MLPRegressor
        import numpy as np
        
        X_dense = X.toarray()  # Convert sparse to dense
        n_samples, n_features = X_dense.shape
        
        # Create target: random projection to embedding_dim
        # This gives us a stable target that has the right dimensionality
        np.random.seed(42)
        projection_matrix = np.random.randn(n_features, embedding_dim) / np.sqrt(n_features)
        X_target = X_dense @ projection_matrix
        
        # Normalize target
        X_target = X_target / (np.linalg.norm(X_target, axis=1, keepdims=True) + 1e-9)
        
        # Train MLP to predict these projections
        # Architecture: input_dim → 512 → 256 → embedding_dim (output)
        mlp = MLPRegressor(
            hidden_layer_sizes=(512, 256),  # Hidden layers
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization
            batch_size=min(32, n_samples // 2),  # Adaptive batch size
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4,
            random_state=42,
            verbose=False,
        )
        
        mlp.fit(X_dense, X_target)
        
        return mlp
    
    def _export_onnx(self, vectorizer, mlp, model_name: str) -> Path:
        """Export TF-IDF + MLP to ONNX format."""
        from sklearn.pipeline import Pipeline
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import StringTensorType
        import pickle
        
        # Create pipeline: TF-IDF → MLP
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('mlp', mlp),
        ])
        
        # Export to ONNX
        # Input: string tensor (batch of text)
        # Output: float tensor (batch of embeddings)
        onnx_model = to_onnx(
            pipeline,
            initial_types=[('input', StringTensorType([None, 1]))],
            target_opset=12,
        )
        
        # Save to custom_models directory
        model_dir = self._cache_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "model.onnx"
        with open(model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # Save vectorizer separately for term analysis
        vectorizer_path = model_dir / "vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        
        return model_dir
    
    def _save_metadata(
        self,
        model_dir: Path,
        corpus: List[str],
        model_name: str,
        embedding_dim: int,
        max_features: int,
    ) -> None:
        """Save model metadata to config.json."""
        # Calculate corpus stats
        num_chunks = len(corpus)
        num_words = sum(len(text.split()) for text in corpus)
        unique_words = len(set(" ".join(corpus).split()))
        
        # Create metadata
        metadata = {
            "model_name": model_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "embedding_dim": embedding_dim,
            "max_features": max_features,
            "corpus_stats": {
                "num_chunks": num_chunks,
                "num_words": num_words,
                "num_unique_terms": unique_words,
            },
            "training_params": {
                "hidden_layers": [512, 256],
                "activation": "relu",
                "solver": "adam",
            },
            "version": "0.2.0",
        }
        
        # Save to file
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(metadata, f, indent=2)
