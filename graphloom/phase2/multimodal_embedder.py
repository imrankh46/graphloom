"""
Multimodal Embedder for GraphLoom Phase 2.

This module provides unified multimodal embedding using Qwen3-VL-Embedding
for computing semantic similarity between queries and triples.
"""

from typing import Union, List, Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np


class BaseMultimodalEmbedder(ABC):
    """Abstract base class for multimodal embedders."""
    
    @abstractmethod
    def embed_query(
        self,
        image: Union[str, Path, np.ndarray, None],
        question: str,
    ) -> np.ndarray:
        """
        Embed a multimodal query (image + question).
        
        Args:
            image: Image path, URL, numpy array, or None for text-only
            question: The question text
            
        Returns:
            Query embedding vector
        """
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding vector
        """
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple text strings in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors (N x embedding_dim)
        """
        pass


class Qwen3VLEmbedder(BaseMultimodalEmbedder):
    """
    Multimodal embedder using Qwen3-VL-Embedding.
    
    Projects both interleaved image-text inputs and pure text into
    a shared semantic space for unified cross-modal retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        device: str = "auto",
        embedding_dim: int = 1536,
        max_length: int = 512,
    ):
        """
        Initialize the Qwen3-VL embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on
            embedding_dim: Output embedding dimension
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
    
    def _load_model(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self.device)
            self._model.eval()
        except ImportError as e:
            raise ImportError(
                "Please install transformers and torch: "
                "pip install transformers torch"
            ) from e
    
    def _mean_pooling(self, model_output, attention_mask) -> np.ndarray:
        """Apply mean pooling to get sentence embeddings."""
        import torch
        
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return (sum_embeddings / sum_mask).cpu().numpy()
    
    def embed_query(
        self,
        image: Union[str, Path, np.ndarray, None],
        question: str,
    ) -> np.ndarray:
        """
        Embed a multimodal query.
        
        For this simplified version, we embed the question text.
        Full multimodal embedding would require vision-language model.
        
        Args:
            image: Image (currently unused in text-only mode)
            question: The question text
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(question)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding vector
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple text strings in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors (N x embedding_dim)
        """
        self._load_model()
        
        import torch
        
        inputs = self._tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)
        
        return embeddings


class SentenceTransformerEmbedder(BaseMultimodalEmbedder):
    """
    Embedder using sentence-transformers library.
    
    A simpler alternative that works well for text-only retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
    ):
        """
        Initialize the sentence-transformer embedder.
        
        Args:
            model_name: Model name from sentence-transformers
            device: Device to run model on
        """
        self.model_name = model_name
        self.device = device
        self._model = None
    
    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError as e:
            raise ImportError(
                "Please install sentence-transformers: "
                "pip install sentence-transformers"
            ) from e
    
    def embed_query(
        self,
        image: Union[str, Path, np.ndarray, None],
        question: str,
    ) -> np.ndarray:
        """Embed a multimodal query (text-only for this embedder)."""
        return self.embed_text(question)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple text strings in batch."""
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)


class MockMultimodalEmbedder(BaseMultimodalEmbedder):
    """
    Mock embedder for testing without model dependencies.
    
    Generates deterministic embeddings based on text hash.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize mock embedder.
        
        Args:
            embedding_dim: Dimension of output embeddings
        """
        self.embedding_dim = embedding_dim
    
    def _hash_to_embedding(self, text: str) -> np.ndarray:
        """Generate a deterministic embedding from text hash."""
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def embed_query(
        self,
        image: Union[str, Path, np.ndarray, None],
        question: str,
    ) -> np.ndarray:
        """Generate mock query embedding."""
        return self._hash_to_embedding(question)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate mock text embedding."""
        return self._hash_to_embedding(text)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings for multiple texts."""
        return np.array([self._hash_to_embedding(t) for t in texts])


def create_multimodal_embedder(
    embedder_type: str = "sentence_transformer",
    **kwargs: Any,
) -> BaseMultimodalEmbedder:
    """
    Factory function to create a multimodal embedder.
    
    Args:
        embedder_type: Type of embedder ('qwen3vl', 'sentence_transformer', 'mock')
        **kwargs: Arguments passed to embedder constructor
        
    Returns:
        Configured multimodal embedder instance
    """
    embedders = {
        "qwen3vl": Qwen3VLEmbedder,
        "sentence_transformer": SentenceTransformerEmbedder,
        "mock": MockMultimodalEmbedder,
    }
    
    if embedder_type not in embedders:
        raise ValueError(f"Unknown embedder type: {embedder_type}. "
                        f"Available: {list(embedders.keys())}")
    
    return embedders[embedder_type](**kwargs)
