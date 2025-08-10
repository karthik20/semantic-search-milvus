"""
ONNX embeddings using fastembed library specifically for all-MiniLM-L6-v2.
"""
import os
import numpy as np
from typing import List
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

class LocalMiniLMEmbeddings(Embeddings):
    """
    Embeddings class using fastembed's TextEmbedding for all-MiniLM-L6-v2.
    
    This class uses the local all-MiniLM-L6-v2 ONNX model from the specified model directory.
    """
    
    def __init__(self, model_dir: str):
        """Initialize with the path to the model directory."""
        # Define paths to the model files
        model_file = os.path.join(model_dir, "model.onnx")
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        
        # Validate that model files exist
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"ONNX model not found at {model_file}")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")
            
        print(f"Using local ONNX model: {model_file}")
        
        # For local models, we'll use fastembed with the "sentence-transformers/all-MiniLM-L6-v2" model
        # This ensures we're using the local ONNX file but with the correct model architecture
        self.embedding_model = TextEmbedding(
            "sentence-transformers/all-MiniLM-L6-v2",  # Model architecture identifier
            cache_dir=os.path.dirname(model_dir),      # Cache in parent directory
            use_onnx=True,                             # Use ONNX runtime
            check_same_parameters=False                # Don't verify parameters (using local files)
        )
        
        # all-MiniLM-L6-v2 has a fixed embedding dimension of 384
        self.embedding_dim = 384
        
        # Verify the model works by embedding a test string
        self._verify_model()
        
    def _verify_model(self):
        """Verify the model works by embedding a test string."""
        try:
            test_embedding = next(self.embedding_model.embed(["test"]))
            if len(test_embedding) != self.embedding_dim:
                print(f"Warning: Expected embedding dimension {self.embedding_dim}, but got {len(test_embedding)}")
                self.embedding_dim = len(test_embedding)
        except Exception as e:
            raise RuntimeError(f"Error verifying model: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        if not texts:
            return []
        
        # fastembed returns a generator, so convert to list
        embeddings = list(self.embedding_model.embed(texts))
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise ValueError(f"Failed to generate embedding for query: '{text}'")
        return embeddings[0]
