import os
import numpy as np
from typing import Any, Dict, List, Optional, Literal
from tokenizers import Tokenizer
import onnxruntime as ort
from langchain_core.embeddings import Embeddings

class OnnxEmbeddings(Embeddings):
    """
    LangChain Embeddings implementation using ONNX for all-MiniLM-L6-v2 (or compatible).
    
    This class uses ONNX Runtime to run a previously exported ONNX model.
    Expects a model directory containing:
    - model.onnx
    - tokenizer.json
    
    Uses CUDA if available and provider = 'AUTO' or 'CUDA'.
    """

    def __init__(
        self,
        model_dir: str,
        embedding_dim: int = 384,
        provider: Literal["CPU", "CUDA"] = "CPU",
        max_length: int = 256,
    ):
        """
        Simplified ONNX embeddings for all-MiniLM-L6-v2 only.
        Assumes model.onnx and tokenizer.json in model_dir.
        Only supports CPU or CUDA provider.
        """
        self.model_dir = model_dir
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        # Load tokenizer
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
        self.tokenizer = Tokenizer.from_file(tok_path)

        # Load ONNX model
        model_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")

        # Setup ONNX Runtime session
        providers = ["CPUExecutionProvider"]
        if provider == "CUDA":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
    
    def _tokenize(self, texts: List[str]) -> tuple:
        """Tokenize and pad/truncate for all-MiniLM-L6-v2."""
        batch = self.tokenizer.encode_batch(texts)
        input_ids = [enc.ids[:self.max_length] + [0] * (self.max_length - len(enc.ids)) for enc in batch]
        attention_mask = [[1] * min(len(enc.ids), self.max_length) + [0] * (self.max_length - len(enc.ids)) for enc in batch]
        return np.array(input_ids, dtype=np.int64), np.array(attention_mask, dtype=np.int64)
    
    def _mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Mean pooling to create sentence embeddings."""
        # token_embeddings: [batch, seq_len, hidden_dim]
        # attention_mask: [batch, seq_len]
        mask_expanded = np.broadcast_to(
            attention_mask[:, :, None], token_embeddings.shape
        ).astype(np.float32)
        
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        if not texts:
            print("[DEBUG] No texts provided to embed_documents.")
            return []

        input_ids, attention_mask = self._tokenize(texts)

        # Prepare token_type_ids (all zeros, same shape as input_ids)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # Prepare inputs for the ONNX model
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        outputs = self.session.run(None, ort_inputs)
        last_hidden_states = outputs[0] if outputs else None  # [batch, seq_len, hidden_dim]
        if last_hidden_states is None:
            print("[ERROR] ONNX model did not return outputs.")
            return []
        embeddings = self._mean_pooling(last_hidden_states, attention_mask)
        normalized_embeddings = self._normalize(embeddings)
        return normalized_embeddings.astype(np.float32).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        embeddings = self.embed_documents([text])
        if not embeddings or not embeddings[0]:
            raise ValueError("Failed to generate embedding for query: '{}'".format(text))
        return embeddings[0]
