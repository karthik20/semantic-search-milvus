import os
import numpy as np
from typing import List, Literal
from tokenizers import Tokenizer
import onnxruntime as ort


def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    # token_embeddings: [batch, seq, hidden]
    # attention_mask: [batch, seq]
    mask_expanded = np.broadcast_to(attention_mask[:, :, None], token_embeddings.shape).astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


class OnnxSentenceEmbedder:
    """
    ONNX Runtime-based sentence embedder for 'all-MiniLM-L6-v2' (or compatible).
    Expects a model directory containing:
    - model.onnx (or model_quant.onnx)
    - tokenizer.json
    Uses CUDA if available and provider = 'AUTO' or 'CUDA'.
    """

    def __init__(
        self,
        model_dir: str,
        embedding_dim: int = 384,
        provider: Literal["AUTO", "CUDA", "CPU"] = "AUTO",
        onnx_model_filename: str = None,
    ):
        self.model_dir = model_dir
        self.embedding_dim = embedding_dim
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
        self.tokenizer = Tokenizer.from_file(tok_path)

        # Pick ONNX model file
        if onnx_model_filename:
            model_path = os.path.join(model_dir, onnx_model_filename)
        else:
            # Prefer quantized if present
            q = os.path.join(model_dir, "model_quant.onnx")
            d = os.path.join(model_dir, "model.onnx")
            model_path = q if os.path.exists(q) else d
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")

        providers = ["CPUExecutionProvider"]
        if provider in ("AUTO", "CUDA"):
            try:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                # Validate CUDA availability
                ort.InferenceSession(model_path, providers=providers)
            except Exception:
                if provider == "CUDA":
                    raise
                providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.inputs = {i.name: i for i in self.session.get_inputs()}

    def _tokenize(self, texts: List[str], max_length: int = 256):
        enc = self.tokenizer.encode_batch(texts)
        # Truncate/pad to max_length
        input_ids = []
        attention_mask = []
        for e in enc:
            ids = e.ids[:max_length]
            mask = [1] * len(ids)
            if len(ids) < max_length:
                pad_len = max_length - len(ids)
                ids = ids + [0] * pad_len
                mask = mask + [0] * pad_len
            input_ids.append(ids)
            attention_mask.append(mask)
        return np.array(input_ids, dtype=np.int64), np.array(attention_mask, dtype=np.int64)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        input_ids, attention_mask = self._tokenize(texts)
        ort_inputs = {}
        if "input_ids" in self.inputs:
            ort_inputs["input_ids"] = input_ids
        else:
            # Some exported models might use 'input' as the name
            first_name = list(self.inputs.keys())[0]
            ort_inputs[first_name] = input_ids

        if "attention_mask" in self.inputs:
            ort_inputs["attention_mask"] = attention_mask

        outputs = self.session.run(None, ort_inputs)
        # Assume last_hidden_state is the first output
        last_hidden = outputs[0]  # [batch, seq, hidden]
        sentence_embeddings = mean_pooling(last_hidden, attention_mask)
        # Normalize (optional, good for cosine; harmless for L2 distances on unit vectors)
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True) + 1e-12
        sentence_embeddings = sentence_embeddings / norms
        return sentence_embeddings.astype(np.float32).tolist()

    # Convenience methods mirroring typical embeddings interfaces
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]