"""Task 2: Dense Retrieval using General and Scientific Pretrained Bi-Encoders.

Models:
1. General-Purpose: 'sentence-transformers/all-MiniLM-L6-v2' (22M params)
2. Scientific/Biomedical: 'pritamdeka/S-PubMedBert-MS-MARCO' (110M params)

Features:
- Caches document embeddings to disk (data/cache/) to prevent re-computation.
- Vectorized GPU matrix multiplication for batch query retrieval.
- Measures Index Time (sec), Query Latency (ms), and Peak GPU Memory (MB).
- Generates val_general.trec, t2_general.trec, and t2_scientific.trec.
"""

import os
import sys
import time
import json
from typing import Dict, List, Tuple, Any
import torch
from sentence_transformers import SentenceTransformer
from src.data_loader import load_corpus, load_claims, get_gold_evidence
from src.metrics import evaluate_retrieval
from src.trec_utils import write_trec_file

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SRC_DIR)
ROOT_DIR = os.path.dirname(CODE_DIR) if os.path.basename(CODE_DIR) == "code" else CODE_DIR
DATA_DIR = os.path.join(ROOT_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class DenseRetriever:
    """Manages dense bi-encoder indexing, embedding caching, and GPU retrieval."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"\nLoading model '{model_name}' onto {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()

    def build_or_load_index(
        self, corpus: Dict[int, Dict[str, Any]], batch_size: int = 64
    ) -> Tuple[List[int], torch.Tensor, float, float]:
        """Encodes all corpus abstracts or loads from cache.

        Returns:
            doc_ids: List of integer doc_ids
            doc_embeddings: Normalized tensor on GPU of shape (N, dim)
            index_time_sec: Time in seconds
            peak_memory_mb: Peak GPU memory used in MB
        """
        safe_name = self.model_name.replace("/", "_")
        cache_path = os.path.join(CACHE_DIR, f"{safe_name}_corpus_emb.pt")
        doc_ids = list(corpus.keys())
        texts = [corpus[did]["text"] for did in doc_ids]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

        start_time = time.perf_counter()

        if os.path.exists(cache_path):
            print(f"Loading cached document embeddings from {cache_path}...")
            doc_embeddings = torch.load(cache_path, map_location=self.device)
            index_time = 0.0  # loaded from cache
        else:
            print(f"Encoding {len(texts)} abstracts on {self.device} (batch_size={batch_size})...")
            with torch.inference_mode():
                embeddings_np = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device,
                )
            doc_embeddings = embeddings_np.to(self.device)
            torch.save(doc_embeddings.cpu(), cache_path)
            doc_embeddings = doc_embeddings.to(self.device)
            index_time = time.perf_counter() - start_time
            print(f"Indexing complete in {index_time:.2f} seconds. Saved to {cache_path}.")

        peak_mem_mb = 0.0
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        return doc_ids, doc_embeddings, index_time, peak_mem_mb

    def batch_retrieve(
        self,
        claims: List[Dict[str, Any]],
        doc_ids: List[int],
        doc_embeddings: torch.Tensor,
        top_k: int = 100,
        prompt: str = None,
    ) -> Tuple[Dict[int, List[Tuple[int, float]]], float]:
        """Encodes claims and retrieves top_k documents using GPU dot product.

        Returns:
            predictions: Dict mapping claim_id -> list of (doc_id, score)
            query_latency_ms: Average query latency in milliseconds per query
        """
        claim_ids = [int(c["id"]) for c in claims]
        if prompt is None:
            if "bge" in self.model_name.lower():
                prompt = "Represent this sentence for searching relevant passages: "
            else:
                prompt = ""
        claim_texts = [f"{prompt}{c['claim']}" for c in claims]

        start_time = time.perf_counter()

        with torch.inference_mode():
            # 1. Encode queries
            query_embeddings = self.model.encode(
                claim_texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device,
            )  # (num_queries, dim)

            # 2. Similarity matrix: Q @ D.T -> (num_queries, num_docs)
            similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T)

            # 3. Top-k selection directly on GPU
            topk_scores, topk_indices = torch.topk(similarity_matrix, k=top_k, dim=1, largest=True)

        elapsed_time = time.perf_counter() - start_time
        avg_latency_ms = (elapsed_time / len(claims)) * 1000.0

        # Move topk results to CPU
        topk_scores_cpu = topk_scores.cpu().tolist()
        topk_indices_cpu = topk_indices.cpu().tolist()

        predictions = {}
        for idx, cid in enumerate(claim_ids):
            row_doc_ids = [doc_ids[i] for i in topk_indices_cpu[idx]]
            row_scores = topk_scores_cpu[idx]
            predictions[cid] = list(zip(row_doc_ids, row_scores))

        return predictions, avg_latency_ms

