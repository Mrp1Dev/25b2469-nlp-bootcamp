"""BM25 Retriever implementation for SciFact.

Supports:
- Flexible tokenization (standard words, compound preservation, stopword filtering)
- BM25Okapi and BM25Plus variants
- Hyperparameter tuning for k1 (term saturation) and b (length normalization)
"""

import re
from typing import Dict, List, Tuple, Set, Optional, Callable
from rank_bm25 import BM25Okapi, BM25Plus
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def tokenize_simple(text: str) -> List[str]:
    """Basic lowercased alphanumeric tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def tokenize_with_compounds(text: str) -> List[str]:
    """Tokenization that preserves hyphenated scientific compounds (e.g., 'anti-inflammatory', 'COVID-19')."""
    # Matches words with optional internal hyphens/dots
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*\b", text.lower())
    return tokens


def tokenize_clean(text: str, remove_stopwords: bool = True, preserve_compounds: bool = True) -> List[str]:
    """Full configurable tokenizer.
    
    Args:
        text: Input string
        remove_stopwords: Whether to filter common English stopwords
        preserve_compounds: Whether to preserve hyphenated terms
    """
    if preserve_compounds:
        tokens = tokenize_with_compounds(text)
    else:
        tokens = tokenize_simple(text)
        
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]
        
    return tokens


class BM25Retriever:
    """Retriever wrapping rank_bm25 with corpus mapping and batch retrieval."""

    def __init__(
        self,
        corpus: Dict[int, Dict[str, any]],
        k1: float = 1.2,
        b: float = 0.75,
        method: str = "okapi",
        tokenize_fn: Optional[Callable[[str], List[str]]] = None,
    ):
        self.doc_ids = list(corpus.keys())
        self.k1 = k1
        self.b = b
        self.method = method.lower()
        self.tokenize_fn = tokenize_fn or (lambda text: tokenize_clean(text, remove_stopwords=True, preserve_compounds=True))

        # Tokenize all documents in corpus (combining title and abstract)
        self.tokenized_corpus = [
            self.tokenize_fn(corpus[doc_id]["text"])
            for doc_id in self.doc_ids
        ]

        if self.method == "plus":
            self.model = BM25Plus(self.tokenized_corpus, k1=self.k1, b=self.b)
        else:
            self.model = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Retrieves top_k documents for a single query text.

        Returns:
            List of (doc_id, score) sorted descending by score.
        """
        query_tokens = self.tokenize_fn(query)
        if not query_tokens:
            # Fallback if query has no tokens
            return [(doc_id, 0.0) for doc_id in self.doc_ids[:top_k]]

        scores = self.model.get_scores(query_tokens)

        # Get top-k indices using partition or full sort
        # Since corpus is 5,183 docs, sorting is very fast (< 1ms per query)
        scored_pairs = list(zip(self.doc_ids, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]

    def batch_retrieve(
        self, claims: List[Dict[str, any]], top_k: int = 100
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Retrieves top_k documents for a list of claims.

        Returns:
            Dict mapping claim_id -> list of (doc_id, score) sorted descending.
        """
        results = {}
        for c in claims:
            cid = int(c["id"])
            query_text = c["claim"]
            results[cid] = self.retrieve(query_text, top_k=top_k)
        return results

