"""Evaluation metrics for Information Retrieval: nDCG@k, Recall@k, and MRR.

Key concepts implemented:
1. nDCG@10 (Normalized Discounted Cumulative Gain):
   Measures ranking quality with position discount. An evidence document at Rank 1
   is weighted higher than at Rank 5, and documents beyond Rank 10 receive 0 weight.

2. Recall@100:
   Measures the fraction of relevant evidence documents captured within the top 100 results.

3. MRR (Mean Reciprocal Rank):
   Measures how quickly the very first relevant document appears (1 / rank of first hit).

Note on SciFact evaluation:
Retrieval metrics are computed exclusively over claims that have at least one gold
evidence document in the corpus (claims with non-empty evidence).
"""

import math
from typing import Dict, List, Set, Tuple, Any


def compute_dcg_at_k(ranked_docs: List[int], gold_docs: Set[int], k: int = 10) -> float:
    """Computes Discounted Cumulative Gain at rank k with binary relevance.
    
    Formula:
        DCG@k = sum_{i=1}^k (rel_i / log2(i + 1))
        where rel_i = 1 if ranked_docs[i-1] in gold_docs, else 0.
    """
    dcg = 0.0
    for i, doc_id in enumerate(ranked_docs[:k], start=1):
        if doc_id in gold_docs:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def compute_idcg_at_k(num_gold_docs: int, k: int = 10) -> float:
    """Computes Ideal DCG at rank k (if all gold documents were at ranks 1..min(num_gold, k))."""
    idcg = 0.0
    ideal_hits = min(num_gold_docs, k)
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(i + 1)
    return idcg


def compute_ndcg_at_k(ranked_docs: List[int], gold_docs: Set[int], k: int = 10) -> float:
    """Computes Normalized DCG at rank k.
    
    Returns:
        nDCG score between 0.0 and 1.0.
    """
    if not gold_docs:
        return 0.0
    dcg = compute_dcg_at_k(ranked_docs, gold_docs, k=k)
    idcg = compute_idcg_at_k(len(gold_docs), k=k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def compute_recall_at_k(ranked_docs: List[int], gold_docs: Set[int], k: int = 100) -> float:
    """Computes Recall at rank k (fraction of gold documents retrieved in top-k)."""
    if not gold_docs:
        return 0.0
    retrieved_top_k = set(ranked_docs[:k])
    hits = len(retrieved_top_k & gold_docs)
    return hits / float(len(gold_docs))


def compute_reciprocal_rank(ranked_docs: List[int], gold_docs: Set[int], max_k: int = 100) -> float:
    """Computes Reciprocal Rank (1 / rank of first relevant document within top max_k)."""
    if not gold_docs:
        return 0.0
    for i, doc_id in enumerate(ranked_docs[:max_k], start=1):
        if doc_id in gold_docs:
            return 1.0 / float(i)
    return 0.0


def evaluate_retrieval(
    predictions: Dict[int, List[int]],
    gold_dict: Dict[int, Set[int]],
    k_ndcg: int = 10,
    k_recall: int = 100,
) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]]]:
    """Evaluates a retrieval run against gold evidence.
    
    Args:
        predictions: Dict mapping claim_id -> list of ranked doc_ids
        gold_dict: Dict mapping claim_id -> set of gold doc_ids
        k_ndcg: Cutoff for nDCG (default: 10)
        k_recall: Cutoff for Recall (default: 100)
        
    Returns:
        summary: Dict with average 'ndcg10', 'recall100', 'mrr', 'num_evaluated'
        per_claim: Dict with per-claim scores {'ndcg10': {...}, 'recall100': {...}, 'mrr': {...}}
    """
    # Evaluate only on claims that have at least one gold abstract
    eval_claim_ids = [cid for cid, gold_docs in gold_dict.items() if len(gold_docs) > 0]
    
    per_claim_ndcg = {}
    per_claim_recall = {}
    per_claim_rr = {}
    
    for cid in eval_claim_ids:
        gold_docs = gold_dict[cid]
        ranked_docs = predictions.get(cid, [])
        
        per_claim_ndcg[cid] = compute_ndcg_at_k(ranked_docs, gold_docs, k=k_ndcg)
        per_claim_recall[cid] = compute_recall_at_k(ranked_docs, gold_docs, k=k_recall)
        per_claim_rr[cid] = compute_reciprocal_rank(ranked_docs, gold_docs, max_k=k_recall)
        
    num_claims = len(eval_claim_ids)
    if num_claims == 0:
        summary = {"ndcg10": 0.0, "recall100": 0.0, "mrr": 0.0, "num_evaluated": 0}
    else:
        summary = {
            f"ndcg{k_ndcg}": sum(per_claim_ndcg.values()) / num_claims,
            f"recall{k_recall}": sum(per_claim_recall.values()) / num_claims,
            "mrr": sum(per_claim_rr.values()) / num_claims,
            "num_evaluated": num_claims,
        }
        
    per_claim = {
        f"ndcg{k_ndcg}": per_claim_ndcg,
        f"recall{k_recall}": per_claim_recall,
        "mrr": per_claim_rr,
    }
    
    return summary, per_claim

