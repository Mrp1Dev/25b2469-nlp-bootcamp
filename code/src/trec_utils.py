"""TREC run file utilities.

Handles formatting, writing, and validating retrieval run files in standard TREC format:
    claim_id Q0 doc_id rank score run_name

Rules enforced by check_format.py:
1. Whitespace-separated lines with 6 fields.
2. Each file must contain ALL claims of its split (162 for val, 300 for eval).
3. Exactly 100 unique doc_ids per claim, ranks 1 to 100 strictly consecutive.
4. Scores must be non-increasing with rank (score at rank 1 >= rank 2 >= ...).
5. All claim_id and doc_id values must be valid integers.
"""

import os
from typing import Dict, List, Tuple, Set, Optional


def write_trec_file(
    output_path: str,
    predictions: Dict[int, List[Tuple[int, float]]],
    run_name: str,
    all_claim_ids: List[int],
    corpus_doc_ids: Optional[List[int]] = None,
    top_k: int = 100,
) -> None:
    """Writes ranked predictions to a TREC file conforming strictly to submission standards.

    Args:
        output_path: Destination path for the .trec file.
        predictions: Dict mapping claim_id -> list of (doc_id, score) sorted descending by score.
        run_name: String tag for the run (e.g. 't1_bm25').
        all_claim_ids: Complete list of claim IDs that MUST be present in the file.
        corpus_doc_ids: Optional pool of all corpus doc_ids for padding if a claim retrieved < top_k.
        top_k: Number of documents per claim (must be 100).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for cid in sorted(all_claim_ids):
            doc_scores = predictions.get(cid, [])
            
            # Ensure unique doc_ids while preserving order
            seen_docs = set()
            unique_doc_scores = []
            for did, score in doc_scores:
                if did not in seen_docs:
                    seen_docs.add(did)
                    unique_doc_scores.append((did, score))
                    
            # If fewer than top_k documents, pad with corpus docs (safety fallback)
            if len(unique_doc_scores) < top_k and corpus_doc_ids:
                min_score = unique_doc_scores[-1][1] - 1.0 if unique_doc_scores else 0.0
                for did in corpus_doc_ids:
                    if did not in seen_docs:
                        seen_docs.add(did)
                        unique_doc_scores.append((did, min_score))
                        if len(unique_doc_scores) == top_k:
                            break
                            
            # Truncate to top_k
            unique_doc_scores = unique_doc_scores[:top_k]
            
            # Format and write lines
            for rank, (did, score) in enumerate(unique_doc_scores, start=1):
                f.write(f"{cid} Q0 {did} {rank} {score:.6f} {run_name}\n")


def read_trec_file(path: str) -> Dict[int, List[int]]:
    """Reads a TREC file and returns a mapping from claim_id to a list of ranked doc_ids (1..100).

    Args:
        path: Path to the .trec file.

    Returns:
        Dict mapping claim_id -> list of doc_ids in rank order.
    """
    runs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            qid, did, rank = int(parts[0]), int(parts[2]), int(parts[3])
            runs.setdefault(qid, {})[rank] = did
            
    out = {}
    for qid, rank_map in runs.items():
        out[qid] = [rank_map[r] for r in sorted(rank_map.keys())]
    return out

