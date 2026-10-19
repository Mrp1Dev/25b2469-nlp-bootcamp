"""Data loader utilities for the SciFact dataset.

Provides clean functions to load:
- corpus.jsonl: The 5,183 scientific paper abstracts
- train_claims.jsonl: 647 claims with training evidence
- val_claims.jsonl: 162 claims for validation and tuning
- eval_claims.jsonl: 300 claims for final evaluation
"""

import json
import os
from typing import Dict, List, Set, Any


def load_corpus(path: str = "data/corpus.jsonl") -> Dict[int, Dict[str, Any]]:
    """Loads the scientific corpus.

    Args:
        path: Path to corpus.jsonl

    Returns:
        Dictionary mapping doc_id (int) -> {
            'doc_id': int,
            'title': str,
            'abstract': List[str],
            'text': str  # Full concatenated text (title + abstract)
        }
    """
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = int(item["doc_id"])
            title = item.get("title", "").strip()
            abstract_sentences = item.get("abstract", [])
            abstract_text = " ".join(s.strip() for s in abstract_sentences if s.strip())
            
            # Full text combines title and abstract
            full_text = f"{title}. {abstract_text}".strip() if title else abstract_text
            
            corpus[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "abstract": abstract_sentences,
                "text": full_text
            }
    return corpus


def load_claims(path: str) -> List[Dict[str, Any]]:
    """Loads claims from a JSONL file (train, val, or eval).

    Args:
        path: Path to the claims JSONL file

    Returns:
        List of claim dictionaries:
        - train/val: {'id': int, 'claim': str, 'evidence': [{'doc_id': int, 'label': str, ...}]}
        - eval: {'id': int, 'claim': str}
    """
    claims = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item["id"] = int(item["id"])
            claims.append(item)
    return claims


def get_gold_evidence(claims: List[Dict[str, Any]]) -> Dict[int, Set[int]]:
    """Extracts ground truth document IDs for claims.

    Args:
        claims: List of claim dictionaries containing 'evidence'

    Returns:
        Dictionary mapping claim_id (int) -> set of gold doc_ids (int)
    """
    gold = {}
    for c in claims:
        cid = int(c["id"])
        evidence_list = c.get("evidence", [])
        gold[cid] = {int(ev["doc_id"]) for ev in evidence_list if "doc_id" in ev}
    return gold

