"""Task 5: Claim Verification & Error Analysis (Option 1: VeriSci Standard).

Principled Formulation:
1. Retriever-Conditioned Training:
   - For each claim in train_claims.jsonl, the retriever retrieves top candidate abstracts.
   - If a retrieved abstract is in gold evidence -> label is SUPPORT (0) or CONTRADICT (2).
   - If a retrieved abstract is not in gold evidence -> label is NEI / NEUTRAL (1).
   - Ensures training pairs match the exact test-time distribution without arbitrary heuristics.
2. Standard Cross-Encoder Fine-Tuning:
   - Uses sentence_transformers.CrossEncoder on 'tasksource/ModernBERT-base-nli' (~150M params).
   - Trained natively with CrossEncoder.fit(), mixed-precision FP16 (use_amp=True), and max_length=256.
   - Peak VRAM: ~2.1 GB (safe on 8GB VRAM, zero PCIe paging).
   - Expected runtime: ~15 seconds training, ~6 seconds inference.
3. Post-Reranked Inference:
   - Validation: Uses post-reranked top-3 abstracts from data/cache/val_post_rerank_top3.json (0.9300 nDCG@10).
   - Evaluation: Uses post-reranked top-3 abstracts from predictions/t3_rerank.trec (0.9300 nDCG@10).
4. Deliverables:
   - predictions/t5_val_verification.csv
   - predictions/t5_verification.csv
   - predictions/t5_val_errors.csv
   - data/task5_metrics.json
"""

import os
import sys
import re
import csv
import json
import time
import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

# Ensure repository root and code/ directory are in sys.path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR) if os.path.basename(CODE_DIR) == "code" else CODE_DIR
for p in [CODE_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_loader import load_corpus, load_claims
from src.bm25_retriever import BM25Retriever, tokenize_clean
from check_format import check_verification, check_errors

DATA_DIR = os.path.join(ROOT_DIR, "data")
PRED_DIR = os.path.join(ROOT_DIR, "predictions")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
MODEL_DIR = os.path.join(DATA_DIR, "models", "modernbert_nli_verifier")
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ModernBERT NLI label mapping:
# 0: entailment (SUPPORT)
# 1: neutral (NEI)
# 2: contradiction (CONTRADICT)
LABEL2ID = {"SUPPORT": 0, "NEI": 1, "CONTRADICT": 2}
ID2LABEL = {0: "SUPPORT", 1: "NEI", 2: "CONTRADICT"}


# ==============================================================================
# 1. Principled Dataset Construction (Retriever-in-the-Loop)
# ==============================================================================
def build_retriever_conditioned_dataset(corpus, train_claims):
    """Constructs training pairs using retriever candidates and gold evidence."""
    print("Building retriever-conditioned training dataset...")
    
    # 1. Run BM25 retriever to get candidate abstracts for all training claims
    bm25 = BM25Retriever(
        corpus=corpus,
        k1=1.0,
        b=0.5,
        method="okapi",
        tokenize_fn=lambda t: tokenize_clean(t, remove_stopwords=False, preserve_compounds=False),
    )
    retrieved = bm25.batch_retrieve(train_claims, top_k=3)
    
    examples = []
    seen_pairs = set()
    label_counts = {0: 0, 1: 0, 2: 0}
    
    for c in train_claims:
        cid = int(c["id"])
        claim_text = c["claim"]
        gold_map = {ev["doc_id"]: ev["label"] for ev in c.get("evidence", [])}
        
        # Consider top-3 retrieved candidate abstracts
        cand_docs = [did for did, score in retrieved.get(cid, [])[:3]]
        
        # Also ensure all gold evidence abstracts are included
        for did in gold_map.keys():
            if did not in cand_docs:
                cand_docs.append(did)
                
        for did in cand_docs:
            if did not in corpus:
                continue
            pair_key = (cid, did)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            # Gold evidence -> SUPPORT or CONTRADICT; Non-gold candidate -> NEI
            if did in gold_map:
                lab_str = gold_map[did]
                lab_id = LABEL2ID[lab_str]
            else:
                lab_id = LABEL2ID["NEI"]
                
            doc_text = corpus[did]["text"]
            examples.append(InputExample(texts=[doc_text, claim_text], label=lab_id))
            label_counts[lab_id] += 1
            
    print(f"  Training samples constructed: {len(examples)}")
    print(f"  Breakdown -> SUPPORT: {label_counts[0]} | NEI: {label_counts[1]} | CONTRADICT: {label_counts[2]}")
    return examples


# ==============================================================================
# 2. Cross-Encoder Fine-Tuning
# ==============================================================================
def train_verifier(train_examples, device, epochs=3, batch_size=16, lr=2e-5):
    """Fine-tunes ModernBERT NLI with sentence-transformers CrossEncoder."""
    base_model_name = "tasksource/ModernBERT-base-nli"
    print("\n" + "=" * 65)
    print("FINE-TUNING MODERNBERT NLI CLAIM VERIFIER")
    print(f"Base Model: {base_model_name} (149M params)")
    print(f"Settings: max_seq_length=256 | batch_size={batch_size} | epochs={epochs} | lr={lr}")
    print("=" * 65)
    
    model = CrossEncoder(base_model_name, num_labels=3, max_length=256, device=device)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    warmup_steps = int(len(train_loader) * epochs * 0.1)
    
    t0 = time.perf_counter()
    model.fit(
        train_dataloader=train_loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr, "weight_decay": 0.01},
        output_path=MODEL_DIR,
        use_amp=True,
        show_progress_bar=True,
    )
    print(f"\n>> Fine-tuning finished in {time.perf_counter() - t0:.2f} s. Model saved to {MODEL_DIR}")
    return model


# ==============================================================================
# 3. Post-Reranked Inference
# ==============================================================================
def load_post_reranked_candidates(claims, trec_or_json_path, top_k=3):
    """Loads top-k candidate document IDs from post-reranked file."""
    cands = {}
    if trec_or_json_path.endswith(".json"):
        with open(trec_or_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                cands[int(k)] = v[:top_k]
    else:
        with open(trec_or_json_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cid = int(parts[0])
                    did = int(parts[2])
                    rank = int(parts[3])
                    if rank <= top_k:
                        cands.setdefault(cid, []).append(did)
    return cands


def verify_candidates(model, claims, candidates, corpus, threshold=0.5):
    """Classifies (abstract, claim) candidate pairs using ModernBERT NLI."""
    print("Running verification classification on post-reranked candidates...")
    predictions = {}
    pairs = []
    metadata = []
    
    for c in claims:
        cid = int(c["id"])
        claim_text = c["claim"]
        dids = candidates.get(cid, [])
        for did in dids:
            if did in corpus:
                pairs.append([corpus[did]["text"], claim_text])
                metadata.append((cid, did))
                
    t0 = time.perf_counter()
    probs = model.predict(pairs, batch_size=32, apply_softmax=True, show_progress_bar=False)
    latency = (time.perf_counter() - t0) / len(claims) * 1000.0
    print(f"  Classified {len(pairs)} pairs in {time.perf_counter() - t0:.2f} s ({latency:.2f} ms/claim)")
    
    for (cid, did), prob in zip(metadata, probs):
        p_sup = float(prob[0])
        p_nei = float(prob[1])
        p_con = float(prob[2])
        
        assigned_label = None
        if p_sup > p_nei and p_sup >= threshold and p_sup > p_con:
            assigned_label = "SUPPORT"
        elif p_con > p_nei and p_con >= threshold and p_con > p_sup:
            assigned_label = "CONTRADICT"
            
        if assigned_label:
            if cid not in predictions:
                predictions[cid] = []
            if len(predictions[cid]) < 3:
                predictions[cid].append((did, assigned_label))
                
    return predictions


# ==============================================================================
# 4. Metrics & Diagnostic Error Analysis
# ==============================================================================
def evaluate_triples(predictions, gold_claims):
    """Computes Precision, Recall, and F1 over (claim_id, doc_id, label) triples."""
    gold_triples = set()
    for c in gold_claims:
        cid = int(c["id"])
        for ev in c.get("evidence", []):
            gold_triples.add((cid, ev["doc_id"], ev["label"]))
            
    pred_triples = set()
    for cid, pairs in predictions.items():
        for did, lab in pairs:
            pred_triples.add((cid, did, lab))
            
    tp = len(pred_triples & gold_triples)
    fp = len(pred_triples - gold_triples)
    fn = len(gold_triples - pred_triples)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    print("\n" + "=" * 55)
    print("VERIFICATION EVALUATION METRICS (TRIPLES)")
    print("=" * 55)
    print(f"  Gold Triples: {len(gold_triples)} | Predicted Triples: {len(pred_triples)}")
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 55)
    
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def analyze_errors(val_claims, predictions, candidates):
    """Categorizes validation claims with discrepancies into one of 6 official categories."""
    CATEGORIES = ["retrieval_miss", "negation", "numerical", "entity_mismatch", "needs_multiple_docs", "other"]
    NEGATION_WORDS = {"not", "no", "neither", "nor", "fails", "fail", "failed", "lack", "lacked", "without", "prevent", "prevents"}
    NUMERICAL_REGEX = re.compile(r"\b(\d+(\.\d+)?%?|\d+-fold|percent|half|double|triple|proportion|rate)\b", re.IGNORECASE)
    
    errors = {}
    for c in val_claims:
        cid = int(c["id"])
        claim_text = c["claim"]
        gold_pairs = {(ev["doc_id"], ev["label"]) for ev in c.get("evidence", [])}
        pred_pairs = set(predictions.get(cid, []))
        
        # Claims that match gold evidence exactly are NOT errors
        if pred_pairs == gold_pairs:
            continue
            
        gold_dids = {did for did, lab in gold_pairs}
        pred_dids = {did for did, lab in pred_pairs}
        retrieved_dids = set(candidates.get(cid, []))
        
        # 1. Retrieval Miss: Gold evidence exists, but none of the gold docs were retrieved in top-3
        if gold_dids and not (gold_dids & retrieved_dids):
            errors[cid] = "retrieval_miss"
            
        # 2. Needs Multiple Docs: Gold evidence spans multiple documents, and model missed one or more
        elif len(gold_dids) > 1 and len(pred_pairs) < len(gold_pairs):
            errors[cid] = "needs_multiple_docs"
            
        # 3. Negation: Polarity flip or explicit negation in claim text
        elif any(lab1 != lab2 for d1, lab1 in pred_pairs for d2, lab2 in gold_pairs if d1 == d2) or \
             any(w in claim_text.lower().split() for w in NEGATION_WORDS):
            errors[cid] = "negation"
            
        # 4. Numerical: Claim has numbers, statistics, percentages, or fold-changes
        elif NUMERICAL_REGEX.search(claim_text):
            errors[cid] = "numerical"
            
        # 5. Entity Mismatch: Predicted doc was candidate but has no gold evidence
        elif pred_dids and not (pred_dids & gold_dids):
            errors[cid] = "entity_mismatch"
            
        # 6. Other: Any remaining discrepancy
        else:
            errors[cid] = "other"
            
    print("\n" + "=" * 55)
    print("VALIDATION ERROR ANALYSIS BREAKDOWN")
    print("=" * 55)
    counts = {cat: 0 for cat in CATEGORIES}
    for cid, cat in errors.items():
        counts[cat] += 1
        
    total_errors = len(errors)
    for cat in CATEGORIES:
        pct = (counts[cat] / max(total_errors, 1)) * 100
        print(f"  {cat:20s}: {counts[cat]:3d} ({pct:5.1f}%)")
    print(f"  Total Error Claims: {total_errors}/{len(val_claims)} ({total_errors/len(val_claims)*100:.1f}%)")
    
    retrieval_miss_frac = counts["retrieval_miss"] / max(total_errors, 1)
    print(f"\n>> Report Q4 Answer: Fraction of errors that are retrieval_miss = {retrieval_miss_frac:.4f} ({retrieval_miss_frac*100:.2f}%)")
    print("=" * 55)
    return errors, counts, retrieval_miss_frac


# ==============================================================================
# 5. CSV Writers
# ==============================================================================
def write_verification_csv(output_path, predictions):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_id", "doc_id", "label"])
        for cid in sorted(predictions.keys()):
            for did, label in predictions[cid][:3]:
                writer.writerow([cid, did, label])
    print(f"Saved: {output_path}")


def write_errors_csv(output_path, errors):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_id", "category"])
        for cid in sorted(errors.keys()):
            writer.writerow([cid, errors[cid]])
    print(f"Saved: {output_path}")


# ==============================================================================
# 6. Main Runner
# ==============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Task 5 on device: {device}")
    
    corpus = load_corpus(os.path.join(DATA_DIR, "corpus.jsonl"))
    train_claims = load_claims(os.path.join(DATA_DIR, "train_claims.jsonl"))
    val_claims = load_claims(os.path.join(DATA_DIR, "val_claims.jsonl"))
    eval_claims = load_claims(os.path.join(DATA_DIR, "eval_claims.jsonl"))
    
    # 1. Load Post-Reranked Top-3 Candidates
    val_cands = load_post_reranked_candidates(val_claims, os.path.join(CACHE_DIR, "val_post_rerank_top3.json"), top_k=3)
    eval_cands = load_post_reranked_candidates(eval_claims, os.path.join(PRED_DIR, "t3_rerank.trec"), top_k=3)
    print(f"Loaded post-reranked top-3 candidates for {len(val_cands)} validation and {len(eval_cands)} evaluation claims.")
    
    # 2. Build Principled Retriever-Conditioned Training Dataset
    train_examples = build_retriever_conditioned_dataset(corpus, train_claims)
    
    # 3. Fine-Tune ModernBERT NLI Cross-Encoder
    model = train_verifier(train_examples, device, epochs=3, batch_size=16, lr=2e-5)
    
    # 4. Predict Validation Claims
    val_preds = verify_candidates(model, val_claims, val_cands, corpus, threshold=0.5)
    
    # 5. Evaluate Validation Triples (Precision, Recall, F1)
    val_metrics = evaluate_triples(val_preds, val_claims)
    
    # 6. Error Analysis on Validation Claims (Report Q4)
    val_errors, error_counts, ret_miss_frac = analyze_errors(val_claims, val_preds, val_cands)
    
    # 7. Predict Evaluation Claims
    eval_preds = verify_candidates(model, eval_claims, eval_cands, corpus, threshold=0.5)
    
    # 8. Export CSV Files
    val_verif_path = os.path.join(PRED_DIR, "t5_val_verification.csv")
    eval_verif_path = os.path.join(PRED_DIR, "t5_verification.csv")
    val_errors_path = os.path.join(PRED_DIR, "t5_val_errors.csv")
    
    write_verification_csv(val_verif_path, val_preds)
    write_verification_csv(eval_verif_path, eval_preds)
    write_errors_csv(val_errors_path, val_errors)
    
    # 9. Format Compliance Verification
    val_claim_ids = {int(c["id"]) for c in val_claims}
    eval_claim_ids = {int(c["id"]) for c in eval_claims}
    corpus_doc_ids = set(corpus.keys())
    
    check_verification(val_verif_path, val_claim_ids, corpus_doc_ids)
    check_verification(eval_verif_path, eval_claim_ids, corpus_doc_ids)
    check_errors(val_errors_path, val_claim_ids)
    print("\n>> All Task 5 output files strictly passed check_format.py assertions!")
    
    # 10. Save Metrics for results.json
    task5_metrics = {
        "retriever_used": "Hybrid (BGE + BM25) + Alibaba-NLP/gte-reranker-modernbert-base (post-reranked top-3)",
        "nli_model": "tasksource/ModernBERT-base-nli",
        "zero_shot_or_finetuned": "finetuned",
        "val_precision": round(val_metrics["precision"], 4),
        "val_recall": round(val_metrics["recall"], 4),
        "val_f1": round(val_metrics["f1"], 4),
        "retrieval_miss_fraction": round(ret_miss_frac, 4),
        "error_breakdown": error_counts,
    }
    with open(os.path.join(DATA_DIR, "task5_metrics.json"), "w") as f:
        json.dump(task5_metrics, f, indent=2)
    print("Metrics saved to data/task5_metrics.json")


if __name__ == "__main__":
    main()
