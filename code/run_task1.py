"""Task 1: BM25 Exploration, Tuning, and Run Generation.

1. Explores corpus and claims text statistics.
2. Performs a grid search over tokenization methods and (k1, b) parameters on the validation set.
3. Selects the optimal parameters based on validation nDCG@10.
4. Exports 'predictions/val_bm25.trec' and 'predictions/t1_bm25.trec'.
5. Verifies files with check_format.py standards.
"""

import os
import sys
import time
import json
import statistics

# Ensure repo root and code/ directory are in sys.path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR) if os.path.basename(CODE_DIR) == "code" else CODE_DIR
for p in [CODE_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_loader import load_corpus, load_claims, get_gold_evidence
from src.metrics import evaluate_retrieval
from src.bm25_retriever import (
    BM25Retriever,
    tokenize_simple,
    tokenize_clean,
    tokenize_with_compounds,
)
from src.trec_utils import write_trec_file
from check_format import check_trec

DATA_DIR = os.path.join(ROOT_DIR, "data")
PRED_DIR = os.path.join(ROOT_DIR, "predictions")


def explore_corpus_and_claims(corpus, train_claims, val_claims, eval_claims):
    print("=" * 60)
    print("STEP 1: CORPUS & CLAIM EXPLORATION")
    print("=" * 60)
    
    doc_word_counts = [len(d["text"].split()) for d in corpus.values()]
    title_word_counts = [len(d["title"].split()) for d in corpus.values() if d["title"]]
    val_claim_lengths = [len(c["claim"].split()) for c in val_claims]
    eval_claim_lengths = [len(c["claim"].split()) for c in eval_claims]
    
    print(f"Total documents in corpus: {len(corpus):,}")
    print(f"Document total length (words): mean={statistics.mean(doc_word_counts):.1f}, "
          f"median={statistics.median(doc_word_counts):.1f}, min={min(doc_word_counts)}, max={max(doc_word_counts)}")
    print(f"Title length (words): mean={statistics.mean(title_word_counts):.1f}, "
          f"median={statistics.median(title_word_counts):.1f}")
    print(f"Validation claim length: mean={statistics.mean(val_claim_lengths):.1f} words, "
          f"min={min(val_claim_lengths)}, max={max(val_claim_lengths)}")
    print(f"Evaluation claim length: mean={statistics.mean(eval_claim_lengths):.1f} words, "
          f"min={min(eval_claim_lengths)}, max={max(eval_claim_lengths)}")
    print()


def tune_bm25(corpus, val_claims, val_gold):
    print("=" * 60)
    print("STEP 2: BM25 HYPERPARAMETER TUNING (VALIDATION SET)")
    print("=" * 60)
    
    tokenizers = {
        "simple_with_stopwords": lambda t: tokenize_clean(t, remove_stopwords=False, preserve_compounds=False),
        "clean_no_stopwords": lambda t: tokenize_clean(t, remove_stopwords=True, preserve_compounds=False),
        "compounds_no_stopwords": lambda t: tokenize_clean(t, remove_stopwords=True, preserve_compounds=True),
    }
    
    k1_candidates = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8]
    b_candidates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.85]
    
    best_config = None
    best_ndcg = -1.0
    best_summary = None
    
    # 1. First test tokenization strategies with default k1=1.2, b=0.75
    print("\n--- Comparing Tokenization Strategies (k1=1.2, b=0.75, Okapi) ---")
    tok_results = {}
    for tok_name, tok_fn in tokenizers.items():
        retriever = BM25Retriever(corpus, k1=1.2, b=0.75, method="okapi", tokenize_fn=tok_fn)
        preds_raw = retriever.batch_retrieve(val_claims, top_k=100)
        preds_ids = {cid: [did for did, score in doc_list] for cid, doc_list in preds_raw.items()}
        summary, _ = evaluate_retrieval(preds_ids, val_gold)
        tok_results[tok_name] = summary
        print(f"  {tok_name:25s} -> nDCG@10: {summary['ndcg10']:.4f} | Recall@100: {summary['recall100']:.4f} | MRR: {summary['mrr']:.4f}")
        
    best_tok_name = max(tok_results, key=lambda k: tok_results[k]["ndcg10"])
    chosen_tok_fn = tokenizers[best_tok_name]
    print(f">> Best tokenization strategy: {best_tok_name} (nDCG@10 = {tok_results[best_tok_name]['ndcg10']:.4f})")
    
    # 2. Grid search over k1 and b using the best tokenization
    print(f"\n--- Grid Searching k1 in {k1_candidates} and b in {b_candidates} ---")
    grid_results = []
    
    # Pre-tokenize once for speed
    corpus_doc_ids = list(corpus.keys())
    tokenized_corpus = [chosen_tok_fn(corpus[did]["text"]) for did in corpus_doc_ids]
    val_claim_tokens = {int(c["id"]): chosen_tok_fn(c["claim"]) for c in val_claims}
    
    from rank_bm25 import BM25Okapi, BM25Plus
    for method_name in ["okapi", "plus"]:
        for k1 in k1_candidates:
            for b in b_candidates:
                if method_name == "plus":
                    bm25_model = BM25Plus(tokenized_corpus, k1=k1, b=b)
                else:
                    bm25_model = BM25Okapi(tokenized_corpus, k1=k1, b=b)
                    
                preds_ids = {}
                for cid, q_tokens in val_claim_tokens.items():
                    if not q_tokens:
                        preds_ids[cid] = corpus_doc_ids[:100]
                    else:
                        scores = bm25_model.get_scores(q_tokens)
                        scored_pairs = list(zip(corpus_doc_ids, scores))
                        scored_pairs.sort(key=lambda x: x[1], reverse=True)
                        preds_ids[cid] = [did for did, score in scored_pairs[:100]]
                        
                summary, _ = evaluate_retrieval(preds_ids, val_gold)
                ndcg = summary["ndcg10"]
                grid_results.append({
                    "method": method_name,
                    "k1": k1,
                    "b": b,
                    "summary": summary
                })
                
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_config = {
                        "tokenizer": best_tok_name,
                        "method": method_name,
                        "k1": k1,
                        "b": b
                    }
                    best_summary = summary
                    print(f"  [NEW BEST] method={method_name}, k1={k1:.1f}, b={b:.2f} -> "
                          f"nDCG@10: {ndcg:.4f}, Recall@100: {summary['recall100']:.4f}, MRR: {summary['mrr']:.4f}")
                          
    print("\n" + "=" * 60)
    print("WINNING BM25 CONFIGURATION:")
    print("=" * 60)
    print(f"  Method:      BM25{best_config['method'].capitalize()}")
    print(f"  Tokenizer:   {best_config['tokenizer']}")
    print(f"  k1:          {best_config['k1']}")
    print(f"  b:           {best_config['b']}")
    print(f"  Val nDCG@10: {best_summary['ndcg10']:.4f}")
    print(f"  Val Rec@100: {best_summary['recall100']:.4f}")
    print(f"  Val MRR:     {best_summary['mrr']:.4f}")
    print("=" * 60)
    
    return best_config, best_summary, chosen_tok_fn


def generate_submission_runs(corpus, val_claims, eval_claims, best_config, tokenize_fn):
    print("\n" + "=" * 60)
    print("STEP 3: GENERATING TREC RUN FILES")
    print("=" * 60)
    
    corpus_doc_ids = list(corpus.keys())
    val_claim_ids = [int(c["id"]) for c in val_claims]
    eval_claim_ids = [int(c["id"]) for c in eval_claims]
    
    # Initialize best retriever
    retriever = BM25Retriever(
        corpus=corpus,
        k1=best_config["k1"],
        b=best_config["b"],
        method=best_config["method"],
        tokenize_fn=tokenize_fn,
    )
    
    # 1. Validation run: predictions/val_bm25.trec (needed for Q1)
    val_preds = retriever.batch_retrieve(val_claims, top_k=100)
    val_trec_path = os.path.join(PRED_DIR, "val_bm25.trec")
    write_trec_file(
        output_path=val_trec_path,
        predictions=val_preds,
        run_name="val_bm25",
        all_claim_ids=val_claim_ids,
        corpus_doc_ids=corpus_doc_ids,
        top_k=100,
    )
    print(f"Successfully generated: {val_trec_path}")
    
    # 2. Evaluation run: predictions/t1_bm25.trec (Task 1 submission)
    eval_preds = retriever.batch_retrieve(eval_claims, top_k=100)
    eval_trec_path = os.path.join(PRED_DIR, "t1_bm25.trec")
    write_trec_file(
        output_path=eval_trec_path,
        predictions=eval_preds,
        run_name="t1_bm25",
        all_claim_ids=eval_claim_ids,
        corpus_doc_ids=corpus_doc_ids,
        top_k=100,
    )
    print(f"Successfully generated: {eval_trec_path}")
    
    # 3. Check with check_format.py validators
    print("\nValidating files against check_format.py standards...")
    check_trec(val_trec_path, set(val_claim_ids), set(corpus_doc_ids))
    print(f"  [OK] {val_trec_path} is 100% valid.")
    check_trec(eval_trec_path, set(eval_claim_ids), set(corpus_doc_ids))
    print(f"  [OK] {eval_trec_path} is 100% valid.")


def main():
    corpus = load_corpus(os.path.join(DATA_DIR, "corpus.jsonl"))
    train_claims = load_claims(os.path.join(DATA_DIR, "train_claims.jsonl"))
    val_claims = load_claims(os.path.join(DATA_DIR, "val_claims.jsonl"))
    eval_claims = load_claims(os.path.join(DATA_DIR, "eval_claims.jsonl"))
    val_gold = get_gold_evidence(val_claims)
    
    explore_corpus_and_claims(corpus, train_claims, val_claims, eval_claims)
    best_config, best_summary, tokenize_fn = tune_bm25(corpus, val_claims, val_gold)
    generate_submission_runs(corpus, val_claims, eval_claims, best_config, tokenize_fn)


if __name__ == "__main__":
    main()

