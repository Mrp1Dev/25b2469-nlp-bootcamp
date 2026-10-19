"""Task 3: Combine & Rerank.

1. Hybrid Fusion:
   - Sweeps alpha in [0.0, 0.1, ..., 1.0] for min-max score interpolation:
     score_hybrid = alpha * score_dense_norm + (1 - alpha) * score_bm25_norm
   - Finds optimal alpha on validation set.
   - Generates predictions/t3_hybrid.trec.

2. Cross-Encoder Reranking:
   - Evaluates Alibaba-NLP/gte-reranker-modernbert-base (149M params, 8k context).
   - Sweeps top-k in [10, 15, 20].
   - Reranks top-k candidates, preserving ranks k+1..100 in original order.
   - Measures query latency (ms) for results.json.
   - Generates predictions/t3_rerank.trec.
"""

import os
import sys
import time
import json
import torch

# Ensure repository root and code/ directory are in sys.path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR) if os.path.basename(CODE_DIR) == "code" else CODE_DIR
for p in [CODE_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sentence_transformers import CrossEncoder
from src.data_loader import load_corpus, load_claims, get_gold_evidence
from src.metrics import evaluate_retrieval
from src.bm25_retriever import BM25Retriever, tokenize_clean
from src.trec_utils import write_trec_file, read_trec_file
from check_format import check_trec

DATA_DIR = os.path.join(ROOT_DIR, "data")
PRED_DIR = os.path.join(ROOT_DIR, "predictions")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def min_max_normalize(scores_dict):
    """Normalizes scores to [0, 1] per query."""
    norm_dict = {}
    for qid, doc_scores in scores_dict.items():
        if not doc_scores:
            norm_dict[qid] = {}
            continue
        scores = [s for _, s in doc_scores]
        min_s = min(scores)
        max_s = max(scores)
        diff = max_s - min_s
        if diff == 0:
            norm_dict[qid] = {did: 1.0 for did, _ in doc_scores}
        else:
            norm_dict[qid] = {did: (s - min_s) / diff for did, s in doc_scores}
    return norm_dict


def build_hybrid_predictions(dense_preds, bm25_preds, alpha, corpus_doc_ids, top_k=100):
    """Combines dense and bm25 predictions using min-max interpolation."""
    norm_dense = min_max_normalize(dense_preds)
    norm_bm25 = min_max_normalize(bm25_preds)

    hybrid_preds = {}
    all_qids = sorted(dense_preds.keys())

    for qid in all_qids:
        dense_map = norm_dense.get(qid, {})
        bm25_map = norm_bm25.get(qid, {})

        # All candidate docs from both top-100 lists
        candidate_docs = set(dense_map.keys()) | set(bm25_map.keys())
        scored = []
        for did in candidate_docs:
            d_score = dense_map.get(did, 0.0)
            b_score = bm25_map.get(did, 0.0)
            h_score = alpha * d_score + (1.0 - alpha) * b_score
            scored.append((did, h_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        hybrid_preds[qid] = scored[:top_k]

    return hybrid_preds


def sweep_hybrid(corpus, val_claims, eval_claims, val_gold):
    print("=" * 60)
    print("STEP 1: HYBRID SEARCH (MIN-MAX NORMALIZED INTERPOLATION)")
    print("=" * 60)

    corpus_doc_ids = list(corpus.keys())
    val_claim_ids = [int(c["id"]) for c in val_claims]
    eval_claim_ids = [int(c["id"]) for c in eval_claims]

    # 1. Load BM25 predictions
    print("Computing BM25 scores...")
    bm25 = BM25Retriever(
        corpus=corpus,
        k1=1.0,
        b=0.5,
        method="okapi",
        tokenize_fn=lambda t: tokenize_clean(t, remove_stopwords=False, preserve_compounds=False),
    )
    val_bm25_raw = bm25.batch_retrieve(val_claims, top_k=100)
    eval_bm25_raw = bm25.batch_retrieve(eval_claims, top_k=100)

    # 2. Load BGE Dense predictions
    print("Computing BGE Dense scores...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bge_emb_path = os.path.join(CACHE_DIR, "bge_base_corpus_emb.pt")
    doc_embeddings = torch.load(bge_emb_path, map_location=device)

    from sentence_transformers import SentenceTransformer
    bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    bge_model.eval()

    prompt = "Represent this sentence for searching relevant passages: "
    val_texts = [f"{prompt}{c['claim']}" for c in val_claims]
    eval_texts = [f"{prompt}{c['claim']}" for c in eval_claims]

    with torch.inference_mode():
        q_val = bge_model.encode(val_texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, device=device)
        sim_val = torch.matmul(q_val, doc_embeddings.T)
        topk_s_val, topk_i_val = torch.topk(sim_val, k=100, dim=1)

        q_eval = bge_model.encode(eval_texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, device=device)
        sim_eval = torch.matmul(q_eval, doc_embeddings.T)
        topk_s_eval, topk_i_eval = torch.topk(sim_eval, k=100, dim=1)

    val_dense_raw = {}
    for idx, cid in enumerate(val_claim_ids):
        d_ids = [corpus_doc_ids[i] for i in topk_i_val[idx].cpu().tolist()]
        d_scores = topk_s_val[idx].cpu().tolist()
        val_dense_raw[cid] = list(zip(d_ids, d_scores))

    eval_dense_raw = {}
    for idx, cid in enumerate(eval_claim_ids):
        d_ids = [corpus_doc_ids[i] for i in topk_i_eval[idx].cpu().tolist()]
        d_scores = topk_s_eval[idx].cpu().tolist()
        eval_dense_raw[cid] = list(zip(d_ids, d_scores))

    # Free dense model and embeddings to avoid VRAM bloat before cross-encoder
    del bge_model, doc_embeddings, q_val, q_eval, sim_val, sim_eval, topk_s_val, topk_i_val, topk_s_eval, topk_i_eval
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Sweep alpha: score = alpha * dense + (1 - alpha) * bm25
    print("\n--- Sweeping Alpha: alpha * Dense + (1 - alpha) * BM25 ---")
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_alpha = None
    best_val_ndcg = -1.0
    best_summary = None

    for alpha in alphas:
        hybrid_val = build_hybrid_predictions(val_dense_raw, val_bm25_raw, alpha, corpus_doc_ids, top_k=100)
        pred_ids = {cid: [did for did, s in dlist] for cid, dlist in hybrid_val.items()}
        summary, _ = evaluate_retrieval(pred_ids, val_gold)
        ndcg = summary["ndcg10"]
        print(f"  alpha = {alpha:.1f} (Dense: {alpha*100:3.0f}%, BM25: {(1-alpha)*100:3.0f}%) -> "
              f"nDCG@10: {ndcg:.4f} | Recall@100: {summary['recall100']:.4f} | MRR: {summary['mrr']:.4f}")

        if ndcg > best_val_ndcg:
            best_val_ndcg = ndcg
            best_alpha = alpha
            best_summary = summary

    print(f"\n>> WINNING HYBRID: alpha = {best_alpha:.2f} with Val nDCG@10 = {best_val_ndcg:.4f}")

    # Generate predictions/t3_hybrid.trec using best alpha
    val_hybrid_best = build_hybrid_predictions(val_dense_raw, val_bm25_raw, best_alpha, corpus_doc_ids, top_k=100)
    eval_hybrid_best = build_hybrid_predictions(eval_dense_raw, eval_bm25_raw, best_alpha, corpus_doc_ids, top_k=100)

    eval_hybrid_path = os.path.join(PRED_DIR, "t3_hybrid.trec")
    write_trec_file(eval_hybrid_path, eval_hybrid_best, "t3_hybrid", eval_claim_ids, corpus_doc_ids, top_k=100)
    check_trec(eval_hybrid_path, set(eval_claim_ids), set(corpus_doc_ids))
    print(f"Verified & saved: {eval_hybrid_path}")

    return best_alpha, best_val_ndcg, val_hybrid_best, eval_hybrid_best


def run_cross_encoder_rerank(
    corpus, val_claims, eval_claims, val_gold, val_first_stage, eval_first_stage, best_val_ndcg=0.0, best_alpha=0.5
):
    print("\n" + "=" * 60)
    print("STEP 2: CROSS-ENCODER RERANKING")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker_name = "Alibaba-NLP/gte-reranker-modernbert-base"
    print(f"Loading cross-encoder '{reranker_name}' onto {device}...")
    reranker = CrossEncoder(reranker_name, device=device, max_length=512)

    corpus_doc_ids = list(corpus.keys())
    val_claim_ids = [int(c["id"]) for c in val_claims]
    eval_claim_ids = [int(c["id"]) for c in eval_claims]

    # Map claim_id to claim text
    val_claim_text = {int(c["id"]): c["claim"] for c in val_claims}
    eval_claim_text = {int(c["id"]): c["claim"] for c in eval_claims}

    k_candidates = [10, 15, 20]
    best_k = None
    best_rerank_ndcg = -1.0
    best_summary = None

    print(f"\n--- Testing top-k values in {k_candidates} on validation set ---")
    for k in k_candidates:
        # Build pairs for top-k docs of each query
        all_pairs = []
        pair_counts = []
        for cid in val_claim_ids:
            q_text = val_claim_text[cid]
            top_k_docs = val_first_stage[cid][:k]
            for did, _ in top_k_docs:
                all_pairs.append([q_text, corpus[did]["text"]])
            pair_counts.append(len(top_k_docs))

        # Predict scores with CrossEncoder
        t0 = time.perf_counter()
        scores = reranker.predict(all_pairs, batch_size=32, show_progress_bar=False)
        total_time = time.perf_counter() - t0
        avg_latency = (total_time / len(val_claim_ids)) * 1000.0

        # Construct reranked output
        reranked_val = {}
        offset = 0
        for cid in val_claim_ids:
            first_stage_docs = val_first_stage[cid]
            count = len(first_stage_docs[:k])
            cand_scores = scores[offset : offset + count]
            offset += count

            # Rerank the top-k items
            top_k_pairs = [(first_stage_docs[i][0], float(cand_scores[i])) for i in range(count)]
            top_k_pairs.sort(key=lambda x: x[1], reverse=True)

            # Ranks k+1..100 keep first stage ranking without duplicates
            seen = {did for did, _ in top_k_pairs}
            remaining = []
            for did, s in first_stage_docs[k:]:
                if did not in seen:
                    seen.add(did)
                    # Assign non-increasing fallback scores
                    fallback_score = top_k_pairs[-1][1] - 1.0 - len(remaining)
                    remaining.append((did, fallback_score))

            full_list = (top_k_pairs + remaining)[:100]
            reranked_val[cid] = full_list

        pred_ids = {cid: [did for did, s in dlist] for cid, dlist in reranked_val.items()}
        summary, _ = evaluate_retrieval(pred_ids, val_gold)
        ndcg = summary["ndcg10"]
        print(f"  k = {k:2d} -> nDCG@10: {ndcg:.4f} | Recall@100: {summary['recall100']:.4f} | "
              f"MRR: {summary['mrr']:.4f} | Latency: {avg_latency:.2f} ms/query")

        if ndcg > best_rerank_ndcg:
            best_rerank_ndcg = ndcg
            best_k = k
            best_summary = summary
            best_latency = avg_latency

    print("\n" + "=" * 60)
    print(f"WINNING RERANKER: k = {best_k} with Val nDCG@10 = {best_rerank_ndcg:.4f}")
    print("=" * 60)

    # Generate predictions/t3_rerank.trec using best_k
    print(f"\nReranking evaluation claims with best k = {best_k}...")
    eval_pairs = []
    for cid in eval_claim_ids:
        q_text = eval_claim_text[cid]
        top_k_docs = eval_first_stage[cid][:best_k]
        for did, _ in top_k_docs:
            eval_pairs.append([q_text, corpus[did]["text"]])

    eval_scores = reranker.predict(eval_pairs, batch_size=32, show_progress_bar=False)

    reranked_eval = {}
    offset = 0
    for cid in eval_claim_ids:
        first_stage_docs = eval_first_stage[cid]
        count = len(first_stage_docs[:best_k])
        cand_scores = eval_scores[offset : offset + count]
        offset += count

        top_k_pairs = [(first_stage_docs[i][0], float(cand_scores[i])) for i in range(count)]
        top_k_pairs.sort(key=lambda x: x[1], reverse=True)

        seen = {did for did, _ in top_k_pairs}
        remaining = []
        for did, s in first_stage_docs[best_k:]:
            if did not in seen:
                seen.add(did)
                fallback_score = top_k_pairs[-1][1] - 1.0 - len(remaining)
                remaining.append((did, fallback_score))

        reranked_eval[cid] = (top_k_pairs + remaining)[:100]

    eval_rerank_path = os.path.join(PRED_DIR, "t3_rerank.trec")
    write_trec_file(eval_rerank_path, reranked_eval, "t3_rerank", eval_claim_ids, corpus_doc_ids, top_k=100)
    check_trec(eval_rerank_path, set(eval_claim_ids), set(corpus_doc_ids))
    print(f"Verified & saved: {eval_rerank_path}")

    # Save metrics for Task 3
    task3_metrics = {
        "hybrid_method": f"Min-Max Normalization (Dense weight={best_alpha:.2f}, BM25 weight={1.0-best_alpha:.2f})",
        "reranker_model": reranker_name,
        "rerank_top_k": best_k,
        "val_ndcg10_hybrid": round(best_val_ndcg, 4),
        "val_ndcg10_rerank": round(best_rerank_ndcg, 4),
        "rerank_query_latency_ms": round(best_latency, 2),
    }
    with open(os.path.join(DATA_DIR, "task3_metrics.json"), "w") as f:
        json.dump(task3_metrics, f, indent=2)

    return task3_metrics


def main():
    corpus = load_corpus(os.path.join(DATA_DIR, "corpus.jsonl"))
    val_claims = load_claims(os.path.join(DATA_DIR, "val_claims.jsonl"))
    eval_claims = load_claims(os.path.join(DATA_DIR, "eval_claims.jsonl"))
    val_gold = get_gold_evidence(val_claims)

    best_alpha, best_hybrid_ndcg, val_hybrid_best, eval_hybrid_best = sweep_hybrid(
        corpus, val_claims, eval_claims, val_gold
    )

    task3_metrics = run_cross_encoder_rerank(
        corpus,
        val_claims,
        eval_claims,
        val_gold,
        val_hybrid_best,
        eval_hybrid_best,
        best_val_ndcg=best_hybrid_ndcg,
        best_alpha=best_alpha,
    )


if __name__ == "__main__":
    main()
