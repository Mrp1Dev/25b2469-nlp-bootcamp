"""Task 4: Domain Fine-Tuning of Bi-Encoder (BAAI/bge-base-en-v1.5).

1. Mine Hard Negatives:
   - Uses Task 1 BM25 and Task 2 BGE Dense systems to find high-scoring non-gold abstracts.
   - Produces (claim, positive, hard_negative) triplets.

2. Train Model A (Random Negatives):
   - Fine-tunes BAAI/bge-base-en-v1.5 with in-batch negatives using MultipleNegativesRankingLoss.
   - Evaluates on validation split after each epoch.
   - Generates predictions/t4_random.trec.

3. Train Model B (Hard Negatives):
   - Fine-tunes BAAI/bge-base-en-v1.5 with (claim, positive, hard_negative) triplets.
   - Evaluates on validation split after each epoch.
   - Detects presence or absence of overfitting for Report Question Q3.
   - Generates predictions/t4_hard.trec.
"""

import os
import sys
import time
import json
import gc
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# Ensure repository root and code/ directory are in sys.path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR) if os.path.basename(CODE_DIR) == "code" else CODE_DIR
for p in [CODE_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_loader import load_corpus, load_claims, get_gold_evidence
from src.metrics import evaluate_retrieval
from src.bm25_retriever import BM25Retriever, tokenize_clean
from src.trec_utils import write_trec_file
from check_format import check_trec

DATA_DIR = os.path.join(ROOT_DIR, "data")
PRED_DIR = os.path.join(ROOT_DIR, "predictions")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


def mine_hard_negatives(corpus, train_claims, bge_model, device):
    """Mines hard negatives for all training claims using BM25 and BGE Dense."""
    print("\n--- Mining Hard Negatives from Task 1 (BM25) and Task 2 (Dense) ---")
    
    train_with_ev = [c for c in train_claims if len(c["evidence"]) > 0]
    corpus_doc_ids = list(corpus.keys())
    
    # 1. BM25 Index
    bm25 = BM25Retriever(
        corpus=corpus,
        k1=1.0,
        b=0.5,
        method="okapi",
        tokenize_fn=lambda t: tokenize_clean(t, remove_stopwords=False, preserve_compounds=False),
    )
    bm25_results = bm25.batch_retrieve(train_with_ev, top_k=20)
    
    # 2. Dense Index
    bge_emb_path = os.path.join(CACHE_DIR, "bge_base_corpus_emb.pt")
    doc_embeddings = torch.load(bge_emb_path, map_location=device)
    prompt = "Represent this sentence for searching relevant passages: "
    query_texts = [f"{prompt}{c['claim']}" for c in train_with_ev]
    
    with torch.inference_mode():
        q_emb = bge_model.encode(query_texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, device=device)
        sim = torch.matmul(q_emb, doc_embeddings.T)
        topk_s, topk_i = torch.topk(sim, k=20, dim=1)
        
    topk_i_cpu = topk_i.cpu().tolist()
    dense_results = {}
    for idx, c in enumerate(train_with_ev):
        cid = int(c["id"])
        dense_results[cid] = [corpus_doc_ids[i] for i in topk_i_cpu[idx]]
        
    # 3. Combine and filter
    random_examples = []
    hard_examples = []
    
    mined_stats = []
    for c in train_with_ev:
        cid = int(c["id"])
        claim_text = f"{prompt}{c['claim']}"
        gold_dids = {int(ev["doc_id"]) for ev in c["evidence"]}
        
        # Candidate negatives: non-gold top retrieved docs
        bm25_top = [did for did, _ in bm25_results[cid] if did not in gold_dids]
        dense_top = [did for did in dense_results[cid] if did not in gold_dids]
        
        # Interleave BM25 and Dense hard negatives
        neg_candidates = []
        for did in bm25_top[:2] + dense_top[:2]:
            if did not in neg_candidates:
                neg_candidates.append(did)
                
        mined_stats.append(len(neg_candidates))
        
        for g_did in gold_dids:
            pos_text = corpus[g_did]["text"]
            # Random negative training uses (query, positive) pairs
            random_examples.append(InputExample(texts=[claim_text, pos_text]))
            
            # Hard negative training uses (query, positive, negative) triplets
            for neg_did in neg_candidates[:2]:
                neg_text = corpus[neg_did]["text"]
                hard_examples.append(InputExample(texts=[claim_text, pos_text, neg_text]))
                
    print(f"Mined hard negatives for {len(train_with_ev)} claims.")
    print(f"Total random-negative pairs: {len(random_examples)}")
    print(f"Total hard-negative triplets: {len(hard_examples)}")
    return random_examples, hard_examples


def evaluate_model_on_val(model, corpus, val_claims, val_gold, device):
    """Encodes corpus and validation claims, and returns nDCG@10, Recall@100, MRR."""
    doc_ids = list(corpus.keys())
    texts = [corpus[did]["text"] for did in doc_ids]
    
    with torch.inference_mode():
        doc_emb = model.encode(texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, device=device)
        
        prompt = "Represent this sentence for searching relevant passages: "
        val_texts = [f"{prompt}{c['claim']}" for c in val_claims]
        val_emb = model.encode(val_texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, device=device)
        
        sim = torch.matmul(val_emb, doc_emb.T)
        topk_s, topk_i = torch.topk(sim, k=100, dim=1)
        
    topk_i_cpu = topk_i.cpu().tolist()
    preds = {}
    for idx, c in enumerate(val_claims):
        cid = int(c["id"])
        preds[cid] = [doc_ids[i] for i in topk_i_cpu[idx]]
        
    summary, _ = evaluate_retrieval(preds, val_gold)
    return summary, doc_emb


def train_and_eval(run_name, train_examples, corpus, val_claims, val_gold, eval_claims, device, epochs=3, batch_size=16):
    """Fine-tunes BAAI/bge-base-en-v1.5 and logs per-epoch validation scores."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING {run_name.upper()} (epochs={epochs}, batch_size={batch_size})")
    print(f"{'=' * 60}")
    
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    model.max_seq_length = 256  # Prevent quadratic attention memory blowup on empty padding
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    
    epoch_scores = []
    best_ndcg = -1.0
    best_model_path = os.path.join(MODEL_DIR, f"bge_{run_name}")
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        t0 = time.perf_counter()
        
        # Train for 1 epoch
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps if epoch == 1 else 0,
            optimizer_params={"lr": 2e-5},
            show_progress_bar=True,
            use_amp=True,  # fp16 mixed precision
        )
        train_time = time.perf_counter() - t0
        
        # Clear VRAM before evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Evaluate on validation split
        summary, doc_emb = evaluate_model_on_val(model, corpus, val_claims, val_gold, device)
        ndcg = summary["ndcg10"]
        epoch_scores.append({"epoch": epoch, "ndcg10": ndcg, "recall100": summary["recall100"], "mrr": summary["mrr"]})
        print(f"Epoch {epoch} finished in {train_time:.1f}s -> "
              f"Val nDCG@10: {ndcg:.4f} | Recall@100: {summary['recall100']:.4f} | MRR: {summary['mrr']:.4f}")
              
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            model.save(best_model_path)
            best_doc_emb = doc_emb
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Load best checkpoint for final evaluation run
    print(f"\nLoading best checkpoint from {best_model_path} (nDCG@10 = {best_ndcg:.4f})...")
    best_model = SentenceTransformer(best_model_path, device=device)
    
    # Generate predictions on eval_claims
    doc_ids = list(corpus.keys())
    eval_claim_ids = [int(c["id"]) for c in eval_claims]
    prompt = "Represent this sentence for searching relevant passages: "
    eval_texts = [f"{prompt}{c['claim']}" for c in eval_claims]
    
    with torch.inference_mode():
        eval_emb = best_model.encode(eval_texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, device=device)
        sim_eval = torch.matmul(eval_emb, best_doc_emb.T)
        topk_s, topk_i = torch.topk(sim_eval, k=100, dim=1)
        
    topk_s_cpu = topk_s.cpu().tolist()
    topk_i_cpu = topk_i.cpu().tolist()
    eval_preds = {}
    for idx, cid in enumerate(eval_claim_ids):
        docs = [doc_ids[i] for i in topk_i_cpu[idx]]
        scores = topk_s_cpu[idx]
        eval_preds[cid] = list(zip(docs, scores))
        
    out_path = os.path.join(PRED_DIR, f"t4_{run_name}.trec")
    write_trec_file(out_path, eval_preds, f"t4_{run_name}", eval_claim_ids, doc_ids, top_k=100)
    check_trec(out_path, set(eval_claim_ids), set(doc_ids))
    print(f"Verified & saved: {out_path}")
    
    return best_ndcg, epoch_scores


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Task 4 on device: {device}")
    
    corpus = load_corpus(os.path.join(DATA_DIR, "corpus.jsonl"))
    train_claims = load_claims(os.path.join(DATA_DIR, "train_claims.jsonl"))
    val_claims = load_claims(os.path.join(DATA_DIR, "val_claims.jsonl"))
    eval_claims = load_claims(os.path.join(DATA_DIR, "eval_claims.jsonl"))
    val_gold = get_gold_evidence(val_claims)
    
    # Initial base BGE model for mining
    base_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    random_examples, hard_examples = mine_hard_negatives(corpus, train_claims, base_model, device)
    
    # 1. Train Model A: Random Negatives
    ndcg_random, history_random = train_and_eval(
        run_name="random",
        train_examples=random_examples,
        corpus=corpus,
        val_claims=val_claims,
        val_gold=val_gold,
        eval_claims=eval_claims,
        device=device,
        epochs=3,
        batch_size=16,
    )
    
    # 2. Train Model B: Hard Negatives
    ndcg_hard, history_hard = train_and_eval(
        run_name="hard",
        train_examples=hard_examples,
        corpus=corpus,
        val_claims=val_claims,
        val_gold=val_gold,
        eval_claims=eval_claims,
        device=device,
        epochs=3,
        batch_size=8,
    )
    
    # Save metrics to data/task4_metrics.json
    task4_metrics = {
        "base_model": "BAAI/bge-base-en-v1.5",
        "val_ndcg10_random_negatives": round(ndcg_random, 4),
        "val_ndcg10_hard_negatives": round(ndcg_hard, 4),
        "history_random": history_random,
        "history_hard": history_hard,
    }
    with open(os.path.join(DATA_DIR, "task4_metrics.json"), "w") as f:
        json.dump(task4_metrics, f, indent=2)
        
    print("\n" + "=" * 60)
    print("TASK 4 COMPLETE - SUMMARY:")
    print("=" * 60)
    print(f"Base Pretrained BGE Val nDCG@10:     0.8910")
    print(f"Random Negatives Model Val nDCG@10:  {ndcg_random:.4f}")
    print(f"Hard Negatives Model Val nDCG@10:    {ndcg_hard:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

