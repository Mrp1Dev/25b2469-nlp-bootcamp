"""Driver script for Task 2: Dense Retrieval.

Runs:
1. General-purpose model: 'BAAI/bge-base-en-v1.5' (109M params)
2. Scientific/biomedical model: 'pritamdeka/S-PubMedBert-MS-MARCO' (110M params)

Generates:
- predictions/val_general.trec (for Q1 comparison)
- predictions/t2_general.trec (eval claims)
- predictions/t2_scientific.trec (eval claims)

Logs index time, query latency, and memory usage for results.json.
"""

import os
import sys
import json
import gc
import torch

# Ensure repository root and code/ directory are in sys.path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR) if os.path.basename(CODE_DIR) == "code" else CODE_DIR
for p in [CODE_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_loader import load_corpus, load_claims, get_gold_evidence
from src.metrics import evaluate_retrieval
from src.dense_retriever import DenseRetriever
from src.trec_utils import write_trec_file
from check_format import check_trec

DATA_DIR = os.path.join(ROOT_DIR, "data")
PRED_DIR = os.path.join(ROOT_DIR, "predictions")


def run():
    print("=" * 65)
    print("TASK 2: DENSE RETRIEVAL (GENERAL VS. SCIENTIFIC)")
    print("=" * 65)

    corpus = load_corpus(os.path.join(DATA_DIR, "corpus.jsonl"))
    val_claims = load_claims(os.path.join(DATA_DIR, "val_claims.jsonl"))
    eval_claims = load_claims(os.path.join(DATA_DIR, "eval_claims.jsonl"))
    val_gold = get_gold_evidence(val_claims)

    val_claim_ids = [int(c["id"]) for c in val_claims]
    eval_claim_ids = [int(c["id"]) for c in eval_claims]
    corpus_doc_ids = list(corpus.keys())

    models = [
        {
            "type": "general",
            "name": "BAAI/bge-base-en-v1.5",
            "eval_trec": os.path.join(PRED_DIR, "t2_general.trec"),
            "val_trec": os.path.join(PRED_DIR, "val_general.trec"),
            "run_name": "t2_general",
        },
        {
            "type": "scientific",
            "name": "pritamdeka/S-PubMedBert-MS-MARCO",
            "eval_trec": os.path.join(PRED_DIR, "t2_scientific.trec"),
            "val_trec": None,
            "run_name": "t2_scientific",
        },
    ]

    task2_results = {}

    for cfg in models:
        mtype = cfg["type"]
        mname = cfg["name"]
        print(f"\n[{mtype.upper()} MODEL] {mname}")
        print("-" * 55)

        retriever = DenseRetriever(mname)

        # 1. Build / load index
        doc_ids, doc_embeddings, index_time, peak_mem = retriever.build_or_load_index(
            corpus, batch_size=64
        )

        # 2. Retrieve for validation claims
        val_preds, val_latency = retriever.batch_retrieve(
            val_claims, doc_ids, doc_embeddings, top_k=100
        )
        val_pred_ids = {cid: [did for did, score in doc_list] for cid, doc_list in val_preds.items()}
        summary, _ = evaluate_retrieval(val_pred_ids, val_gold)

        print(f"  Validation nDCG@10: {summary['ndcg10']:.4f}")
        print(f"  Validation Rec@100: {summary['recall100']:.4f}")
        print(f"  Validation MRR:     {summary['mrr']:.4f}")
        print(f"  Indexing Time:      {index_time:.2f} s")
        print(f"  Query Latency:      {val_latency:.2f} ms/query")
        print(f"  Peak GPU Memory:    {peak_mem:.1f} MB")

        # 3. Retrieve for eval claims
        eval_preds, eval_latency = retriever.batch_retrieve(
            eval_claims, doc_ids, doc_embeddings, top_k=100
        )

        # 4. Save validation TREC if needed (for general model)
        if cfg["val_trec"]:
            write_trec_file(
                output_path=cfg["val_trec"],
                predictions=val_preds,
                run_name="val_general",
                all_claim_ids=val_claim_ids,
                corpus_doc_ids=corpus_doc_ids,
                top_k=100,
            )
            print(f"  Saved: {cfg['val_trec']}")
            check_trec(cfg["val_trec"], set(val_claim_ids), set(corpus_doc_ids))
            print(f"  [OK] {cfg['val_trec']} verified.")

        # 5. Save eval TREC
        write_trec_file(
            output_path=cfg["eval_trec"],
            predictions=eval_preds,
            run_name=cfg["run_name"],
            all_claim_ids=eval_claim_ids,
            corpus_doc_ids=corpus_doc_ids,
            top_k=100,
        )
        print(f"  Saved: {cfg['eval_trec']}")
        check_trec(cfg["eval_trec"], set(eval_claim_ids), set(corpus_doc_ids))
        print(f"  [OK] {cfg['eval_trec']} verified.")

        task2_results[mtype] = {
            "model_name": mname,
            "val_ndcg10": summary["ndcg10"],
            "val_recall100": summary["recall100"],
            "val_mrr": summary["mrr"],
            "index_time_sec": round(index_time, 2),
            "query_latency_ms": round(val_latency, 2),
            "memory_mb": round(peak_mem, 1),
        }

        # Clean up GPU memory before loading next model
        del retriever
        del doc_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary JSON dump for easy reference
    metrics_path = os.path.join(DATA_DIR, "task2_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(task2_results, f, indent=2)

    print("\n" + "=" * 65)
    print("TASK 2 COMPLETE - SUMMARY COMPARISON")
    print("=" * 65)
    for mtype, res in task2_results.items():
        print(f"  [{mtype.capitalize()}] {res['model_name']}:")
        print(f"      Val nDCG@10:    {res['val_ndcg10']:.4f}")
        print(f"      Val Recall@100: {res['val_recall100']:.4f}")
        print(f"      Index Time:     {res['index_time_sec']} s")
        print(f"      Query Latency:  {res['query_latency_ms']} ms")
        print(f"      Peak GPU VRAM:  {res['memory_mb']} MB")
    print("=" * 65)


if __name__ == "__main__":
    run()
