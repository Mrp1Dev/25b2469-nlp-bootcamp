# SciFact Claim Retrieval and Verification Pipeline

**Author**: Mehraj Pranami  
**Roll No**: 25b2469  
**Hardware Environment**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), CUDA 12.4, Python 3.11  

---

## 1. Setup & Environment

All models run within Python 3.11 using PyTorch with CUDA 12.4.

```bash
# Set cache directories
export HF_HOME="D:/hf_cache"

# Install required dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers sentence-transformers rank-bm25 scikit-learn accelerate datasets
```

---

## 2. Reproduction Commands (Tasks 1 to 5)

All scripts are located in `code/` and can be executed either from the repository root:
```bash
python code/run_task1.py
python code/run_task2.py
python code/run_task3.py
python code/run_task4.py
python code/run_task5.py
```
or directly from inside the `code/` directory:
```bash
cd code
python run_task1.py
python run_task2.py
python run_task3.py
python run_task4.py
python run_task5.py
```

### Task 1: Lexical Baseline (BM25)
* Explores corpus and claim statistics.
* Tunes BM25 parameters ($k_1=1.0, b=0.5$) on validation split.
* Generates `predictions/val_bm25.trec` and `predictions/t1_bm25.trec`.
```bash
python code/run_task1.py
```

### Task 2: Dense Retrieval
* Evaluates general dense model (`BAAI/bge-base-en-v1.5`) and domain scientific model (`pritamdeka/S-PubMedBert-MS-MARCO`).
* Generates `predictions/val_general.trec`, `predictions/t2_general.trec`, and `predictions/t2_scientific.trec`.
* Logs latency and memory to `data/task2_metrics.json`.
```bash
python code/run_task2.py
```

### Task 3: Combine & Rerank
* Sweeps score interpolation alpha between BGE and BM25 (optimal $\alpha=0.70$).
* Reranks top-$k$ candidates with `Alibaba-NLP/gte-reranker-modernbert-base` ($k=15$).
* Generates `predictions/t3_hybrid.trec` and `predictions/t3_rerank.trec`.
* Logs metrics to `data/task3_metrics.json`.
```bash
python code/run_task3.py
```

### Task 4: Domain Fine-Tuning
* Mines hard negatives from BM25 and BGE false positive candidates.
* Fine-tunes `BAAI/bge-base-en-v1.5` twice: Model A (random negatives) and Model B (hard negatives).
* Generates `predictions/t4_random.trec` and `predictions/t4_hard.trec`.
```bash
python code/run_task4.py
```

### Task 5: Claim Verification & Error Analysis
* Fine-tunes `tasksource/ModernBERT-base-nli` on retriever-conditioned SciFact training pairs.
* Predicts `SUPPORT`, `CONTRADICT`, or `NEI` on post-reranked top-3 candidates.
* Generates:
  * `predictions/t5_val_verification.csv`
  * `predictions/t5_verification.csv`
  * `predictions/t5_val_errors.csv`
* Logs verification metrics and error categorization to `data/task5_metrics.json`.
```bash
python code/run_task5.py
```

---

## 3. Format Verification
To verify the entire submission format and all generated runs:
```bash
python check_format.py .
```
Expected output:
```
Task completeness (1.0 = all files valid)
  task1: 1.00
  task2: 1.00
  task3: 1.00
  task4: 1.00
  task5: 1.00
All good.
```

