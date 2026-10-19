# SciFact Claim Retrieval and Verification Report

**Author**: Mehraj Pranami  
**Roll No**: 25b2469  

---

### Q1. Give two claim IDs where BM25 has higher per-claim nDCG@10 than t2_general, and two where t2_general beats BM25. Explain why.

Two claims in which BM25 beats t2_general: Claim 673 (1.000 vs t2_general's 0.356) and Claim 381 (1.000 vs 0.500). These claims contain phrases like "LDL cholesterol" or "environmental factors" that show up in the gold abstracts, which BM25 is good at. The dense t2_general gets distracted by similar concept papers instead.

Two claims where t2_general beats BM25: Claim 1223 and Claim 1311 (1.000 vs 0.000). The gold abstracts say the same thing in different words, like "malaria parasitemia clearance" instead of "removing parasites" (1223). With no word overlap, BM25 misses them, but the dense embeddings are similar.

---

### Q2. Which Task 3 component gave the largest validation gain? What did it cost in latency?

The cross-encoder ReRanker (`Alibaba-NLP/gte-reranker-modernbert-base` with k=15) gave the biggest validation gain. It raised nDCG@10 from 0.9017 (70-30 BGE-BM25 hybrid) to 0.9299, an increase of 2.8 points. For comparison, that hybrid only improved pure BGE from 0.8910 to 0.9017, just 1.1 points.

In latency cost, this wasn't unsubstantial though. Hybrid fusion was just 1ms per query, but the cross encoder took 308.76ms, which is 300x slower, because of its cross attention over the claim and each of the 15 abstracts. Might be worth it though, it gave me the best score out of all the tasks in this repo.

---

### Q3. Did hard negatives help? Did you see overfitting, and how did you detect it?

Yes, hard negatives did help. I found them by looking for false positives in BM25 and BGE's rankings, and fine tuning the BGE on them in task 4 gave 1.9 points over random negatives, nDCG@10 going from 0.9098 to 0.9285.

Yes, i did see overfitting in task 4. Over 3 epochs, validation nDCG@10 peaked at epoch 2 and went down (0.9285->0.9188) in epoch 3, despite training loss going down (0.078->0.043). I detected it by evaluating on the val set every epoch and watching val nDCG@10 fall while training loss kept going down. That is standard overfitting.

---

### Q4. What fraction of your validation verification errors are retrieval_miss?

Out of 56 error claims, 3 were retrieval_miss, so 5.36%. Which means in ~95% of the cases, the retrieval algorithm (the best one, ie task 3 reranker) fetched the correct evidence within the top 3, but the verifier's reasoning led to a wrong classification about what it was. The ones it got wrong were where it struggled with borderline neutral predictions with ~0.5 confidence, or confusing double negatives in papers, or similar sounding genes/organisms.

---

### Q5. What is the single change you would try next, and why?

Well, you disallowed LLMs (decoders) for the assignment, but I would otherwise use CoT capable LLMs to go through the abstracts when classifying, so double negatives and confusing language can be dealt with using their intelligence.

But, sticking strictly to the rules, sentence-level rationale extraction should be useful. Right now the verifier reads the entire abstract, and the more irrelevant sentences there are the more confused it gets. A small sentence selector that pulls out the top one/two rationale sentences would reduce the noise for the ModernBERT NLI verifier, and end to end verification F1 should go up.
