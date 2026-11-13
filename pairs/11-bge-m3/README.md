# M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation

Pair: Oliver Wakeford, Mathis Lesgourgues. Method: BGE-M3, one encoder that outputs dense, sparse and multi-vector representations, evaluated head by head and fused with the paper's weighted sum (1·dense + 0.3·sparse + 1·multi-vec). How to run, in at most 10 lines (BGE-M3 has 568M parameters: run on a GPU, Onyxia or Colab):

```
pip install -r pairs/11-bge-m3/requirements.txt  
python pairs/11-bge-m3/method.py                    # mode All → runs/
python pairs/11-bge-m3/method.py --mode dense       # or sparse, multi-vec, dense+sparse → runs_<mode>/
python testbed.py score pairs/11-bge-m3/runs --method "BGE-M3 (All)" --pair 11-bge-m3 --against rrf
python testbed.py score pairs/11-bge-m3/runs_dense --method "BGE-M3 (Dense)" --pair 11-bge-m3 --against rrf --out pairs/11-bge-m3/results_dense.json
python pairs/11-bge-m3/translate_scifact.py --report   # share of correct translations in check_30.csv
python pairs/11-bge-m3/crosslingual.py               # French vs English queries, each mode → results_crosslingual.json
python pairs/11-bge-m3/tune_weights.py               # fusion weights tuned on SciFact train → results_weights.json
```

## Files

| File | What it is |
|---|---|
| `method.py` | the five retrieval modes (`--mode`, default `all`) |
| `results.json` | mode All, the row of the shared table |
| `results_<mode>.json` | the other modes, kept out of the shared table |
| `translate_scifact.py` | translates the 300 SciFact test queries with `facebook/nllb-200-distilled-600M` |
| `queries_fr.json` | the French queries, same ids as the English ones |
| `check_30.csv` | the 30 most technical queries checked by hand (1 = correct, 0 = wrong) |
| `crosslingual.py`, `results_crosslingual.json` | each mode with French and English SciFact queries, French − English with p and CI |
| `tune_weights.py`, `results_weights.json` | mode All's weights tuned on the 809 SciFact training queries, applied once to test |
| `error_analysis.md` | the queries French loses most on, per mode |

## Results (mode All)

| Dataset | nDCG@10 | Recall@100 |
|---|:-:|:-:|
| SciFact | 0.692 | 0.914 |
| NFCorpus | 0.337 | 0.289 |
| ArguAna | 0.528 | 0.979 |

## Translation check

26 of the 30 checked queries are correct. The 4 errors change a technical term or the meaning: *penetrance* → « pénétration », *mottle* → « mouette », the acronym AMI → MMI, and *quiescent* attached to the follicle instead of the stem cells.

## Cross-lingual (SciFact, French queries, English corpus)

| Mode | nDCG@10 EN | nDCG@10 FR | Δ [95% CI] | p |
|---|:-:|:-:|:-:|:-:|
| Dense | 0.633 | 0.586 | −0.047 [−0.069, −0.028] | <10⁻⁴ |
| Sparse | 0.637 | 0.442 | −0.195 [−0.237, −0.153] | <10⁻¹⁷ |
| Multi-vec | 0.683 | 0.634 | −0.049 [−0.072, −0.028] | <10⁻⁴ |
| Dense+Sparse | 0.667 | 0.606 | −0.062 [−0.085, −0.040] | <10⁻⁷ |
| All | 0.692 | 0.634 | −0.057 [−0.081, −0.035] | <10⁻⁵ |

Sparse needs shared tokens, so it loses most. On the 26 queries checked as correct the drops are small and not significant (All −0.008, p = 0.68), but those queries were picked for their acronyms, which match in both languages. See `error_analysis.md`.

## Tuned weights (SciFact)

Grid over w_lex ∈ {0, 0.1, 0.2, 0.3, 0.5, 1} and w_mul ∈ {0, 0.5, 1, 1.5, 2}, w_dense = 1, on the training queries only. The grid is flat: every w_mul ≥ 0.5 is within 0.007 of the best. The best, (1, 0.1, 1.5), gives 0.689 nDCG@10 on test against 0.692 for the paper's (1, 0.3, 1): −0.002 [−0.009, +0.005], p = 0.53. The paper's weights stay the default.
