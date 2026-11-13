# M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation

Pair: Oliver Wakeford, Mathis Lesgourgues. Method: BGE-M3, one encoder that outputs dense, sparse and multi-vector representations, evaluated head by head and fused with the paper's weighted sum (1·dense + 0.3·sparse + 1·multi-vec). How to run, in at most 10 lines (BGE-M3 has 568M parameters: run on a GPU, Onyxia or Colab):

```
pip install -r pairs/11-bge-m3/requirements.txt  
python pairs/11-bge-m3/method.py                    # mode All → runs/
python pairs/11-bge-m3/method.py --mode dense       # or sparse, multi-vec, dense+sparse → runs_<mode>/
python testbed.py score pairs/11-bge-m3/runs --method "BGE-M3 (All)" --pair 11-bge-m3 --against rrf
python testbed.py score pairs/11-bge-m3/runs_dense --method "BGE-M3 (Dense)" --pair 11-bge-m3 --against rrf --out pairs/11-bge-m3/results_dense.json
python pairs/11-bge-m3/translate_scifact.py --report   # share of correct translations in check_30.csv
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

## Results (mode All)

| Dataset | nDCG@10 | Recall@100 |
|---|:-:|:-:|
| SciFact | 0.692 | 0.914 |
| NFCorpus | 0.337 | 0.289 |
| ArguAna | 0.528 | 0.979 |

## Translation check

26 of the 30 checked queries are correct. The 4 errors change a technical term or the meaning: *penetrance* → « pénétration », *mottle* → « mouette », the acronym AMI → MMI, and *quiescent* attached to the follicle instead of the stem cells.
