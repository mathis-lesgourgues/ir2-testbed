"""Cross-lingual check: the 300 SciFact test queries in English and in French, against the English corpus, for each mode.

Usage: python pairs/11-bge-m3/crosslingual.py   → runs/crosslingual/ and results_crosslingual.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import testbed
import method

RUNS = HERE / "runs" / "crosslingual"
OUT = HERE / "results_crosslingual.json"
N_WORST = 5


def get_run(mode, lang, queries):
    # runs are saved, so scoring can be redone without encoding again
    path = RUNS / f"{mode.replace('+', '_')}_{lang}.json"
    if not path.exists():
        testbed.save_run(method.search("scifact", mode, queries), path)
    return json.load(open(path))


def summary(ndcg, recall, ndcg_en, recall_en):
    return {"ndcg10": round(float(ndcg.mean()), 4), "recall100": round(float(recall.mean()), 4),
            "ndcg10_vs_en": testbed.compare(ndcg, ndcg_en), "recall100_vs_en": testbed.compare(recall, recall_en)}


def main():
    _, queries_en, _ = testbed.load("scifact")
    queries_fr = json.load(open(HERE / "queries_fr.json", encoding="utf-8"))
    checked = {r["qid"] for r in csv.DictReader(open(HERE / "check_30.csv", encoding="utf-8")) if r["correct"] == "1"}

    results = {}
    for mode in method.MODES:
        qids, ndcg_en, recall_en = testbed.score_run(get_run(mode, "en", queries_en), "scifact")
        _, ndcg_fr, recall_fr = testbed.score_run(get_run(mode, "fr", queries_fr), "scifact")
        ok = np.array([q in checked for q in qids])       # the hand-checked queries translated correctly

        # queries that lose the most nDCG@10 in French, for the error analysis
        loss = ndcg_en - ndcg_fr
        worst = [{"qid": qids[i], "ndcg10_en": round(float(ndcg_en[i]), 3), "ndcg10_fr": round(float(ndcg_fr[i]), 3),
                  "en": queries_en[qids[i]], "fr": queries_fr[qids[i]]} for i in np.argsort(-loss)[:N_WORST]]

        results[mode] = {
            "en": {"ndcg10": round(float(ndcg_en.mean()), 4), "recall100": round(float(recall_en.mean()), 4)},
            "fr": summary(ndcg_fr, recall_fr, ndcg_en, recall_en),
            "fr_checked": dict(n_queries=int(ok.sum()), **summary(ndcg_fr[ok], recall_fr[ok], ndcg_en[ok], recall_en[ok])),
            "worst": worst,
        }

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)

    print("| Mode | nDCG@10 EN | nDCG@10 FR | Δ [95% CI] | p | Recall@100 EN | Recall@100 FR |")
    print("|---|:-:|:-:|:-:|:-:|:-:|:-:|")
    for mode, r in results.items():
        c = r["fr"]["ndcg10_vs_en"]
        print(f"| {mode} | {r['en']['ndcg10']:.3f} | {r['fr']['ndcg10']:.3f} | {c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
              f"| {c['p']:.2g} | {r['en']['recall100']:.3f} | {r['fr']['recall100']:.3f} |")
    for mode, r in results.items():
        c = r["fr_checked"]["ndcg10_vs_en"]
        print(f"{mode}, {r['fr_checked']['n_queries']} checked queries: Δ nDCG@10 {c['delta']:+.3f}, p {c['p']:.2g}")


if __name__ == "__main__":
    main()
