"""Fusion weights of mode All, tuned on the 809 SciFact training queries, then applied once to the 300 test queries.

Usage: python pairs/11-bge-m3/tune_weights.py   → results_weights.json
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytrec_eval

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import testbed
import method

W_LEX = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
W_MUL = [0, 0.5, 1.0, 1.5, 2.0]          # the dense weight stays at 1, as in the paper
PAPER = (method.W_LEX, method.W_MUL)
OUT = HERE / "results_weights.json"


def load_train():
    """SciFact training queries and their qrels."""
    folder = testbed.DATA / "scifact"
    qrels = {}
    with open(folder / "qrels" / "train.tsv") as f:
        next(f)
        for line in f:
            q, d, grade = line.split("\t")
            qrels.setdefault(q, {})[d] = int(grade)
    queries = {}
    with open(folder / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in qrels:
                queries[q["_id"]] = q["text"]
    return queries, qrels


def candidate_scores(queries, D_dense, D_lex, D_mul, vocab):
    """The dense top-N_CAND of each query, with the three scores of each candidate, as in mode All."""
    qids = list(queries)
    Q_dense, Q_lex, Q_mul = method.encode([queries[q] for q in qids])
    S_dense = Q_dense @ D_dense.T
    S_lex = (method.to_sparse(Q_lex, vocab) @ D_lex.T).toarray()
    scores = {}
    for i, q in enumerate(qids):
        cand = method.top(S_dense[i], method.N_CAND)
        s_mul = method.colbert_batch(Q_mul[i], [D_mul[j] for j in cand])
        scores[q] = (cand, S_dense[i, cand], S_lex[i, cand], s_mul)
    return scores


def fuse(scores, ids, w_lex, w_mul):
    run = {}
    for q, (cand, s_dense, s_lex, s_mul) in scores.items():
        s = s_dense + w_lex * s_lex + w_mul * s_mul
        order = np.argsort(-s)[:testbed.K]
        run[q] = {ids[cand[k]]: float(s[k]) for k in order}
    return run


def train_ndcg(run, qrels):
    # ranks instead of scores, as in testbed.score_run (pytrec_eval breaks ties by id)
    ranked = {q: {d: float(testbed.K - i) for i, d in enumerate(sorted(r, key=r.get, reverse=True))} for q, r in run.items()}
    s = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(ranked)
    return float(np.mean([s[q]["ndcg_cut_10"] for q in qrels]))


def main():
    corpus, test_queries, _ = testbed.load("scifact")
    ids = list(corpus)
    vocab = len(method.model.tokenizer)
    D_dense, D_lex, D_mul = method.encode(testbed.doc_texts(corpus))
    D_lex = method.to_sparse(D_lex, vocab)

    # grid on the training queries only
    train_queries, train_qrels = load_train()
    train = candidate_scores(train_queries, D_dense, D_lex, D_mul, vocab)
    grid = {w: train_ndcg(fuse(train, ids, *w), train_qrels) for w in itertools.product(W_LEX, W_MUL)}
    best = max(grid, key=grid.get)

    # the test queries are scored once, with the paper's weights and with the tuned ones
    test = candidate_scores(test_queries, D_dense, D_lex, D_mul, vocab)
    _, ndcg_paper, recall_paper = testbed.score_run(fuse(test, ids, *PAPER), "scifact")
    _, ndcg_tuned, recall_tuned = testbed.score_run(fuse(test, ids, *best), "scifact")

    results = {
        "n_train_queries": len(train_queries),
        "grid": [{"w_lex": wl, "w_mul": wm, "train_ndcg10": round(v, 4)} for (wl, wm), v in grid.items()],
        "paper": {"w_lex": PAPER[0], "w_mul": PAPER[1], "train_ndcg10": round(grid[PAPER], 4),
                  "test_ndcg10": round(float(ndcg_paper.mean()), 4), "test_recall100": round(float(recall_paper.mean()), 4)},
        "tuned": {"w_lex": best[0], "w_mul": best[1], "train_ndcg10": round(grid[best], 4),
                  "test_ndcg10": round(float(ndcg_tuned.mean()), 4), "test_recall100": round(float(recall_tuned.mean()), 4)},
        "tuned_vs_paper": {"ndcg10": testbed.compare(ndcg_tuned, ndcg_paper), "recall100": testbed.compare(recall_tuned, recall_paper)},
    }
    json.dump(results, open(OUT, "w"), indent=1)

    print("train nDCG@10, rows w_lex, columns w_mul", W_MUL)
    for wl in W_LEX:
        print(f"{wl:>4}", " ".join(f"{grid[(wl, wm)]:.4f}" for wm in W_MUL))
    for name in ("paper", "tuned"):
        r = results[name]
        print(f"{name}: w_lex {r['w_lex']}, w_mul {r['w_mul']} → test nDCG@10 {r['test_ndcg10']:.4f}, Recall@100 {r['test_recall100']:.4f}")
    c = results["tuned_vs_paper"]["ndcg10"]
    print(f"tuned − paper, nDCG@10: {c['delta']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], p {c['p']:.2g}")


if __name__ == "__main__":
    main()
