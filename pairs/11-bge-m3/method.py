"""BGE-M3 on the testbed, with the five retrieval modes of the paper (section 4.1).

Modes:
    dense          rank all docs by s_dense
    sparse         rank all docs by s_lex
    multi-vec      rerank the dense top-200 by s_mul
    dense+sparse   rerank the union of dense top-1000 and sparse top-1000 by 1·s_dense + 0.3·s_lex
    all (default)  rerank the dense top-200 by 1·s_dense + 0.3·s_lex + 1·s_mul

Usage: python method.py                          → all datasets, mode all   → runs/
       python method.py arguana --mode sparse    → one dataset, one mode    → runs_sparse/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # repo root, for testbed.py
import testbed
from FlagEmbedding import BGEM3FlagModel

MODES = ["dense", "sparse", "multi-vec", "dense+sparse", "all"]
W_DENSE, W_LEX, W_MUL = 1.0, 0.3, 1.0             # the paper's weights for MIRACL / MKQA
N_CAND = 200                                      # dense candidates reranked (multi-vec, all), as in the paper
N_HYBRID = 1000                                   # top-k from dense and from sparse (dense+sparse), as in the paper
MAX_LEN = 512                                     # 200 words fit; ArguAna queries are full arguments
CHUNK = 1000                                      # encode in chunks to keep RAM low
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


def encode(texts, colbert=True):
    """Encode texts with the three BGE-M3 heads (multi-vectors only if colbert=True).

    Returns:
        dense: np.ndarray (n_texts, 1024), one normalized [CLS] vector per text
        lexical:   list of n_texts dicts {token id (str): weight}, only non-zero tokens
        mul_vec:   list of n_texts np.ndarray (n_tokens_i, 1024), one vector per token, fp16 (empty if colbert=False)
    """
    dense, lexical, mul_vec = [], [], []
    for i in range(0, len(texts), CHUNK):
        out = model.encode(texts[i:i + CHUNK], batch_size=32, max_length=MAX_LEN,
                           return_dense=True, return_sparse=True, return_colbert_vecs=colbert)
        dense.append(np.asarray(out["dense_vecs"], dtype=np.float32))
        lexical += out["lexical_weights"]
        if colbert:
            mul_vec += [np.asarray(v, dtype=np.float16) for v in out["colbert_vecs"]]
    return np.vstack(dense), lexical, mul_vec


def to_sparse(weights, vocab):
    """Convert lexical weights to a sparse (text × token) matrix, so that S_lex = Q @ D.T.

    Returns:
        csr_matrix (n_texts, vocab): row = text, column = token id, cell = weight (0 if absent)
    """
    rows, cols, vals = [], [], []
    for i, w in enumerate(weights):
        for tok, x in w.items():
            rows.append(i)
            cols.append(int(tok))
            vals.append(float(x))
    return csr_matrix((vals, (rows, cols)), shape=(len(weights), vocab))


def colbert_batch(q, docs):
    """Late interaction for one query against all its candidates at once, on GPU:
    each query token takes its best document token, averaged over query tokens."""
    # Padding: docs have different token counts, so we pad them with zeros to the longest one (L)
    # to stack them in a single tensor D of shape (candidates, L, 1024).
    L = max(len(d) for d in docs)
    D = np.zeros((len(docs), L, q.shape[1]), dtype=np.float16)
    mask = np.zeros((len(docs), L), dtype=bool)
    for k, d in enumerate(docs):
        D[k, :len(d)] = d
        mask[k, :len(d)] = True
    q = torch.from_numpy(np.asarray(q, dtype=np.float16)).to(DEVICE)
    D, mask = torch.from_numpy(D).to(DEVICE), torch.from_numpy(mask).to(DEVICE)
    sim = torch.einsum("qh,kdh->kqd", q, D).float()
    sim = sim.masked_fill(~mask[:, None, :], float("-inf"))
    return sim.max(dim=2).values.mean(dim=1).cpu().numpy()


def top(scores, k):
    """Indices of the k highest scores."""
    return np.argsort(-scores)[:k]


def search(name, mode="all", queries=None):
    corpus, test_queries, _ = testbed.load(name)
    queries = queries or test_queries                           # e.g. the French SciFact queries
    ids, qids = list(corpus), list(queries)
    colbert = mode in ("multi-vec", "all")                      # multi-vectors only when needed (RAM)
    D_dense, D_lex, D_mul = encode(testbed.doc_texts(corpus), colbert)   # same text as the baselines
    Q_dense, Q_lex, Q_mul = encode([queries[q] for q in qids], colbert)  # no query prefix for M3

    vocab = len(model.tokenizer)
    S_dense = Q_dense @ D_dense.T                               
    S_lex = (to_sparse(Q_lex, vocab) @ to_sparse(D_lex, vocab).T).toarray()

    run = {}
    for i, q in enumerate(tqdm(qids, desc=f"{name} {mode}")):
        if mode == "dense":
            cand = top(S_dense[i], testbed.K + 1)
            s = S_dense[i, cand]
        elif mode == "sparse":
            cand = top(S_lex[i], testbed.K + 1)
            s = S_lex[i, cand]
        elif mode == "dense+sparse":
            cand = np.union1d(top(S_dense[i], N_HYBRID), top(S_lex[i], N_HYBRID))
            s = W_DENSE * S_dense[i, cand] + W_LEX * S_lex[i, cand]
        else:                                                   # multi-vec and all: rerank the dense top-200
            cand = top(S_dense[i], N_CAND)
            s_mul = colbert_batch(Q_mul[i], [D_mul[j] for j in cand])
            if mode == "multi-vec":
                s = s_mul
            else:
                s = W_DENSE * S_dense[i, cand] + W_LEX * S_lex[i, cand] + W_MUL * s_mul
        order = np.argsort(-s)[:testbed.K + 1]                  # top 101 (ArguAna drops one)
        run[q] = {ids[cand[k]]: float(s[k]) for k in order}
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=testbed.DATASETS, help="default: all three")
    parser.add_argument("--mode", choices=MODES, default="all")
    args = parser.parse_args()

    out_dir = HERE / ("runs" if args.mode == "all" else "runs_" + args.mode.replace("+", "_"))
    for name in args.datasets:
        testbed.save_run(search(name, args.mode), out_dir / f"{name}.json")
        print(name, args.mode, "done", flush=True)