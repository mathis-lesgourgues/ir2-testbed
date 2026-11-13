"""Translate the 300 SciFact test queries to French with NLLB-200 (600M), and pick 30 to check by hand.

To access the translation report : python pairs/11-bge-m3/translate_scifact.py --report   → share of correct translations in the filled check_30.csv
Ran to get the 30 most difficult queries: python pairs/11-bge-m3/translate_scifact.py            → queries_fr.json (300 queries) + check_30.csv (30 to check)
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))         
import testbed

MODEL = "facebook/nllb-200-distilled-600M"
OUT_JSON = HERE / "queries_fr.json"
OUT_CSV = HERE / "check_30.csv"
N_CHECK = 30
# Technical terms first: acronyms, numbers, hyphenated words, Greek letters
TECHNICAL = re.compile(r"\b[A-Z]{2,}\w*|\d|\w+-\w+|[α-ωΑ-Ω]")


def translate(texts, batch_size=16):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(testbed.device()).eval()
    fra = tok.convert_tokens_to_ids("fra_Latn")
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**batch, forced_bos_token_id=fra, num_beams=4, max_length=256)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        print(f"{len(out)}/{len(texts)}", flush=True)
    return out


def main():
    _, queries, _ = testbed.load("scifact")
    qids = list(queries)
    fr = translate([queries[q] for q in qids])
    queries_fr = dict(zip(qids, fr))                              # same ids as the English queries
    json.dump(queries_fr, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)

    # The 30 queries with the most technical terms
    ranked = sorted(qids, key=lambda q: -len(TECHNICAL.findall(queries[q])))[:N_CHECK]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "english", "french", "correct", "comment"])
        for q in ranked:
            w.writerow([q, queries[q], queries_fr[q], "", ""])
    print(f"wrote {OUT_JSON} and {OUT_CSV}: fill the 'correct' column with 1 or 0")


def report():
    rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8")))
    marks = [int(r["correct"]) for r in rows if r["correct"].strip() in ("0", "1")]
    print(f"{sum(marks)}/{len(marks)} correct ({sum(marks) / len(marks):.0%}), {len(rows) - len(marks)} not checked yet")
    for r in rows:
        if r["correct"].strip() == "0":
            print(f"- {r['english']}\n  → {r['french']}\n  {r['comment']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    report() if parser.parse_args().report else main()