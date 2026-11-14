# Error analysis: French queries on the English SciFact corpus

From `results_crosslingual.json` (`worst`: the 5 queries that lose the most nDCG@10 per mode).

**Sparse loses most (−0.195 nDCG@10, Recall@100 0.894 → 0.751).** Its score only counts tokens that appear in both the query and the document, so a correct French query shares almost nothing with an English abstract. Its five worst queries all fall from 1.0 to 0. Four are translated correctly, for example *Taxation of sugar-sweetened beverages…* → *La taxation des boissons sucrées…*. What survives is names and acronyms (Fz/PCP, TNF-α), which are the same in both languages.

**Dense and multi-vec lose about 0.05.** Their worst cases are mostly translation, not the model:
- 1204, *quiescent* → *silencieux*, attached to the follicle (one of the 4 errors in `check_30.csv`)
- 533 and 532, *bypass* → *contournement* instead of *pontage*
- 960, *Polymeal* → *polymère*, which changes the meaning (sparse loses it too)
- 513, *fitness* → *aptitude*, a literal translation

Some losses have no translation error: 837 (*NR5A2… endometrial tissues*) and 644 (*Insulin increases risk of severe kidney failure*) are both translated correctly and still drop to 0.

**All keeps its lead in French (0.634, equal with multi-vec), but the fusion gain over multi-vec disappears.** Sparse adds noise once it stops matching words, while Recall@100 doesn't move (0.914 → 0.915).

**The 26 hand-checked queries lose much less (All −0.008, p = 0.68).** This doesn't show that the full loss comes from translation errors: these 26 were picked for having the most acronyms and numbers, which match in both languages. They are not a random sample.
