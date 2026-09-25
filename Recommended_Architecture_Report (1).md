# Recommended Architecture — Business Entity Resolution Challenge
### Blocking, Retrieval and Matching Design, Grounded in Dataset EDA + LSBlock

**Author context:** Amazon ML Challenge 2026 — Business Entity Resolution
**Basis:** `Amazon_ML_Challenge_2026_Entity_Resolution_Report.md` (internal EDA), `problem_statement.md`, and Karapiperis, Tjortjis & Verykios, *"LSBlock: A Hybrid Blocking System Combining Lexical and Semantic Similarity Search for Record Linkage,"* ADBIS 2025.

---

## 1. Executive Summary

Recommended architecture, in one line: **country-partitioned, multi-channel hybrid blocking (deterministic high-confidence key + MinHash/LSH lexical retrieval + a small multilingual sentence-embedding model, fine-tuned with contrastive learning on our own ground truth, retrieved via HNSW ANN, thresholds learned from labeled data) → rich pairwise feature vector → gradient-boosted-tree pairwise classifier → per-tier calibrated thresholding for multi-match, macro-F0.5-optimal decisions.**

This is not a from-scratch redesign of your EDA's conclusions — it is the direct continuation of them, now cross-checked against four external sources: the LSBlock paper (hybrid lexical+semantic blocking), Azzalini et al.'s semantics-based blocking paper (embedding+ANN vs. embedding+clustering blocking), the Doan et al. Magellan paper (production-scale EM system experience), and Karapiperis, Tjortjis & Verykios's *Comprehensive Survey of Deep Learning for Entity Resolution* (a taxonomy of essentially every DL blocking/matching family in the field). Cross-referencing these against our own numbers changed one design decision (the embedding model should be fine-tuned, not generic — §7) and confirmed the rest, including several designs we deliberately did **not** adopt despite them appearing in the literature (§11.1). Your EDA already diagnosed the exact failure mode (deterministic blocking plateaus at ~85.8% recall at ~1.3B candidates) and already prescribed the fix (§20–21: approximate/character-n-gram retrieval). This report gives a concrete, dataset-fitted design for that retrieval layer, borrowing the two specific mechanisms LSBlock validates empirically — **MinHash/LSH for the lexical channel** and **HNSW-indexed dense embeddings for the semantic channel**, each governed by a **threshold learned from your own `train_ground_truth.tsv`**, not a hand-picked cutoff.

The final matcher stays a tree-based model (LightGBM/XGBoost/CatBoost). Nothing in the EDA, in LSBlock's own numbers, or in the wider survey argues for replacing that with a heavier model at this scale — see §11.1 for the specific alternatives considered and rejected (cross-encoders, graph-based collective matchers, LLM zero/few-shot matching), and §10 for why Magellan's own production numbers, at a scale directly comparable to ours, reinforce it rather than argue against it. LSBlock's benchmark results also show its own hybrid retrieval is a **blocking/candidate-generation** system (precision at the blocking stage ranges from 0.08 to 0.69 depending on domain), not a decision-making one — the two-thresholds trick in LSBlock just gives you a smarter blocking gate. The F0.5 score is still won or lost by the pairwise classifier and the calibration step that follows it, exactly as your EDA's §27 architecture already implies.

---

## 2. Constraints and Numbers That Drive Every Design Choice Below

| Constraint / fact | Source | Design consequence |
|---|---|---|
| ~17.3 trillion possible S1×(S2∪S3) test pairs | EDA §3 | Brute force impossible; blocking is mandatory, not optional |
| Deterministic block union reaches only 85.8%/85.79% recall (S2/S3) at ~1.34B candidates | EDA §17 | Structural blocking alone has plateaued — need a fundamentally different retrieval mechanism, not another structural key |
| Block A (country+housenumber+exact name): 97.8–97.9% precision, ~14.7–15.2% recall | EDA §16, §18 | Free, near-certain matches — should bypass the ML classifier entirely (or get an extremely permissive threshold) |
| Block B (country+housenumber+name-prefix): ~50–54% recall at 9–12% precision | EDA §16, §18 | Strong recall engine but needs the classifier to clean it up |
| Ground truth: 94.4% of S1 have ≥1 match, most common total = 3, up to 11 | EDA §8–9 | Rules out top-1/nearest-neighbor matching; must be multi-label with per-entity thresholding |
| Singletons (5.6% of S1) score 1.0 if empty, 0.0 if any false match | Problem statement, EDA §10 | The classifier + threshold must be conservative — false merges on a singleton are maximally costly under F0.5 |
| F0.5 metric, macro-averaged per S1 | Problem statement §Evaluation | Precision weighted 2× over recall — threshold calibration must directly optimize F0.5, not F1 or accuracy |
| France unseen in training, 15% of test S1 | EDA §7 | Country must be an open-set feature; any embedding/lexical retrieval component must generalize to a country it never saw labels for |
| Model must be MIT/Apache-2.0, ≤8B params | Problem statement §Constraints | Every learned component (embedding model, classifier) must be checked against this — ruled out: any closed-license or API-based model |
| No external lookups/APIs/geocoding | Problem statement §Academic Integrity | Every similarity signal must be computable from the three source files alone — pretrained *generic* text encoders are fine (they don't look up business identities), calling an ER API or geocoder is not |
| `candidate_pairs.tsv` must be the *exact* set fed to the final model | Problem statement §Output Format | The retrieval architecture below is not a "rough draft" blocking pass — its output is literally the file you submit |

---

## 3. Why the Existing Deterministic-Only Design Has to Change

Your EDA is unusually clear about *why* it stalls, and it's worth restating precisely because it dictates the fix:

- **Exact-match blocks are precise but brittle.** Exact normalized name+address hits 100% precision but only 2.7% (S2) / 0.005% (S3) recall (EDA §12) — almost no real-world noisy pair survives a fully exact match on both fields simultaneously.
- **Loosening the deterministic key to gain recall costs candidates combinatorially, not linearly.** Going from Block A → B → C → D roughly quadruples recall gain per step while candidate volume grows by one to two orders of magnitude each time (561K → 21.5M → 84.9M → 538M candidates, EDA §16). This is the classic signature of *token/prefix blocking hitting its structural ceiling* — you're trying to buy semantic tolerance with syntactic keys, and it doesn't scale.
- **The remaining ~14% of true pairs are exactly the cases structural keys can't reach by construction** (EDA §20): spelling changes, abbreviations, token reordering, transliteration, missing fields. None of these preserve a shared *prefix* or *exact token*, which is the only thing a deterministic key can test.

This is precisely the class of problem retrieval-based blocking (lexical *similarity*, not lexical *equality*, plus semantic similarity) is built for — and it's why the EDA's own §21 and §27 already point toward approximate retrieval as the next step. The rest of this report operationalizes that step using the mechanism LSBlock validates.

---

## 4. Recommended Architecture

```text
                              S1 REFERENCE (test: 1.73M rows)
                                         │
                                         ▼
                              NORMALIZATION (unicode/case/punct — already built)
                                         │
                                         ▼
                    ┌───────────────────┼────────────────────┐
                    ▼                   ▼                    ▼
          COUNTRY PARTITIONING   (US / India / France / any-unseen — open-set)
                    │
        ┌───────────┼─────────────────────────────┐
        ▼           ▼                             ▼
   TIER 0:       TIER 1:                      TIER 2:
   Deterministic Lexical retrieval             Semantic retrieval
   high-confidence  (MinHash/LSH over char      (multilingual sentence
   key (Block A)    q-grams, name & name+addr    embeddings + FAISS HNSW,
   country+house#   indexes, Jaccard threshold   name & name+addr indexes,
   +exact name      learned from ground truth)   cosine threshold learned
   → auto-tag                                    from ground truth)
   "high_conf"
        │                 │                             │
        └─────────────────┼─────────────────────────────┘
                           ▼
                  UNION + DEDUPLICATION
                  (tag each candidate: high_conf / lexical / semantic /
                   more than one channel — becomes a feature, not discarded)
                           │
                           ▼
                  CANDIDATE_PAIRS.TSV
              (this is the literal file submitted —
               target: tens of millions, not billions)
                           │
                           ▼
                  PAIRWISE FEATURE ENGINEERING
        (string similarity, structural agreement, retrieval-channel
         scores as features, country/source metadata, interactions)
                           │
                           ▼
                  GRADIENT-BOOSTED TREE CLASSIFIER
             (LightGBM/XGBoost/CatBoost — MIT/Apache-2.0, tiny vs. 8B cap)
                           │
                           ▼
              PER-TIER THRESHOLD CALIBRATION FOR MACRO F0.5
        (high_conf pairs: looser threshold / near auto-accept;
         retrieval-only pairs: stricter threshold, tuned on
         held-out S1-level validation split)
                           │
                           ▼
                MULTI-MATCH, SINGLETON-AWARE OUTPUT
                     matching_results.tsv
```

### Why this is a genuine redesign and not a relabeling of the old plan

The old plan (union of A, B, C, D, plus more structural blocks) tries to buy recall by *loosening exactness*. This plan buys recall by *adding a similarity dimension the structural approach cannot express at all* — approximate lexical distance (MinHash/Jaccard) and approximate semantic distance (embedding cosine). Both are retrieval mechanisms with a **fixed, bounded output size per S1 entity** (top-K), which is the property that stops recall gains from ever causing another 10–100× candidate explosion. That bounded-output property is the entire reason this design can plausibly close the 14% recall gap without repeating the 1.3-billion-candidate outcome.

---

## 5. Tier 0 — Keep the Deterministic High-Confidence Block, Change Its Role

Block A (country + house number + exact normalized name) stays, unmodified, because the numbers justify it on their own: 97.8–97.9% precision at ~15% recall is about as close to "free correct answers" as this dataset offers. Its role changes, though:

- Instead of being "one signal among many" fed generically into the classifier, **tag every Tier-0 hit explicitly** (`source_channel = high_conf`). This becomes a categorical feature the classifier can learn to trust almost unconditionally, and it lets you set a deliberately looser acceptance threshold specifically for this tier during calibration (see §9) — you are not obligated to apply one global threshold to structurally different evidence.
- Do **not** expand Tier 0 to Blocks B/C/D directly into the candidate set at full width. B/C/D's precision (7–12%, 2–3%, 0.4%) is too low to trust without the classifier, and their candidate cost (21M–538M) is exactly what the new retrieval tiers are meant to replace at a fraction of the cost. If you want to keep B as an extra recall net (it does add real recall other channels might miss), sample or cap it rather than taking it in full — treat it as a fallback signal, not a primary blocking channel.

---

## 6. Tier 1 — Lexical Retrieval via MinHash/LSH (the LSBlock lexical channel)

**Mechanism (adapted from LSBlock's reference implementation):**

1. Build character q-grams (q = 3 or 4; test both) of the normalized text for two separate keys per record: `name` and `name + address` (kept separate, mirroring your EDA's own finding in §12 that name and address blocks are complementary, not interchangeable).
2. Compute a MinHash signature (e.g., 64–128 permutations) over each q-gram set.
3. Use LSH banding on the MinHash signatures to bucket records without needing all-pairs comparison — this is what makes it sub-linear at 5M+ records per source, unlike naive TF-IDF cosine over the full corpus.
4. Within country partitions, for every S1 record, retrieve the union of S2/S3 records sharing at least one LSH band, then compute exact Jaccard similarity only within that shortlist.
5. Accept a candidate if its Jaccard similarity exceeds a **learned** threshold (see §8) rather than a hand-picked one.

**Why this specifically, not plain TF-IDF cosine (which the EDA's §21 originally proposed):** both catch typos, punctuation noise, and partial token overlap. MinHash/LSH's advantage is architectural: LSH banding gives you approximate near-neighbor retrieval with tunable, predictable cost, which matters directly at your scale (5M × 5M-ish comparisons per source pair). TF-IDF cosine top-K would need its own ANN index (e.g., via FAISS on sparse vectors) to scale the same way — so the practical choice is "LSH-banded MinHash" vs. "ANN-indexed TF-IDF," and the former has more precedent in the record-linkage literature at this scale (it's literally why LSBlock uses it) and one less moving part.

**Expected contribution:** this channel is aimed squarely at the noise categories your EDA's §20 lists that are lexical in nature — typos, abbreviation variants, punctuation, partial token overlap, missing components. It will not, by construction, help with transliteration-heavy or word-order-transposed cases where the surface character sequences diverge a lot even though the meaning doesn't — that's what Tier 2 is for.

---

## 7. Tier 2 — Semantic Retrieval via Multilingual Dense Embeddings + HNSW (the LSBlock semantic channel)

**Mechanism (adapted from LSBlock's reference implementation, `faiss_*_semantic.py`):**

1. Encode `name` and `name + address` (again, two separate indexes per your EDA's complementarity finding) using a pretrained sentence-embedding model.
2. **Model choice — deviating deliberately from LSBlock's own default:** their reference code uses `all-MiniLM-L6-v2`, an English-only model. Given your dataset spans US, India (with transliteration variants explicitly called out in the noise-pattern spec), and an unseen-in-training France, start from a **multilingual** sentence-transformer instead — e.g. `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, ~118M params) or `LaBSE` if cross-lingual quality matters more than latency. Both are comfortably inside the ≤8B-parameter, MIT/Apache-2.0 constraint, and neither performs any external lookup — they are generic pretrained text encoders, not entity-resolution services, so this stays inside the fair-play rules.
3. **Fine-tune this base model with supervised contrastive learning on your own `train_ground_truth.tsv` before building the index — do not deploy it purely off-the-shelf.** This is a deliberate departure from both LSBlock's reference implementation and from Azzalini et al.'s semantics-based blocking paper, both of which deploy their sentence embeddings unsupervised (LSBlock uses an off-the-shelf SBERT model; Azzalini et al. go further and *deliberately* train their sentence-composition RNN on an external corpus, specifically to keep the whole blocking phase label-free, on the stated assumption that in most data-integration scenarios no training data exists at all). That assumption doesn't hold here — you have 7.6M+ labeled S1-to-match links. The DL survey's discussion of SC-Block makes the relevant point explicit: the labeled data prepared for training the matcher can be reused, at no extra labeling cost, to train a better blocker, using a contrastive loss that pulls matched-pair embeddings together and pushes non-matches apart. Do this: sample positive pairs from ground truth, sample **hard negatives** from your Tier 0/1 candidate pools specifically (not random S1×S2/S3 pairs — your EDA already warned that random negatives make a classifier look better than it is by being too easy to distinguish; the same logic applies to fine-tuning an encoder), and fine-tune with a standard contrastive or triplet loss. This stays inside the MIT/Apache-2.0, ≤8B constraint since you are fine-tuning the same small base model's weights, not swapping to a larger one.
4. Build a FAISS `IndexHNSWFlat` per country partition per field (`efConstruction≈60`, `efSearch≈16` as a starting point, per LSBlock's own settings — tune from there), using the fine-tuned model's embeddings.
5. For every S1 record, retrieve top-K nearest neighbors (K in the 5–20 range; wider K trades recall for candidate volume) from the corresponding S2/S3 index.
6. Accept a candidate if its inner-product/cosine similarity exceeds a **learned** threshold (§8).

**A/B this against the off-the-shelf embedding before committing.** Fine-tuning is expected to help, per the general pattern documented across the survey's blocking taxonomy (supervised contrastive methods like SC-Block consistently produce smaller, more precise candidate sets than self-supervised or unsupervised ones at the same recall level) — but "expected to help in general" is not the same as "measured to help on this dataset." Run both versions through the same recall/candidate-volume measurement in §7 before locking in the fine-tuned version as final.

**Why this specifically:** dense embeddings capture things character n-grams and MinHash structurally cannot — word-order transpositions ("Sharma Electronics" vs. "Electronics Sharma"), DBA/trade-name substitutions, landmark-based address phrasing ("Near SBI ATM" vs. the literal street address), and semantically-equivalent-but-lexically-distant transliteration variants. These are exactly the residual noise categories in your EDA's §20 that are *not* purely lexical.

**Why HNSW specifically:** it's the standard choice for this scale (millions of vectors, need for real top-K rather than exact brute force), and it's what LSBlock itself validates. Their `Voters10` benchmark (10M records, recall ≈0.95, precision ≈0.60, using multiprocessed FAISS index sharding across CPU cores) is the closest existing precedent to your per-source scale (5M records in S2, 5.3M in S3) and is a reasonable sanity check for what to expect before you've measured your own numbers.

---

## 8. Learning the Acceptance Thresholds Instead of Guessing Them

This is the part of LSBlock most directly transferable and most valuable independent of everything else: **don't hand-pick a Jaccard or cosine cutoff.** Their procedure, adapted to your files:

1. Sample matched pairs from `train_ground_truth.tsv` and an equal number of non-matched pairs (random S1×S2/S3 pairs not in the ground truth — LSBlock uses this exact "random plus known-negative" sampling for threshold fitting, separate from the *hard-negative* sampling you'll use later for the classifier itself; the goals are different — the threshold model wants a simple, honest similarity/label relationship, not hard cases).
2. Compute the Jaccard (MinHash) or cosine (embedding) similarity for every sampled pair.
3. Fit a tiny 1-input, 2-hidden-layer MLP (16 units, as in LSBlock's reference code) as a scalar classifier on that single similarity score.
4. Read the optimal threshold off the validation precision-recall curve (maximize F1, or better — swap in F0.5 directly here too, since that's the actual competition metric, not F1).

This gives you two defensible, data-derived acceptance thresholds — one per retrieval channel — instead of an arbitrary "cosine > 0.8" guess, and it's cheap (a few thousand labeled pairs, a few seconds of training) relative to everything else in the pipeline. It also gives you something concrete and reproducible to put in the methodology write-up.

**Important distinction to preserve:** this threshold decides *inclusion in the candidate set* (recall-oriented, can afford to be permissive), not the *final match decision* (precision-oriented, must be conservative). Don't let a single similarity threshold do both jobs — that's the mistake the deterministic-blocks-only design implicitly made, and it's why the report's next section keeps a completely separate classifier + threshold stage.

---

## 9. Candidate Fusion and What Goes Into `candidate_pairs.tsv`

- Union all three tiers' outputs per S1 entity, deduplicated by (S1 id, S2/S3 id).
- Keep a per-candidate `source_channel` tag (`high_conf`, `lexical`, `semantic`, or a combination) and the raw similarity score(s) from whichever channel(s) surfaced it — these become classifier features, not just provenance metadata.
- This union, after dedup, **is** `candidate_pairs.tsv` — per the problem statement, this must be the exact set the classifier scores at inference, not an earlier, looser pass. Do not generate a broader set and silently prune afterward; whatever you export here has to match what you actually ran the model over.
- Target volume: your existing deterministic union alone was ~1.34B candidates for 85.8% recall. The retrieval tiers should let you *raise* recall while *lowering* volume substantially, because top-K retrieval per S1 is bounded (roughly `|S1| × K` per channel, i.e. tens of millions, not billions, for K in the 5–20 range) rather than growing with corpus size the way a loosened structural key does.

---

## 10. Pairwise Feature Engineering for the Classifier

Largely as your EDA's §22 already specifies, with two additions from the new retrieval tiers:

**Name features:** exact normalized match, Levenshtein similarity, Jaro-Winkler similarity, character n-gram cosine, MinHash Jaccard score (from Tier 1, even for candidates surfaced by a different channel — compute it as a feature regardless of provenance), token Jaccard, token overlap, prefix agreement, length difference.

**Address features:** exact normalized match, character similarity, token Jaccard, house-number agreement, postal/ZIP/PIN agreement, numeric token overlap, length difference.

**Retrieval-channel features (new):** embedding cosine similarity (Tier 2, computed as a feature regardless of which channel found the pair), `source_channel` one-hot/categorical, count of channels that independently surfaced this pair (agreement across channels is itself a strong signal — a pair found by *both* lexical and semantic retrieval is more trustworthy than one found by only one).

**Metadata:** country match (as a feature, never as a hard filter beyond initial partitioning — remember France must still work), source indicator (S2 vs. S3), address missingness, name/address lengths.

**Interactions:** name similarity × address similarity, house-number match × name similarity, exact-name + partial-address, `source_channel = high_conf` × everything else (lets the model learn that Tier-0 hits need less corroborating evidence).

---

## 11. Final Matching Model

**Primary recommendation: gradient-boosted trees (LightGBM, XGBoost, or CatBoost).** This is unchanged from the EDA's own §23, and LSBlock's results give independent reason not to deviate: LSBlock's own reported precision at the *blocking* stage swings from 0.08 to 0.69 across benchmark domains — the paper's contribution is a better candidate generator, not a better final classifier, and its own architecture still needs a downstream decision step for anything precision-sensitive. Trees remain the right choice here because:

- Your features are dense, hand-engineered, tabular similarity scores — exactly what GBMs are built for.
- You need to score tens of millions of candidate pairs; a GBM does this in minutes on CPU with no GPU dependency.
- Calibrated probability outputs make F0.5-threshold tuning direct and interpretable (unlike, say, a margin-based SVM).
- Fully compliant with the MIT/Apache-2.0, ≤8B constraint at essentially zero risk, since these are not "licensed pretrained models" in the sense the constraint is worried about — they're trained from scratch on your own labeled pairs.

**Optional second-stage refinement (only if time and validation numbers justify it):** a small, permissively-licensed cross-encoder or lightweight transformer (well under 1B params, e.g. a distilled model) that re-scores only the pairs in the GBM's *uncertain* probability band (say, 0.3–0.7), rather than the full candidate set. This mirrors your own EDA's §27 conclusion — restrict any heavier language model to a small hard-case subset rather than the full candidate universe — and mirrors the authors' own stated follow-up work (`hybrid_vectors.py` in the LSBlock repo, fusing a transformer embedding with a FastText sub-word embedding via a trainable gated network for typo-and-meaning robustness in one vector). That fusion idea is explicitly described by its authors as current, in-progress work rather than a validated, published result, so treat it as a stretch goal, not a baseline dependency.

**Optional, worth a genuine A/B rather than assuming an answer: the fine-tuned bi-encoder itself as a standalone matcher.** The DL survey notes a recent empirical finding that when pre-trained bi-encoders are systematically fine-tuned with hard negatives, they can rival or exceed cross-encoders as standalone end-to-end matchers on some benchmarks — meaning the same fine-tuned embedding model built for Tier 2 retrieval (§7) could, in principle, also make the final match/no-match decision via a learned cosine threshold, skipping the GBM entirely. Don't assume this wins on your data; measure it. Run both — GBM-on-features and threshold-on-fine-tuned-embedding — on the same held-out validation split and macro-F0.5 metric, and keep whichever wins, or keep both and treat disagreement between them as a signal of an uncertain pair (see the cascade idea below).

**What I would not do:** use an LLM or a single large embedding-similarity threshold as the *final* matcher without validating it against the GBM first. Both your EDA (§27) and LSBlock's own precision numbers argue against defaulting to it — retrieval-stage similarity is a recall tool, and dataset scale here (potentially tens of millions of final candidate pairs even after the improved blocking) makes anything heavier than a GBM impractical as the primary decision layer by default. See §11.1 for the full reasoning on what's excluded and why.

---

## 11.1 Alternatives Considered and Explicitly Rejected

The wider literature offers several plausible-looking alternatives to the design above. Each is addressed here so the reasoning is on record, rather than silently omitted.

**Clustering-based blocking (project embeddings via PCA/t-SNE, then cluster with Birch/DBSCAN/K-Means/hierarchical clustering) — rejected for retrieval, kept in mind only as a possible post-hoc analysis tool.** Azzalini et al.'s semantics-based blocking paper reports this approach (`Emb–Clust`) slightly outperforming LSH-based retrieval (`Emb–LSH`) on small benchmarks — six datasets, all under ~5,000 records per side — and running faster there. Two things make this inapplicable at our scale. First, t-SNE has no clean, cheap way to place a new query point into an existing low-dimensional projection without materially recomputing the projection — it is fundamentally a batch, non-inductive technique, whereas HNSW is built exactly for incremental nearest-neighbor queries against a fixed index. Second, Azzalini et al.'s own reported execution times make the scaling problem visible even at their toy scale: their `Emb–Clust`/`Emb–LSH` runs took from tens of seconds to over 1,000 seconds on datasets with only a few thousand records per side (their DBLP-ACM run, ~2,600×2,300 records, took over 1,000 seconds for `Emb-LSH`). At 5M+ records per source, this does not scale, full stop — HNSW-indexed ANN retrieval (§7) is the correct tool here, not a runner-up.

**Custom compositional neural encoders (RNN/LSTM/attention composing word-level embeddings into tuple vectors, à la DeepMatcher or Azzalini et al.'s own bi-LSTM+GloVe sentence composition) — rejected in favor of pretrained-then-fine-tuned sentence-transformers.** Both source papers effectively concede this point themselves: the DL survey is explicit that the field's third generation of embedding models (SBERT-style sentence transformers) superseded RNN-composed embeddings specifically because native sentence-level pretraining produces embeddings that are directly comparable via cosine similarity, without needing a custom compositional architecture bolted on top; Azzalini et al.'s paper predates SBERT and is explicitly building a compositional RNN because nothing better was available for their unsupervised setting at the time. Given that a pretrained multilingual sentence-transformer, fine-tuned on our own ground truth (§7), is both simpler to implement and empirically stronger per the survey's own account of how the field evolved, there is no reason to build a custom RNN/LSTM composition layer here.

**Cross-encoder transformers (DITTO, EMTransformer, JointBERT) as the primary matcher — rejected on computational grounds, not accuracy grounds.** These architectures concatenate both records into one sequence and run cross-attention between them, which the survey identifies as the primary computational bottleneck of the entire DL-ER field: the cost scales quadratically with sequence length and requires a full forward pass per candidate pair, with no way to pre-compute anything offline. Even after the improved blocking design in this report gets candidate volume down from ~1.3B to an estimated tens of millions, that volume is still far too large for a per-pair cross-encoder forward pass to be practical. This is a scale argument, not a quality argument — cross-encoders are not being rejected because they're worse at matching, but because they're the wrong tool for a corpus this size as the primary decision layer.

**Graph-based/collective matchers (GNEM, HierGAT, GraphER, FlexER) — rejected as unnecessary for this problem's structure.** These architectures earn their added memory and runtime cost (message-passing over a candidate graph, or multiple graph layers for FlexER's "multiple intents") when a single pair's correct label genuinely depends on evidence from *other*, related candidate pairs, or when there's more than one valid definition of what counts as a match. Neither condition is clearly present here: the problem statement gives a single match definition and treats Source 1 as the deduplicated reference set, with no signal that pairwise evidence needs to be shared across an entity's candidate set to be classified correctly. Adding this machinery without a demonstrated need would be complexity looking for a justification, not complexity earned by the problem.

**LLM zero-shot/few-shot matching as the pipeline's primary matcher — rejected on cost and latency grounds, consistent with your own EDA's conclusion.** The survey's own citation of Peeters et al.'s empirical cost analysis is unambiguous: current industrial practice treats LLMs as high-precision *rerankers* over a small candidate subset, not as a replacement for the full matching pipeline, specifically because per-pair inference latency and API cost don't scale to millions of comparisons. This directly reconfirms what your own EDA already concluded in §27 — restrict any heavier language model, if used at all, to a small hard-case subset.

---

## 10.1 Evidence From Magellan's Production Deployments (Scale Sanity Check)

Separately from the DL-specific comparisons above, the Magellan paper (Doan et al., CACM 2020) is useful here not for its blocking/matching *techniques* — Magellan's PyMatcher is explicitly a classical-ML system (decision trees, logistic regression, random forests over hand-engineered similarity features), the same family as the GBM recommended in §11 — but as **production-scale evidence that this class of model is not a compromise choice at your data size.** Their CloudMatcher deployment on a commercial farm/ranch policy-holder matching task ran at 109,974 × 4,922,505 records — a scale directly comparable to our S1 (millions) × S2/S3 (5M each) problem — and achieved 99.5% precision at 95% recall using exactly this class of learned matcher, no deep learning involved. This is corroborating evidence for the architecture choice in §11, not the primary justification for it (the primary justification is the scale/latency argument specific to cross-encoders), but it is a useful sanity check that a well-executed classical pairwise classifier is a proven approach at this exact order of magnitude, not merely a fallback for when deep learning isn't available.

One process habit from the same paper is worth adopting regardless of model choice: Magellan's PyMatcher guide has users **downsample before experimenting** — shrinking million-row tables to ~100K-row samples specifically because iterating at full scale is too slow to support the trial-and-error that EM inevitably requires. Apply the same discipline here: every blocking-channel comparison, threshold-learning run, and feature-importance check in this pipeline should run against a country-stratified sample (e.g. 100–200K S1 entities) first, with only the final inference run touching the full 1.7M-entity test set. This is a discipline for iterating quickly, not an architectural component, but it materially affects whether the rest of this design is actually buildable on a challenge timeline.

---

## 12. Multi-Match Decision Logic and Threshold Calibration

- **No top-1.** Score every candidate independently; keep everything the classifier scores above threshold. This is non-negotiable given 94.4% of S1 have ≥1 match and the modal count is 3 (EDA §8).
- **Per-tier thresholds, not one global cutoff.** Calibrate separately for:
  - `high_conf` (Tier 0) candidates — can use a looser threshold since the channel itself is ~98% precise before the classifier even runs.
  - Lexical-only candidates.
  - Semantic-only candidates.
  - Multi-channel-agreement candidates (found by 2+ tiers) — likely deserves its own, more permissive threshold, since cross-channel agreement is itself evidence.
  - Consider a separate cut for S2 vs. S3 and for records with missing addresses (3.3–3.4% of both, per EDA §5–6), since missing-address records structurally can't benefit from address-based features.
- **Optimize thresholds directly on macro F0.5**, computed exactly as the competition scores it (per-S1, then averaged, singletons included), on a held-out validation split — not on F1, not on global precision/recall, and not eyeballed.
- **Split by S1 entity, not by pair**, before doing any of this (EDA §24) — keep every candidate belonging to one S1 entity entirely in train or entirely in validation, to avoid leakage from a pair-level random split.

---

## 13. Handling the Open-Set Country Requirement (France)

- Country partitioning at the retrieval stage must handle an unseen value gracefully — implement it as "partition by whatever string value is present," not as a fixed enum with a fallback branch, so France's partition is built and searched exactly like US's and India's, just without any training-time supervision.
- The multilingual embedding model (§7) is the main lever for generalizing to France specifically — a model that has only ever seen English text will have degraded, unpredictable behavior on French business/address text. This is the strongest argument for spending the extra parameter budget on a multilingual encoder rather than defaulting to LSBlock's own English-only reference choice.
- Evaluate validation performance broken out by country (as your EDA's own §29 step 7 already plans) specifically to catch this failure mode before it shows up as a leaderboard surprise — France has no training signal at all, so a country-blind validation average could hide a real problem there.
- Do not let any classifier feature *require* a country value from a fixed set (e.g., a one-hot encoding trained only on {US, India} will silently break or zero out on France) — use it as a raw categorical/string match feature (`country_S1 == country_S2/S3`) rather than a closed one-hot.

---

## 14. Engineering and Scale Notes

- Per your own EDA §26, the current bottleneck is data engineering, not tensor computation — normalize once, convert every blocking/lexical key to integer IDs, keep relationships in compact arrays, and avoid Python-level `iterrows()`/tuple-set construction at the 5M-plus-record scale.
- GPU is well-spent on: batched embedding inference (Tier 2), FAISS index construction/search if using GPU FAISS, and batched MinHash computation — not on the general bookkeeping logic.
- At your scale, mirror LSBlock's own approach to the 10M-record `Voters10` benchmark: shard the corpus across CPU/GPU workers, build local per-shard indexes, dispatch queries in parallel, merge results centrally. This is directly reusable for both the MinHash/LSH and the HNSW stage.

---

## 15. Risk Register

| Risk | Mitigation |
|---|---|
| Retrieval-stage precision is very low on noisy data (LSBlock's own Amazon-Google/Abt-Buy numbers: 0.08–0.12) | Expected and acceptable — this is a blocking-stage number, not the final score; the GBM classifier exists specifically to fix this |
| Candidate volume still too large even after adding retrieval tiers | Cap top-K per channel; consider capping Tier-1/2 retrieval to S1 entities that didn't already get a Tier-0 hit, to avoid redundant work on already-resolved entities |
| Multilingual embedding model underperforms on India/France relative to English-only model | A/B test both on a held-out validation slice per country before committing; this is measurable, not something to assume either way |
| Threshold-learning MLP overfits on a small labeled sample | Use a reasonably large, class-balanced sample (LSBlock samples half the ground truth); re-validate the learned threshold on a separate held-out slice, not the same sample used to fit it |
| Singleton false-merges dominate the F0.5 penalty | Calibrate thresholds with singleton-inclusive macro F0.5 specifically, not an aggregate/micro metric that could mask this |
| `candidate_pairs.tsv` drifts from what the model actually scored | Generate it programmatically from the same candidate-generation function that feeds the classifier at inference time — never hand-maintain a "close enough" version |
| External-lookup rule violated inadvertently | Pretrained embedding models are fine (generic text encoders, not entity databases); anything that queries a live API, geocoder, or business registry is not — audit dependencies against this line explicitly in the methodology doc |

---

## 16. Summary of What Changed From the Original Deterministic-Only Plan

| | Original plan | This report's recommendation |
|---|---|---|
| Recall lever | Loosen structural key (A→B→C→D) | Add lexical (MinHash/LSH) and semantic (embedding+HNSW) retrieval channels |
| Recall ceiling observed | 85.8% at ~1.34B candidates | Expected higher recall at a fraction of the candidate volume (top-K bounded per S1, not corpus-size-dependent) |
| Threshold selection | Implicit (structural key match / no match) | Explicitly learned from `train_ground_truth.tsv` per channel, optimized toward F0.5 |
| Final matcher | GBM on structural + string features | Same GBM family, plus retrieval-channel scores and cross-channel-agreement as new features |
| Country handling | Constraint on structural keys | Partition boundary for retrieval indexes + multilingual embedding choice for generalization |
| Candidate cost model | Grows combinatorially with key looseness | Bounded by top-K per retrieval channel |

---

## References

1. Karapiperis, D., Tjortjis, C., Verykios, V. (2026). *LSBlock: A Hybrid Blocking System Combining Lexical and Semantic Similarity Search for Record Linkage.* ADBIS 2025, LNCS vol. 16043, pp. 131–146. https://doi.org/10.1007/978-3-032-05281-0_9
2. Reference implementation: https://github.com/dimkar121/LSBlock
3. Karapiperis, D., Tjortjis, C., Verykios, V. (2026). *A Comprehensive Survey of Deep Learning for Entity Resolution.* ACM Computing Surveys, Vol. 58, No. 14, Article 366. https://doi.org/10.1145/3828660
4. Doan, A., Konda, P., Suganthan G.C., P., Govind, Y., Paulsen, D., Chandrasekhar, K., Martinkus, P., Christie, M., et al. (2020). *Magellan: Toward Building Ecosystems of Entity Matching Solutions.* Communications of the ACM, Vol. 63, No. 8, pp. 83–91. https://doi.org/10.1145/3405476
5. Azzalini, F., Jin, S., Renzi, M., Tanca, L. (2021). *Blocking Techniques for Entity Linkage: A Semantics-Based Approach.* Data Science and Engineering, Vol. 6, pp. 20–38. https://doi.org/10.1007/s41019-020-00146-w
6. Internal EDA: `Amazon_ML_Challenge_2026_Entity_Resolution_Report.md`
7. Challenge specification: `problem_statement.md`
