# BUILDING.md — Implementation Guide for a Coding Agent

This file tells an implementing agent **what to build, in what order, with what interfaces, and how to know each step actually worked**, to realize the architecture in `Recommended_Architecture_Report.md`. It assumes the agent has read that report; this document is the executable checklist version of it.

Work through the phases **in order**. Do not start a phase until the previous phase's "Definition of done" is met — each phase produces an artifact the next phase depends on, and the whole point of this design is that recall/precision at each stage is *measured*, not assumed.

---

## 0. Repository layout to create

Build everything under this structure from the start — it maps directly onto the required final submission package, so there's no later repackaging step.

```text
code/business_entity_resolution/
├── README.md                     # exact run instructions (fill in as you go, not at the end)
├── requirements.txt               # pin every version
├── config.yaml                    # every tunable in one place (see §1)
├── src/
│   ├── __init__.py
│   ├── normalize.py                # Phase 1
│   ├── blocking/
│   │   ├── __init__.py
│   │   ├── tier0_deterministic.py  # Phase 2
│   │   ├── tier1_lexical.py        # Phase 3 (MinHash/LSH)
│   │   ├── tier2_semantic.py       # Phase 4 (embeddings + HNSW)
│   │   └── fuse_candidates.py      # Phase 6
│   ├── thresholds/
│   │   ├── __init__.py
│   │   └── learn_threshold.py      # Phase 5
│   ├── features/
│   │   ├── __init__.py
│   │   └── pairwise_features.py    # Phase 7
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train_classifier.py     # Phase 8
│   │   └── calibrate_thresholds.py # Phase 9
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── run_train.py            # orchestrates Phases 1–9 on train data
│   │   └── run_inference.py        # orchestrates Phases 1–4,6,7,10 on test data
│   └── utils/
│       ├── __init__.py
│       ├── io.py                   # tsv readers/writers matching exact schema
│       └── validation_split.py     # S1-level split (Phase 8 dependency)
└── notebooks/                      # optional, exploration only — nothing load-bearing here
```

Output locations (already fixed by the problem statement — do not change):
```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

---

## 1. `config.yaml` — build this first, reference it everywhere

Every number below is a starting point from the architecture report, not a final value — the agent should treat each as a variable to be tuned once the corresponding phase is measurable, and record final chosen values here.

```yaml
paths:
  train_source1: dataset/train/train_source1.tsv
  train_source2: dataset/train/train_source2.tsv
  train_source3: dataset/train/train_source3.tsv
  train_ground_truth: dataset/train/train_ground_truth.tsv
  test_source1: dataset/test/test_source1.tsv
  test_source2: dataset/test/test_source2.tsv
  test_source3: dataset/test/test_source3.tsv
  output_dir: output/

normalization:
  unicode_form: NFKC
  lowercase: true
  strip_punct: true

tier0_deterministic:
  key_fields: [country, house_number, exact_name]

tier1_lexical:
  qgram_size: 4              # try 3 and 4, keep whichever gives better recall/candidate ratio
  minhash_permutations: 96   # 64–128 range
  lsh_bands: 24              # bands x rows-per-band = permutations
  fields: [name, name_address]   # build two separate MinHash indexes

tier2_semantic:
  model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  embedding_dim: 384
  fields: [name, name_address]
  hnsw:
    ef_construction: 60
    ef_search: 16
    m: 32
  top_k: 10                  # start at 10, sweep 5-20 during tuning

threshold_learning:
  sample_pairs_per_class: null   # default: len(ground_truth) // 2, per class
  mlp_hidden_units: 16
  optimize_for: f0.5             # NOT f1 — deviates from LSBlock reference, intentional

classifier:
  library: lightgbm              # lightgbm | xgboost | catboost — pick one, document choice
  license_check: "MIT/Apache-2.0, confirm before locking in"
  max_depth: -1
  num_leaves: 63
  learning_rate: 0.05

calibration:
  metric: macro_f0.5
  split_unit: source1_entity_id  # never split by pair — see Phase 8
  val_fraction: 0.15
  per_tier_thresholds: true      # separate threshold for high_conf / lexical / semantic / multi-channel
```

**Definition of done for Phase 0:** `config.yaml` exists, every path resolves, every module below reads its parameters from this file (no hardcoded magic numbers in code).

---

## 2. Phase 1 — Normalization (`src/normalize.py`)

Reuse the normalization already validated in the EDA report:

```python
import unicodedata, re

def normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text
```

Also implement, in the same file, the two structural extractors already used in the EDA (needed by Tier 0):

- `extract_house_number(address: str) -> str | None` — first numeric token.
- `extract_name_prefix(name: str) -> str` — first 4 chars of the normalized name with spaces removed.

**Apply normalization once**, at load time, to every record in S1/S2/S3 (train and test). Persist normalized fields alongside raw ones — do not re-normalize inside every downstream function; that's the exact `iterrows()`/repeated-string-op bottleneck the EDA report flags in its performance section.

**Definition of done:** running `normalize.py` against a sample of S1/S2/S3 rows produces stable, deterministic output; house-number and name-prefix extraction match the examples already validated in the EDA (spot-check 20 rows by hand).

---

## 3. Phase 2 — Tier 0 deterministic block (`tier0_deterministic.py`)

This is a straight port of the EDA's Block A — do not modify its definition, only its output format:

```python
def tier0_candidates(s1_df, s2_df, s3_df) -> pd.DataFrame:
    """
    Returns columns: source1_entity_id, candidate_entity_id, source_channel='high_conf'
    Key: (country, house_number, exact_normalized_name)
    """
```

Implementation approach: build a dict keyed on `(country, house_number, name)` → list of S2/S3 ids; for every S1 row, look up its key and emit all matches. This is a hash-join, not a nested loop — must run in minutes, not hours, even at 2M+ S1 rows.

**Definition of done:** on the training set, reproduces the EDA's own measured numbers for Block A (≈97.8–97.9% precision, ≈14.7–15.2% recall against `train_ground_truth.tsv`) within a small tolerance. If it doesn't reproduce those numbers, something in normalization or key construction has drifted — fix before proceeding, since every later phase is evaluated relative to this baseline.

---

## 4. Phase 3 — Tier 1 lexical retrieval (`tier1_lexical.py`)

1. For each of `name` and `name_address` (concatenation), compute character q-grams (config: `qgram_size`) per record.
2. Compute MinHash signatures (config: `minhash_permutations`) over the q-gram sets. Use an existing library (`datasketch` is the standard choice — MIT licensed, check before locking in) rather than hand-rolling MinHash.
3. Build an LSH index (banding, config: `lsh_bands`) **per country partition, per field** — this means one LSH index per (country, field) pair, not one global index. Insert all S2/S3 signatures for that country.
4. For every S1 record in that country partition, query the LSH index to get a shortlist, then compute *exact* Jaccard similarity only within that shortlist (LSH gives you the candidate shortlist cheaply; Jaccard on the shortlist gives you the real similarity score).
5. Do **not** apply an acceptance threshold yet inside this module — return every shortlisted pair with its raw Jaccard score. Thresholding happens in Phase 5/6, using a threshold *learned* from labeled data, not hardcoded here.

```python
def tier1_candidates(s1_df, s2_df, s3_df, country_value) -> pd.DataFrame:
    """
    Returns columns: source1_entity_id, candidate_entity_id, field,
                      jaccard_score, source_channel='lexical'
    Must be callable independently per country partition, including a
    country value never seen at training time (open-set requirement).
    """
```

**Definition of done:** run on a sample (e.g. one country partition of train) and measure recall@no-threshold (what fraction of true pairs appear *anywhere* in the shortlist, before any cutoff) against `train_ground_truth.tsv`. This number is your recall ceiling for this channel — record it. Also record wall-clock time per 100K S1 records; if it's not sub-linear-looking as country-partition size grows, the LSH banding parameters need revisiting before scaling to the full 5M-record sources.

---

## 5. Phase 4 — Tier 2 semantic retrieval (`tier2_semantic.py`)

1. Load the embedding model named in `config.yaml` (`sentence-transformers` library). **Before locking this in, verify its license and parameter count against the challenge's MIT/Apache-2.0, ≤8B constraint** — write the verification result into `README.md`.
2. Encode `name` and `name_address` for every record, per country partition, per source (S1, S2, S3). Batch this — do not encode one string at a time.
3. Build a FAISS `IndexHNSWFlat` (params from `config.yaml`: `ef_construction`, `ef_search`, `m`) per (country, field, source) combination.
4. For every S1 record, query top-`k` (config) nearest neighbors from the corresponding S2 and S3 indexes.
5. As in Phase 3, return raw similarity scores, unthresholded — thresholding is a Phase 5/6 concern.

```python
def tier2_candidates(s1_df, s2_df, s3_df, country_value) -> pd.DataFrame:
    """
    Returns columns: source1_entity_id, candidate_entity_id, field,
                      cosine_score, source_channel='semantic'
    Must run cleanly on an unseen country (France) — no country-specific
    branching, no assumption that country ∈ {US, India}.
    """
```

**Definition of done:** recall@top-k (before thresholding) measured against `train_ground_truth.tsv`, same protocol as Phase 3. Also run an A/B check: does the multilingual model actually outperform an English-only model (e.g. `all-MiniLM-L6-v2`) on the India partition specifically? Record the result — this determines whether the multilingual-model recommendation in the architecture report actually holds for this dataset, or whether it should be revisited.

---

## 6. Phase 5 — Threshold learning (`learn_threshold.py`)

Implements the LSBlock two-threshold trick, one instance per channel (lexical Jaccard, semantic cosine):

```python
def learn_threshold(train_pairs_with_scores, ground_truth, score_column) -> float:
    """
    1. Sample matched pairs from ground_truth (config: sample size).
    2. Sample an equal number of pairs NOT in ground_truth (random S1 x S2/S3,
       excluding known positives) as negatives.
    3. Compute `score_column` for every sampled pair (already available from
       Phase 3/4 output — do not recompute).
    4. Fit a tiny MLP (1 input -> hidden_units -> 1, sigmoid) as a scalar
       classifier on (score, label).
    5. Read the threshold off the precision-recall curve that maximizes F0.5
       (NOT F1 -- this is a deliberate change from the LSBlock reference,
       because F0.5 is the actual competition metric).
    6. Return the threshold, and log precision/recall AT that threshold on
       a held-out slice of the sampled pairs (never the same slice used to
       fit it).
    """
```

Run this once for the lexical channel's Jaccard score and once for the semantic channel's cosine score, per field if scores differ meaningfully by field (check before assuming one threshold serves both `name` and `name_address`).

**Definition of done:** two (or four, if per-field) thresholds are produced and logged with their precision/recall on a held-out sample. These are retrieval-stage (recall-oriented) thresholds — expect precision at this stage to be low (LSBlock's own benchmark precision ranged 0.08–0.69 depending on domain); that is normal and expected, not a bug to chase.

---

## 7. Phase 6 — Candidate fusion (`fuse_candidates.py`)

```python
def fuse(tier0_df, tier1_df, tier2_df, learned_thresholds) -> pd.DataFrame:
    """
    1. Filter tier1_df and tier2_df using the learned thresholds from Phase 5.
       tier0_df is NOT filtered — it's already ~98% precise by construction.
    2. Union all three on (source1_entity_id, candidate_entity_id).
    3. For any pair appearing in more than one tier, merge into a single row
       and set source_channel to reflect ALL contributing tiers (e.g.
       "lexical+semantic"), keeping every raw score as a separate column
       (jaccard_score, cosine_score) rather than collapsing them.
    4. Deduplicate strictly: no duplicate (source1_entity_id, candidate_entity_id)
       rows, no duplicate candidate_entity_id within a single S1's list.
    5. Write output/candidate_pairs.tsv in the EXACT required schema:
       source1_entity_id <tab> candidate_entity_ids (comma-joined, no quoting).
       Every S1 entity gets exactly one row, even if its candidate list is empty.
    """
```

**Critical constraint, restated because it's easy to violate accidentally:** `candidate_pairs.tsv` must be the literal set fed to the classifier at inference — generate it from this exact function's output, in the exact same pipeline run as inference, not from a separate "looks close enough" script. Run `utils/validate_submission.py` against a draft of this file as soon as it exists, before building anything downstream — catching a schema bug here is much cheaper than catching it after the classifier is trained on top of it.

**Definition of done:** total recall of the fused candidate set against `train_ground_truth.tsv` exceeds the old deterministic-only ceiling of 85.8%, and total candidate count is meaningfully below the old ~1.34B (target: tens of millions). If recall doesn't improve, revisit Phase 3/4 thresholds (too strict) or top-k (too narrow) before touching anything else.

---

## 8. Phase 7 — Pairwise feature engineering (`pairwise_features.py`)

For every row in the fused candidate set, compute:

```python
def compute_features(candidate_row, s1_record, s2s3_record) -> dict:
    """
    Name features: exact_match, levenshtein_sim, jaro_winkler_sim,
                    char_ngram_cosine, minhash_jaccard (recompute even if this
                    pair wasn't surfaced by tier1 -- always fill this feature),
                    token_jaccard, token_overlap, prefix_agreement, len_diff

    Address features: exact_match, char_sim, token_jaccard,
                       house_number_agreement, postal_agreement,
                       numeric_token_overlap, len_diff

    Retrieval features: embedding_cosine (recompute even if not surfaced by
                         tier2), source_channel (categorical), n_channels_agreeing

    Metadata: country_match (string equality, never a fixed one-hot --
              must not break on France), source_indicator (S2 vs S3),
              address_missing_s2s3, name_len, address_len

    Interactions: name_sim * address_sim, house_number_match * name_sim,
                  exact_name_and_partial_address,
                  is_high_conf * everything_else
    """
```

**Important:** several features (MinHash Jaccard, embedding cosine) must be computed for *every* candidate pair regardless of which tier originally surfaced it — a pair found only by Tier 0 still needs its lexical/semantic scores filled in as classifier features, not left null, or the classifier will overfit to `source_channel` as a proxy instead of learning from the actual similarity signal.

**Definition of done:** feature matrix has no unexpected nulls (nulls should only appear where a field is genuinely missing in the source data, e.g. address_missing), and a quick correlation check shows each feature group has *some* signal against the label (sanity check before full training — if `exact_match` on name has near-zero correlation with the label, something upstream is broken).

---

## 9. Phase 8 — Classifier training (`train_classifier.py`)

1. **Split by `source1_entity_id`** (`utils/validation_split.py`) — all candidate pairs for a given S1 stay entirely in train or entirely in validation. Never split by pair.
2. **Hard negatives, not random negatives:** positives = pairs present in `train_ground_truth.tsv`; negatives = candidate pairs from the fused set (Phase 6) that are *not* in the ground truth. Do not supplement with random unrelated S1/S2/S3 pairs — the EDA explicitly warns that random negatives make the classifier look better than it is by being too easy to distinguish.
3. Train the chosen GBM (config: `classifier.library`) on the Phase 7 feature matrix.
4. Log feature importance — this both sanity-checks the pipeline and is useful content for the methodology write-up.

**Definition of done:** validation AUC/PR-AUC is reasonable (no formal target — this is a diagnostic step, not the final metric) and feature importances make qualitative sense (name/address similarity features should dominate; `source_channel` alone should not be the top feature — if it is, the model is learning shortcuts instead of real similarity).

---

## 10. Phase 9 — Threshold calibration for F0.5 (`calibrate_thresholds.py`)

1. Score the validation split with the trained classifier.
2. Compute macro F0.5 **exactly as the competition does**: per-S1 F0.5 (precision/recall over that S1's predicted vs. true match set, with an empty-empty match scoring 1.0), then averaged across all S1 in the validation split, singletons included.
3. Sweep candidate probability thresholds and pick the one maximizing this exact metric — not F1, not global precision/recall.
4. If `calibration.per_tier_thresholds: true`, repeat this sweep separately for: `high_conf` pairs, lexical-only, semantic-only, multi-channel-agreement pairs, and optionally split further by S2-vs-S3 or address-missing status. Confirm empirically that per-tier thresholds actually beat a single global threshold before adding this complexity to the final pipeline — measure it, don't assume it.
5. Evaluate the calibrated pipeline's macro F0.5 **broken out by country** (US, India, and — critically — France) on the validation split, even though France has no training labels to calibrate against directly; this is the check that catches silent generalization failure before it becomes a leaderboard surprise.

**Definition of done:** a final threshold (or set of per-tier thresholds) is chosen, and its macro F0.5 on the held-out validation split is recorded as the number to beat/track. This value is the actual reference point for whether any subsequent tuning is helping.

---

## 11. Phase 10 — Inference pipeline (`run_inference.py`)

Runs Phases 1–4, 6, 7 on `dataset/test/*`, applies the trained classifier and calibrated thresholds, and writes:

- `output/candidate_pairs.tsv` — literally the Phase 6 output on test data.
- `output/matching_results.tsv` — for every `test_source1.tsv` entity, the comma-joined list of candidate ids scoring above the calibrated threshold for their respective tier, empty string if none.

**Before anything else, run:**
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
Fix every reported issue before considering the run complete. In particular, verify by construction (not just by validator pass) that:
- Every S1 test entity has exactly one row in both files.
- `matching_results.tsv`'s IDs are a subset of `candidate_pairs.tsv`'s IDs for every S1 (a mismatch here means Phase 6 and Phase 10 drifted apart — re-run from a single shared pipeline invocation, don't patch the output files by hand).

**Definition of done:** validator passes with exit code 0, and a manual spot-check of 20 S1 entities (mix of singleton-predicted and multi-match-predicted, across all three countries) looks qualitatively sane against the raw source records.

---

## 12. Order-of-execution summary (what to actually run, in sequence)

```bash
# 1. Sanity-check normalization and Tier 0 reproduce EDA numbers
python3 -m src.pipeline.run_train --phase tier0_only

# 2. Build and measure Tier 1 recall ceiling (no thresholding yet)
python3 -m src.pipeline.run_train --phase tier1_measure

# 3. Build and measure Tier 2 recall ceiling (no thresholding yet)
python3 -m src.pipeline.run_train --phase tier2_measure

# 4. Learn thresholds, fuse candidates, measure fused recall + volume
python3 -m src.pipeline.run_train --phase fuse

# 5. Feature engineering + classifier training + F0.5 calibration
python3 -m src.pipeline.run_train --phase train_full

# 6. Full inference on test set
python3 -m src.pipeline.run_inference

# 7. Validate before touching the leaderboard
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Each numbered step should be runnable and checkable independently — do not write one monolithic script that does everything with no intermediate checkpoints. The whole point of building it this way is that if the leaderboard score is disappointing, you can tell *which phase* is underperforming (recall ceiling too low? classifier miscalibrated? threshold wrong for one country?) instead of debugging the entire pipeline at once.

---

## 13. Things that will silently break the submission if skipped

- Forgetting to re-fill `minhash_jaccard`/`embedding_cosine` features for candidates surfaced only by Tier 0 (Phase 7 note) — leads to a classifier that overfits to provenance instead of similarity.
- Country handled as a fixed enum/one-hot anywhere in the pipeline instead of open-set string equality — will break or silently misbehave on France.
- Any step that reads `train_ground_truth.tsv` fitting on the *same* pairs it's later evaluated against (threshold learning, classifier training, calibration all need their own held-out slices, per Phases 5/8/9).
- `candidate_pairs.tsv` generated by a different code path than what the classifier actually scored — always regenerate both files from one pipeline run, never patch by hand.
- Any pretrained model or library dependency not checked against MIT/Apache-2.0 + ≤8B params before it's locked into the pipeline — check at the point of first use (Phase 4), not at submission time.
