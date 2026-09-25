# Business Entity Resolution -- Pipeline

Implements the architecture in `Recommended_Architecture_Report.md`, following the
phase-by-phase checklist in `BUILDING.md`: country-partitioned, multi-channel hybrid
blocking (deterministic high-confidence key + MinHash/LSH lexical retrieval +
multilingual sentence-embedding/HNSW semantic retrieval, thresholds learned from
labeled data) -> rich pairwise feature vector -> gradient-boosted-tree classifier ->
per-tier calibrated thresholding for macro-F0.5-optimal, multi-match, singleton-aware
output.

## Layout

```text
business_entity_resolution/
├── README.md                     # this file
├── requirements.txt               # pinned dependencies
├── config.yaml                    # every tunable in one place
├── src/
│   ├── normalize.py                # Phase 1 -- normalization + structural extractors
│   ├── blocking/
│   │   ├── tier0_deterministic.py  # Phase 2 -- (country, house_number, exact_name) block
│   │   ├── tier1_lexical.py        # Phase 3 -- MinHash/LSH lexical retrieval
│   │   ├── tier2_semantic.py       # Phase 4 -- multilingual embeddings + FAISS HNSW
│   │   └── fuse_candidates.py      # Phase 6 -- candidate fusion -> candidate_pairs.tsv
│   ├── thresholds/
│   │   └── learn_threshold.py      # Phase 5 -- learned Jaccard/cosine acceptance thresholds
│   ├── features/
│   │   └── pairwise_features.py    # Phase 7 -- pairwise feature engineering
│   ├── model/
│   │   ├── train_classifier.py     # Phase 8 -- LightGBM training on hard negatives
│   │   └── calibrate_thresholds.py # Phase 9 -- macro-F0.5 threshold calibration
│   ├── pipeline/
│   │   ├── run_train.py            # orchestrates Phases 1-9 on train data
│   │   └── run_inference.py        # orchestrates Phases 1-4,6,7,10 on test data
│   └── utils/
│       ├── io.py                   # config loading + exact-schema TSV IO
│       ├── validation_split.py     # S1-level train/val split
│       └── metrics.py              # macro F0.5, exactly as the competition scores it
└── notebooks/                      # exploration only, nothing load-bearing
```

Paths in `config.yaml` (e.g. `dataset/train/train_source1.tsv`, `output/`) are
resolved relative to the **project root** -- the directory containing `dataset/`,
`output/` and `code/` (this is `student_resource/` in this submission). Resolution
happens once, in `src/utils/io.py::resolve_path`, so every module can be invoked
from any working directory. Override the root with the `BER_PROJECT_ROOT`
environment variable if you relocate the dataset.

## Setup

```bash
cd student_resource/code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate   # or use uv/conda
pip install -r requirements.txt
```

Tier 2 (semantic retrieval) depends on `sentence-transformers` + `faiss-cpu` (and
transitively `torch`), which are large downloads. Every other phase (Tier 0, Tier 1,
threshold learning, feature engineering, classifier training, calibration) runs with
only the core dependencies -- Tier 2 modules import these lazily and are skipped with
a warning if they are not installed, so you can iterate on the rest of the pipeline
without them.

## Run, end to end

Each phase is independently runnable and checkable -- do not skip straight to
`train_full` on your first run; confirm each phase's "definition of done" in
`BUILDING.md` first.

```bash
# 1. Sanity-check normalization and Tier 0 reproduce EDA numbers (~97.8-97.9%
#    precision, ~14.7-15.2% recall against train_ground_truth.tsv)
python3 -m src.pipeline.run_train --phase tier0_only

# 2. Build and measure Tier 1 recall ceiling (no thresholding yet)
python3 -m src.pipeline.run_train --phase tier1_measure

# 3. Build and measure Tier 2 recall ceiling (no thresholding yet; requires the
#    optional Tier-2 dependencies)
python3 -m src.pipeline.run_train --phase tier2_measure

# 4. Learn thresholds, fuse candidates, measure fused recall + volume
python3 -m src.pipeline.run_train --phase fuse

# 5. Feature engineering + classifier training + F0.5 calibration
python3 -m src.pipeline.run_train --phase train_full

# 6. Full inference on the test set -> output/candidate_pairs.tsv,
#    output/matching_results.tsv
python3 -m src.pipeline.run_inference

# 7. Validate before touching the leaderboard (run from student_resource/)
cd ../..
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Add `--sample 50000` to any `run_train.py` invocation to restrict the number of S1
rows for fast local iteration (the pipeline's own "downsample before experimenting"
discipline -- see architecture report §10.1); full-scale runs should omit it.

Artifacts (learned retrieval thresholds, the trained classifier, and the calibrated
decision thresholds) are written to `code/business_entity_resolution/artifacts/` and
are what `run_inference.py` loads -- `run_train.py --phase train_full` must be run at
least once before `run_inference.py`.

## Design notes / where to find what

- **Country is always an open string-equality field**, never a fixed enum or
  one-hot -- every module (Tier 1/2 partitioning, the `country_match` feature)
  works unmodified on France, which has zero training labels. See
  `src/normalize.py::normalize_dataframe` and `Recommended_Architecture_Report.md`
  §13.
- **`candidate_pairs.tsv` is generated by the same `fuse()` call that feeds the
  classifier**, both in `run_train.py` (Phase 6, for measurement) and in
  `run_inference.py` (Phase 10, for submission) -- never hand-patched.
- **Retrieval-channel features (MinHash Jaccard, embedding cosine) are recomputed
  for every candidate pair**, regardless of which tier originally surfaced it
  (`src/features/pairwise_features.py::compute_features`), so the classifier
  learns from similarity, not from `source_channel` provenance alone.
- **Negatives for classifier training are hard negatives** -- candidate pairs from
  the fused set that are not in the ground truth -- never random unrelated
  S1/S2/S3 pairs (`src/model/train_classifier.py`).
- **All splitting is S1-entity-level**, never pair-level (`src/utils/validation_split.py`).
- **Macro F0.5 is computed exactly as the competition scores it** -- per S1,
  singletons included, then averaged (`src/utils/metrics.py`), and is the metric
  both threshold-learning (Phase 5) and calibration (Phase 9) directly optimize.

## License and size compliance (problem statement §Constraints.5)

- **Classifier: LightGBM.** MIT licensed. Trained from scratch on our own labeled
  pairwise features -- not a redistributed pretrained model, so it is not subject
  to a parameter-count concern in the sense the constraint targets, but LightGBM's
  own footprint here (a few hundred trees, depth-limited) is trivially under any
  reasonable size reading of the 8B-parameter cap.
- **Tier 2 embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.**
  Apache-2.0 licensed, ~118M parameters -- comfortably inside the MIT/Apache-2.0,
  <=8B constraint. It is a generic pretrained text encoder (no entity database, no
  external API call), so using it for embedding-based retrieval does not violate the
  "no external lookups" fair-play rule. Re-verify the license/param count on the
  Hugging Face model card before changing `tier2_semantic.model_name` in
  `config.yaml`.
- **Threshold-learning MLP (`src/thresholds/learn_threshold.py`):** a tiny
  `sklearn.neural_network.MLPClassifier` (1 input -> 16 hidden units -> 1 output),
  scikit-learn (BSD licensed). This model is a calibration diagnostic, not the final
  matcher, but is included here for completeness.
- No component in this pipeline queries an external API, geocoder, or business
  registry; every similarity signal is computed from `business_name`,
  `business_address` and `country` in the three provided source files.
