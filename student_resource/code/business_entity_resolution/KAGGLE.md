# Running this pipeline on Kaggle

Kaggle notebooks give you a writable `/kaggle/working/`, a read-only
`/kaggle/input/`, free GPU (T4/P100), and a 9-12h session limit. This pipeline's
paths resolve relative to a *project root* that contains `dataset/`, `output/`
and `code/` as siblings (see `src/utils/io.py::resolve_path`), so the Kaggle
setup below just recreates that layout under `/kaggle/working/`.

## 1. Get the code onto Kaggle

Pick ONE:

**Option A -- GitHub (simplest if you've pushed this repo):**
```python
# Notebook settings: Internet = ON
!git clone https://github.com/<you>/<your-repo>.git /kaggle/working/repo
CODE_SRC = "/kaggle/working/repo/student_resource/code/business_entity_resolution"
```

**Option B -- upload as a Kaggle Dataset (no GitHub needed):**
1. Zip `student_resource/code/business_entity_resolution/` locally.
2. Kaggle -> "New Dataset" -> upload the zip -> attach it to this notebook as
   an input (e.g. it will appear at `/kaggle/input/ber-code/`).
```python
!unzip -q /kaggle/input/ber-code/business_entity_resolution.zip -d /kaggle/working/
CODE_SRC = "/kaggle/working/business_entity_resolution"
```

## 2. Get the dataset onto Kaggle

Upload `dataset/train/*.tsv` and `dataset/test/*.tsv` as a Kaggle Dataset too
(e.g. named `ber-dataset`), attach it to the notebook, then symlink it into
the project layout (Kaggle inputs are read-only, which is fine -- we only need
to *read* from `dataset/`; `output/` and `artifacts/` are separate writable
dirs).

## 3. Full setup cell

```python
import os, pathlib, shutil

PROJECT_ROOT = pathlib.Path("/kaggle/working/student_resource")
CODE_DST = PROJECT_ROOT / "code" / "business_entity_resolution"
DATASET_SRC = "/kaggle/input/ber-dataset/dataset"   # <-- adjust to your input slug

PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "output").mkdir(exist_ok=True)
(CODE_DST.parent).mkdir(parents=True, exist_ok=True)

if not CODE_DST.exists():
    shutil.copytree(CODE_SRC, CODE_DST)  # CODE_SRC from step 1

dataset_link = PROJECT_ROOT / "dataset"
if not dataset_link.exists():
    os.symlink(DATASET_SRC, dataset_link)

# Tell the pipeline exactly where the project root is (belt-and-suspenders;
# it also auto-detects this from the copied file location).
os.environ["BER_PROJECT_ROOT"] = str(PROJECT_ROOT)

print(list(PROJECT_ROOT.iterdir()))
```

## 4. Install the extra dependencies

Kaggle's base image already ships `pandas`, `numpy`, `scikit-learn`,
`lightgbm`, `torch`, `transformers`. You only need:

```python
!pip install -q datasketch rapidfuzz sentence-transformers faiss-cpu pyyaml joblib
```

(GPU notebooks: `sentence-transformers` will automatically use the GPU for
Tier 2 encoding -- no code change needed. `faiss-cpu` builds/searches the HNSW
index on CPU, which is fine at Kaggle's per-session data volumes; swap to
`faiss-gpu` only if you also shard the corpus, which is out of scope for a
single-notebook run.)

## 5. Run the phases

Everything is invoked as `python -m src.pipeline.<module>` with the working
directory set to `code/business_entity_resolution` -- do this with `%cd` (a
persistent magic) rather than `os.chdir` inside a `!` shell cell, since each
`!`-prefixed cell is its own subshell.

```python
%cd /kaggle/working/student_resource/code/business_entity_resolution
```

```python
# 1. Sanity-check Tier 0 against the EDA reference numbers. Use --sample while
#    iterating -- Kaggle sessions are time-boxed, and the full S1 file is 2M+ rows.
!python -m src.pipeline.run_train --phase tier0_only --sample 50000
```

```python
# 2. Tier 1 lexical recall ceiling (no threshold yet)
!python -m src.pipeline.run_train --phase tier1_measure --sample 50000
```

```python
# 3. Tier 2 semantic recall ceiling (uses the GPU automatically if the notebook
#    accelerator is set to GPU T4 x2 / P100)
!python -m src.pipeline.run_train --phase tier2_measure --sample 50000
```

```python
# 4. Learn thresholds + fuse candidates + measure fused recall/volume
!python -m src.pipeline.run_train --phase fuse --sample 50000
```

```python
# 5. Feature engineering + classifier training + F0.5 calibration
!python -m src.pipeline.run_train --phase train_full --sample 50000
```

Once you're happy with the numbers on a sample, drop `--sample ...` and re-run
step 5 (it re-runs 1-4 internally) at full scale -- budget real wall-clock time
for this; the 2M/5M/5.3M row full run will not fit comfortably in a single
interactive session on the free tier, so consider running it as a Kaggle
**batch/scheduled notebook** (Save & Run All) rather than interactively.

```python
# 6. Inference on the test set -> output/candidate_pairs.tsv, output/matching_results.tsv
!python -m src.pipeline.run_inference
```

```python
# 7. Validate before submitting (run from the project root, per the validator's own convention)
%cd /kaggle/working/student_resource
!python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

(`utils/validate_submission.py` isn't part of `code/business_entity_resolution`
-- it ships alongside `dataset/` and `output/` in the original
`student_resource/` package. Copy it into the symlinked/uploaded dataset, or
upload it as a third small Kaggle Dataset input, so step 7 has it on disk.)

## 6. Get the outputs back off Kaggle

```python
import shutil
shutil.make_archive("/kaggle/working/output", "zip", "/kaggle/working/student_resource/output")
```

Download `/kaggle/working/output.zip` from the notebook's Output/Files pane
(or, if this is a competition notebook, the two TSVs under `output/` are
already visible there directly).

## Notes specific to the Kaggle environment

- **Session limits:** GPU quota and 9-12h max session length mean Tier 2
  (embedding all of S1/S2/S3) and full classifier training may need to be
  split across a couple of "Save & Run All" batch jobs, persisting
  intermediate artifacts (`code/business_entity_resolution/artifacts/`) via a
  Kaggle Dataset "New Version" upload between runs, rather than one
  interactive session end to end.
- **`/kaggle/working` is the only writable path** -- this is exactly why
  `output_dir` and `artifacts_dir` in `config.yaml` are resolved under the
  project root you build in `/kaggle/working/student_resource/`, never under
  `/kaggle/input/`.
- **No internet at scoring time / on some competition kernels:** if the
  competition disables internet for submission notebooks, `pip install` and
  downloading the `sentence-transformers` model weights from Hugging Face at
  runtime will fail. Pre-download the model into a Kaggle Dataset (`model.save
  ("./paraphrase-multilingual-MiniLM-L12-v2")` in an internet-enabled notebook,
  then upload that folder) and point `tier2_semantic.model_name` in
  `config.yaml` at the local path instead of the Hugging Face model id.
