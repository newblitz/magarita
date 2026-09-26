# OPTIMIZE.md — Fixing OOM Crashes and Making the Pipeline Fit in 13GB RAM

## 0. Context for the agent (read this before touching code)

This project builds an entity-resolution pipeline for the Amazon ML Challenge 2026 (see `Recommended_Architecture_Report.md` and `BUILDING.md` for the full design). In short: three source files (S1 ~1.7–2M rows, S2 ~5M rows, S3 ~5.3M rows), a multi-tier blocking pipeline (Tier 0 deterministic key match → Tier 1 MinHash/LSH lexical retrieval → Tier 2 fine-tuned sentence-embedding + HNSW retrieval), feature engineering, and a GBM classifier.

**The current symptom:** the process crashes and restarts during **Tier 0** (the deterministic `country + house_number + exact_name` block, described in `BUILDING.md` §3) on both Lightning AI and Kaggle, which both cap the environment at **13GB of RAM**. Because the whole pipeline has been running as one long process/notebook session, a crash in Tier 0 kills the kernel before Tier 1, Tier 2, or the classifier ever get to run.

**Important terminology correction, stated up front so the agent doesn't chase the wrong resource:** Tier 0 is a pure CPU hash-join over string/categorical fields — it does not use a GPU or any tensors, so **it cannot be a VRAM problem.** The 13GB limit that matters here is **system RAM**, not GPU VRAM. If profiling later reveals GPU-side OOM in Tier 2 (the embedding model), that is a separate, genuinely VRAM-bound problem and is addressed separately in §6 — do not conflate the two while fixing Tier 0.

**Two separate problems must be solved, not one:**
1. **Tier 0 itself is using too much RAM** and needs to be rewritten to use a fraction of its current footprint.
2. **The pipeline's process architecture lets one tier's crash block every other tier.** Even after Tier 0 is fixed, the pipeline must be restructured so a future crash in any one tier doesn't prevent the others from running — this is arguably the more important fix, because it makes the whole system resilient rather than just fixing today's specific crash.

Work through both. §1–4 fix Tier 0's memory footprint. §5 fixes the process-isolation problem so this class of failure can never again take down the whole pipeline. §6 hardens Tier 1/2/3 pre-emptively using the same lessons, since they will hit the same ceiling at the same scale if left as originally sketched in `BUILDING.md`.

---

## 1. Diagnose before rewriting — find the actual peak, don't guess

Before changing a single line of the join logic, instrument it. Guessing at "the" cause and rewriting blind risks fixing the wrong thing while the real leak persists.

1. Add memory logging at the start of `tier0_deterministic.py` and at every major step inside it (after each file load, after each derived-column computation, after the join, after writing output):
```python
import psutil, os

def log_mem(tag):
    rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[MEM] {tag}: {rss_gb:.2f} GB RSS", flush=True)
```
2. Run Tier 0 on a **small country-stratified sample first** (e.g. 200K S1 rows, proportionally sampled S2/S3 rows) — per the downsampling discipline already recommended in `Recommended_Architecture_Report.md` §10.1 — and confirm the memory-logging instrumentation itself works and produces a sane, small curve.
3. Only then run on progressively larger slices (1M, 3M, full) while watching the logged curve, to find the point where memory grows non-linearly or spikes suddenly — that spike, not intuition, tells you which step to rewrite first.
4. If `psutil` isn't available in the environment, use `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` (Linux reports this in KB) as a fallback with no extra dependency.

**Definition of done for this step:** a printed/logged memory curve exists for at least one full Tier 0 run (or a run that crashes, with the last successful log line showing where it died), before any rewrite begins.

---

## 2. Likely root causes in the current Tier 0 implementation (verify, don't assume)

Based on how `BUILDING.md` §3 specifies Tier 0 (a hash-join keyed on `(country, house_number, name)` across S1/S2/S3), the classic ways this blows past 13GB on a naive pandas implementation are, roughly in order of likely impact:

1. **Loading all three files with pandas' default `dtype` inference.** Pandas stores string columns as `object` dtype — an array of Python string pointers, each with per-object overhead on top of the string's own bytes. For ~5M rows × 3–4 text columns (`entity_id`, `business_name`, `business_address`, `country`), this alone can easily consume several GB per table before any processing happens.
2. **Keeping both raw and normalized columns simultaneously**, and/or keeping S1, S2, and S3 fully loaded in memory at the same time as separate DataFrames, rather than processing one country partition at a time and releasing memory between partitions.
3. **Using `pandas.merge()` directly on `(country, house_number, name)`** — if the join key isn't highly selective (e.g., many missing house numbers collapse to the same key, or a common name repeats across many records within a country), pandas' merge can produce an intermediate result far larger than either input table before any deduplication happens. This "join fan-out" is a classic silent memory blow-up and is a strong first suspect.
4. **Building the join as a Python dict of `tuple → list`** (e.g. `{(country, house_number, name): [id1, id2, ...]}`) instead of a vectorized/array-based join. Python dict/tuple/list overhead multiplies the effective memory cost of every key by several times versus an equivalent array-based representation.
5. **Notebook-specific accumulation (Lightning AI Studio / Kaggle notebooks specifically):** if Tier 0 runs across multiple notebook cells, every cell's output and every previously-assigned large variable stays alive in the kernel's namespace unless explicitly deleted — so by the time the join runs, the kernel may already be holding several dead-but-referenced copies of S1/S2/S3 from earlier exploratory cells.

Confirm which of these actually dominates using the logging from §1, then apply the matching fix from §3–4. In practice, more than one of these is usually true simultaneously at this scale, so plan to apply all the applicable fixes rather than stopping at the first one found.

---

## 3. Fix: rewrite Tier 0's data loading and join strategy

Apply these in order — each is a strict downgrade in peak memory versus the naive approach, with no loss of correctness.

### 3.1 Load with explicit, minimal dtypes and immediately downcast

```python
import pandas as pd

dtype_map = {
    "entity_id": "string",       # or convert to int32 index — see 3.3
    "business_name": "string",
    "business_address": "string",
    "country": "category",       # country has a small, bounded set of distinct values
}

def load_source(path):
    return pd.read_csv(path, sep="\t", dtype=dtype_map, usecols=list(dtype_map.keys()))
```
`country` as `category` is a large, safe win on its own — a handful of distinct values (US, India, France, and whatever else appears in test) stored as small integer codes instead of repeated full strings across millions of rows.

### 3.2 Process one country partition at a time, and free memory between partitions

Do not hold S1, S2, and S3 fully loaded simultaneously for the whole run. Instead:

```python
import gc

def run_tier0(s1_path, s2_path, s3_path, output_writer):
    countries = get_distinct_countries(s1_path, s2_path, s3_path)  # one cheap pass, or read from config
    for country in countries:
        s1_part = load_source_filtered(s1_path, country)
        s2_part = load_source_filtered(s2_path, country)
        s3_part = load_source_filtered(s3_path, country)

        candidates = join_partition(s1_part, s2_part, s3_part)
        output_writer.append(candidates)   # write incrementally — see 3.4

        del s1_part, s2_part, s3_part, candidates
        gc.collect()
```
This turns "hold everything in RAM at once" into "hold one country's slice in RAM at a time," which is exactly the kind of partitioning already recommended for Tier 1/2 in `BUILDING.md` — apply it to Tier 0 too, since nothing about the deterministic join requires cross-country data to be resident simultaneously (country is already part of the join key, so cross-country pairs can never match anyway).

### 3.3 Replace string IDs with compact integer codes for the join itself

String comparison and string-keyed hashing are both slower and far more memory-hungry than integer operations at this scale. Build a one-time mapping and join on integers:

```python
def encode_ids(df, id_col):
    codes, uniques = pd.factorize(df[id_col])
    return codes, uniques   # uniques[i] recovers the original string entity_id

# join on the integer-coded key, not the raw strings
# recover original entity_id strings only when writing final output rows
```
Apply the same integer-encoding idea to the join key itself: instead of joining on a `(country_code, house_number_string, name_string)` tuple, hash or factorize the composite key into a single integer column per table, and join on that single integer column. A single-integer-column join is both faster and dramatically lighter than a multi-column string-tuple join in pandas.

### 3.4 Never accumulate the full candidate output in memory — write incrementally

Whatever join strategy is used, do not build one giant in-memory DataFrame/list of every candidate pair across the whole run and write it once at the end. Open the output file once and append each partition's results as they're produced:

```python
class IncrementalTSVWriter:
    def __init__(self, path, columns):
        self.f = open(path, "w")
        self.f.write("\t".join(columns) + "\n")
    def append(self, df):
        df.to_csv(self.f, sep="\t", header=False, index=False, mode="a")
    def close(self):
        self.f.close()
```
This bounds peak memory to "one partition's candidates" rather than "all candidates across the whole test set," which matters a great deal once Tier 0 plus Tier 1/2 fusion is producing tens of millions of rows (per `Recommended_Architecture_Report.md` §9's target volume).

### 3.5 Prefer a sort-merge join over a hash join if the composite key still isn't selective enough

If, after §3.1–3.3, a specific country partition (e.g., a very common country or a generic default `house_number` value covering many rows with missing addresses) is still spiking memory because the join key repeats too often within that partition, switch that partition's join from a hash join to a **sort-merge join**: sort both sides by the composite key (an O(n log n), low-memory operation using numpy/pandas' native sort) and walk both sorted arrays with two pointers, emitting matches directly to the incremental writer instead of materializing a full cross-product of same-key groups in memory at once. This trades a small amount of extra CPU time (sorting) for a hard cap on peak memory during the match-emission step, which is the right trade at 13GB.

---

## 4. Fix: consider replacing pandas entirely for Tier 0, if §3 isn't enough

If the diagnostic logging in §1 shows that even the rewritten pandas approach in §3 is still uncomfortably close to the 13GB ceiling on the largest country partition, do not keep tuning pandas further — switch engines. Two well-tested options, in order of recommendation:

1. **DuckDB.** DuckDB can run a SQL join directly against the TSV files (or Parquet — see below) with automatic spill-to-disk when memory pressure is high, and a query planner that doesn't require holding full tables in Python-object form at all:
```python
import duckdb

duckdb.sql(f"""
    COPY (
        SELECT s1.entity_id AS source1_entity_id, s2.entity_id AS candidate_entity_id
        FROM read_csv('{s1_path}', delim='\t') s1
        JOIN read_csv('{s2_path}', delim='\t') s2
        ON s1.country = s2.country
           AND s1.house_number = s2.house_number
           AND s1.normalized_name = s2.normalized_name
    ) TO '{output_path}' (FORMAT CSV, DELIMITER '\t')
""")
```
   DuckDB's out-of-core execution model is built exactly for this kind of "join two tables bigger than RAM, on a constrained machine" task, and it requires no separate cluster or setup — it's a single pip-installable library. This is very likely the single highest-leverage change available if pandas keeps being tight even after §3's optimizations.
2. **Polars**, if the agent prefers to stay closer to a pandas-like API — Polars' Rust-backed columnar engine has a much lower per-row memory overhead than pandas for string-heavy data, and supports lazy, streaming execution (`pl.scan_csv(...).filter(...).join(...).sink_csv(...)`) that never materializes the full dataset in memory at once.

Either way, **convert the source TSVs to Parquet once, up front**, regardless of which engine is used for the join itself:
```python
import pandas as pd
pd.read_csv(s2_path, sep="\t", dtype=dtype_map).to_parquet("s2.parquet")
```
Parquet is columnar and compressed, so every subsequent read (Tier 0 retry, Tier 1, Tier 2, feature engineering) is both faster and lighter on memory than re-parsing the raw TSV every time — this benefits every later phase in `BUILDING.md`, not just Tier 0.

---

## 5. Fix the process-isolation problem: one tier crashing must not block the others

This is the more structurally important fix. Right now, per the reported symptom, Tier 0 crashing (and the environment auto-restarting the kernel) prevents Tier 1, 2, and 3 from ever running in the same session. Fix the pipeline's execution model, not just Tier 0's memory use:

1. **Run each tier as a separate OS process launched from the command line, not as sequential cells/calls inside one long-lived notebook kernel or one long-lived Python process.** Concretely: `python -m src.blocking.tier0_deterministic --config config.yaml`, then (separately, as its own process invocation) `python -m src.blocking.tier1_lexical --config config.yaml`, and so on — exactly the CLI structure already sketched in `BUILDING.md` §12, but the requirement here is stricter: **these must be genuinely separate process launches** (separate `python ...` invocations, e.g. from a shell script or separate notebook cells that each call `subprocess.run([...])` or use `!python script.py` rather than importing and calling functions in-process), not function calls within one importable pipeline object living in one process's memory space for the whole run.
2. **Each tier must read its input from disk (the previous tier's checkpointed output) and write its own output to disk, with no in-memory hand-off between tiers.** This is what actually decouples them: if Tier 0's process crashes, Tier 1 doesn't need Tier 0's Python objects — it needs Tier 0's *output file*, which either exists (partial progress preserved, see §5.3) or doesn't (rerun just Tier 0).
3. **Make Tier 0 itself resumable, since a mid-run crash currently loses all progress.** Because §3.2 already partitions Tier 0 by country, checkpoint at the partition boundary: write each country's candidates to its own output file (or append with a per-country "done" marker/manifest file), and on restart, skip any country already marked complete:
```python
import json, os

def already_done(country, manifest_path="tier0_manifest.json"):
    if not os.path.exists(manifest_path):
        return False
    with open(manifest_path) as f:
        return country in json.load(f).get("completed", [])

def mark_done(country, manifest_path="tier0_manifest.json"):
    manifest = {"completed": []}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest["completed"].append(country)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
```
   This means a crash on, say, the 4th of 5 country partitions loses only that partition's in-progress work, not the whole tier's.
4. **In notebook environments specifically (Lightning AI, Kaggle), don't run the pipeline as notebook cells at all for anything beyond quick exploration.** Write the pipeline as proper `.py` scripts/modules under `src/`, and invoke them from a notebook cell only via `!python -m src.pipeline.run_train --phase tier0` (a genuine subprocess), or better, from a terminal. This sidesteps the notebook-kernel memory-accumulation problem in §2 point 5 entirely, since each `!python ...` invocation gets a clean process with its own memory space that's fully released when the script exits — regardless of what the notebook kernel itself is still holding onto.

**Definition of done for this section:** killing the Tier 0 process mid-run (e.g. `kill -9` on it deliberately, as a test) does not prevent Tier 1 from being launched and running successfully afterward against whatever Tier 0 output already exists on disk; and rerunning Tier 0 after a kill resumes from the last completed country partition rather than starting over.

---

## 6. Pre-emptively harden Tier 1, Tier 2, and Tier 3 the same way

Tier 0 hit the ceiling first only because it runs first. Tier 1 (MinHash/LSH over 5M+ records) and Tier 2 (embedding + HNSW index construction over the same scale) will hit the same 13GB ceiling if left as originally sketched, for the same underlying reasons — do not wait for them to crash before applying these:

- **Tier 1 (MinHash/LSH):** build LSH indexes one country partition at a time (already specified in `BUILDING.md` §4, but re-confirm this is actually implemented that way, not just planned that way); persist each partition's MinHash signatures to disk (Parquet) rather than keeping all partitions' signatures resident simultaneously; use the `datasketch` library's built-in serialization rather than hand-rolled Python objects for signatures.
- **Tier 2 (embeddings + HNSW):** encode records in **batches** (e.g. 10K–50K at a time, not the whole corpus in one `model.encode(all_texts)` call) and write embeddings to disk incrementally (as `.npy` memory-mapped arrays or Parquet with a vector column) rather than holding a `(12M+ records, 384-dim)` float array in RAM simultaneously — that alone is tens of GB before HNSW even gets involved. Build the HNSW index incrementally (`index.add_items(batch_embeddings)` per batch) rather than requiring the full embedding matrix to be resident before index construction starts. **This step is the one place GPU VRAM genuinely matters** (the embedding model's forward pass): keep the encoding batch size small enough to fit comfortably in whatever GPU VRAM the environment provides (this is a separate, genuinely GPU-side budget from the 13GB system-RAM figure this whole document otherwise addresses), and move embeddings to CPU/disk immediately after each batch rather than accumulating them on GPU.
- **Tier 3 / feature engineering / classifier training:** compute features and train the GBM in chunks over the candidate set rather than loading the full fused candidate set (tens of millions of rows × ~20–30 feature columns) into one DataFrame at once — LightGBM in particular supports incremental/chunked training via its `Dataset` API with reference datasets built from disk-backed data rather than requiring one in-memory pandas DataFrame for the whole training set.

Apply the same three general principles used to fix Tier 0 to every one of these: **(a)** partition and process incrementally rather than all-at-once, **(b)** write/checkpoint to disk between partitions rather than accumulating in memory, **(c)** run each as an isolated process invocation so a crash in one doesn't cascade into the others.

---

## 7. Verification checklist before declaring this fixed

- [ ] Memory-logging instrumentation (§1) shows peak RSS staying comfortably under 13GB (leave real headroom — target well under the ceiling, not right up against it, since the notebook environment itself and any background processes also consume some of that budget) for a full Tier 0 run on the complete S1/S2/S3 test data, not just a sample.
- [ ] Tier 0 produces identical (or near-identical, modulo any intentional join-strategy change) candidate output before and after the rewrite, checked against the small-sample run from §1 as a regression check — a memory fix that silently changes which candidates get produced is not an acceptable fix.
- [ ] Killing the Tier 0 process mid-run and restarting it resumes from the last completed country partition rather than restarting from scratch (§5.3).
- [ ] Tier 1 can be launched and can complete successfully in a session where Tier 0's process already exited (crashed or completed), reading only from Tier 0's checkpointed disk output (§5.2).
- [ ] The same partition-incrementally / checkpoint-to-disk / isolate-as-a-process pattern has been applied to Tier 1, Tier 2, and the feature-engineering/classifier step (§6), not left as an open risk for later.
- [ ] All raw TSV inputs have been converted to Parquet once (§4), and every phase from Tier 0 onward reads Parquet rather than re-parsing TSV on every run.
