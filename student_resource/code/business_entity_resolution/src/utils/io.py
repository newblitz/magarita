"""TSV readers/writers matching the exact competition schema, plus config loading.

Every path in ``config.yaml`` is relative to the *project root* -- the directory
that contains ``dataset/``, ``output/`` and ``code/`` (this is ``student_resource/``
in this repository). Resolving paths in one place, here, means every module reads
its parameters from ``config.yaml`` with no hardcoded magic paths elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import yaml

# src/utils/io.py -> src/utils -> src -> business_entity_resolution -> code -> project root
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[4]
CODE_ROOT = _THIS_FILE.parents[2]  # code/business_entity_resolution/
DEFAULT_CONFIG_PATH = CODE_ROOT / "config.yaml"

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
RESULT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def resolve_path(path: str | os.PathLike) -> Path:
    """Resolve a (possibly relative) config path against the project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = Path(os.environ.get("BER_PROJECT_ROOT", PROJECT_ROOT))
    return (root / p).resolve()


def load_config(config_path: str | os.PathLike | None = None) -> dict:
    """Load ``config.yaml``. Every tunable in the pipeline should come from here."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def read_source_tsv(path: str | os.PathLike, nrows: int | None = None) -> pd.DataFrame:
    """Read one of the *_source{1,2,3}.tsv files with the exact expected schema."""
    df = pd.read_csv(
        resolve_path(path),
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected column(s) {missing}")
    return df


def read_ground_truth_tsv(path: str | os.PathLike, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        resolve_path(path),
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )
    missing = set(GROUND_TRUTH_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected column(s) {missing}")
    return df


def ground_truth_to_pairs(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Explode ``train_ground_truth.tsv`` into one row per (s1_id, matched_id)."""
    records = []
    for s1, ids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        ids = ids.strip()
        if not ids:
            continue
        for mid in ids.split(","):
            mid = mid.strip()
            if mid:
                records.append((s1, mid))
    return pd.DataFrame(records, columns=["source1_entity_id", "candidate_entity_id"])


def _id_list_dict_to_frame(
    id_lists: Mapping[str, Iterable[str]], all_s1_ids: Iterable[str], id_col: str
) -> pd.DataFrame:
    """Build the two-column TSV frame, guaranteeing exactly one row per required S1 id."""
    rows = []
    for s1 in all_s1_ids:
        ids = id_lists.get(s1, ())
        # De-duplicate while keeping the output deterministic.
        joined = ",".join(sorted(set(ids)))
        rows.append((s1, joined))
    return pd.DataFrame(rows, columns=["source1_entity_id", id_col])


def write_id_list_tsv(
    id_lists: Mapping[str, Iterable[str]],
    all_s1_ids: Iterable[str],
    out_path: str | os.PathLike,
    id_col: str,
) -> Path:
    """Write a matching_results.tsv / candidate_pairs.tsv-shaped file.

    ``id_lists`` maps source1_entity_id -> iterable of S2/S3 ids. Every id in
    ``all_s1_ids`` gets exactly one row, in the exact required schema: no quoting,
    comma-joined, empty string when there are no matches/candidates.
    """
    out_path = resolve_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = _id_list_dict_to_frame(id_lists, all_s1_ids, id_col)
    df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
    return out_path


def read_id_list_tsv(path: str | os.PathLike, id_col: str) -> dict[str, set[str]]:
    """Inverse of :func:`write_id_list_tsv` -- read a candidate/matching file back in."""
    df = pd.read_csv(resolve_path(path), sep="\t", dtype=str, keep_default_na=False)
    out: dict[str, set[str]] = {}
    for s1, ids in zip(df["source1_entity_id"], df[id_col]):
        ids = ids.strip()
        out[s1] = set(x.strip() for x in ids.split(",") if x.strip()) if ids else set()
    return out
