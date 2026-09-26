"""Phase 2 -- Tier 0 deterministic block.

Straight port of the EDA's Block A: (country, house_number, exact normalized name).
Implemented as a hash-join (pandas merge on the composite key), not a nested loop,
so it runs in minutes even at 2M+ S1 rows. Its role in the fused pipeline is to be
tagged ``source_channel = high_conf`` and fed to the classifier as ~98%-precise
evidence, not thresholded away.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.normalize import normalize_dataframe
from src.utils.io import load_config, read_source_tsv, resolve_path
from src.utils.mem_monitor import log_mem

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["source1_entity_id", "candidate_entity_id", "source_channel"]


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[
        df["house_number"].fillna("").ne("")
        & df["norm_name"].fillna("").ne(""),
        ["entity_id", "norm_country", "house_number", "norm_name"],
    ]


def _country_values(s1_df: pd.DataFrame) -> list[str]:
    return [str(value) for value in s1_df["norm_country"].dropna().unique()]


def iter_tier0_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    skip_countries: set[str] | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield one deduplicated Tier 0 result per S1 country.

    The join uses the three key columns directly. This avoids constructing a
    full composite string column and bounds the merge to one country at a time.
    """
    log_mem("Tier 0 start")
    s1_valid = _valid_rows(s1_df)
    log_mem("Tier 0 after S1 key filtering")

    skip_countries = skip_countries or set()
    for country in _country_values(s1_df):
        if country in skip_countries:
            log.info("Tier 0 skipping completed country=%s", country)
            continue
        s1_part = s1_valid[s1_valid["norm_country"] == country]
        if s1_part.empty:
            continue
        log_mem(f"Tier 0 before country={country}")
        parts: list[pd.DataFrame] = []
        for other_df in (s2_df, s3_df):
            if other_df is None or other_df.empty:
                continue
            other_valid = _valid_rows(other_df)
            other_part = other_valid[other_valid["norm_country"] == country]
            if other_part.empty:
                del other_valid, other_part
                continue
            merged = s1_part.merge(
                other_part,
                on=["norm_country", "house_number", "norm_name"],
                suffixes=("_s1", "_other"),
                sort=False,
            )
            if not merged.empty:
                parts.append(
                    pd.DataFrame(
                        {
                            "source1_entity_id": merged["entity_id_s1"].astype(str),
                            "candidate_entity_id": merged["entity_id_other"].astype(str),
                            "source_channel": "high_conf",
                        }
                    )
                )
            del other_valid, other_part, merged
            gc.collect()

        if parts:
            result = pd.concat(parts, ignore_index=True)
            result = result.drop_duplicates(
                subset=["source1_entity_id", "candidate_entity_id"]
            ).reset_index(drop=True)
            del parts
        else:
            result = _empty_result()
        log_mem(f"Tier 0 after country={country}")
        yield country, result
        del result, s1_part
        gc.collect()
        log_mem(f"Tier 0 after releasing country={country}")


def tier0_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
    """Returns columns: source1_entity_id, candidate_entity_id, source_channel='high_conf'.

    Key: (country, house_number, exact_normalized_name). Expects ``s1_df``/``s2_df``/
    ``s3_df`` to already carry the normalized columns from :func:`normalize.normalize_dataframe`.
    """
    frames = [frame for _, frame in iter_tier0_candidates(s1_df, s2_df, s3_df) if not frame.empty]
    if not frames:
        return _empty_result()
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id"]
    ).reset_index(drop=True)


def write_tier0_checkpoint(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    output_dir: str | os.PathLike,
) -> Path:
    """Write resumable per-country Tier 0 files and return the manifest path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "tier0_manifest.json"
    completed: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            completed = set(json.load(handle).get("completed", []))

    for country, frame in iter_tier0_candidates(
        s1_df, s2_df, s3_df, skip_countries=completed
    ):
        final_path = output_path / f"tier0_{country}.tsv"
        temp_path = final_path.with_suffix(".tsv.tmp")
        frame.to_csv(temp_path, sep="\t", index=False)
        os.replace(temp_path, final_path)
        completed.add(country)
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            json.dump({"completed": sorted(completed)}, handle, indent=2)
        os.replace(manifest_tmp, manifest_path)
        log.info("Tier 0 checkpointed country=%s rows=%d", country, len(frame))
        log_mem(f"Tier 0 checkpointed country={country}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable Tier 0 as an isolated process.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    paths = config["paths"]
    s1 = normalize_dataframe(
        read_source_tsv(paths["train_source1"], nrows=args.sample), config, keep_raw=False
    )
    s2 = normalize_dataframe(read_source_tsv(paths["train_source2"]), config, keep_raw=False)
    s3 = normalize_dataframe(read_source_tsv(paths["train_source3"]), config, keep_raw=False)
    log_mem("Tier 0 after loading and normalization")
    output_dir = args.output_dir or resolve_path(paths["artifacts_dir"]) / "tier0"
    manifest = write_tier0_checkpoint(s1, s2, s3, output_dir)
    log.info("Tier 0 complete: %s", manifest)


if __name__ == "__main__":
    main()
