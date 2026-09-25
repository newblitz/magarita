"""Phase 4 -- Tier 2 semantic retrieval (multilingual dense embeddings + FAISS HNSW,
the LSBlock semantic channel).

1. Load the embedding model named in config.yaml (`sentence-transformers`).
   Model choice: ``paraphrase-multilingual-MiniLM-L12-v2`` -- Apache-2.0, ~118M
   params, comfortably inside the MIT/Apache-2.0, <=8B constraint, and a generic
   pretrained text encoder (no external lookup, fair-play compliant). See
   ``README.md`` for the license verification note.
2. Encode ``name`` and ``name_address`` per country partition, per source,
   batched (never one string at a time).
3. Build a FAISS ``IndexHNSWFlat`` per (country, field, source) combination.
4. For every S1 record, query top-k nearest neighbors from the S2 and S3 indexes
   for that country partition.
5. Return raw similarity scores, unthresholded -- thresholding is a Phase 5/6
   concern, exactly as for Tier 1.

``sentence-transformers`` and ``faiss`` are imported lazily so the rest of the
pipeline (tiers 0/1, features, classifier, calibration) can run in environments
where these heavier optional dependencies aren't installed yet.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.normalize import field_column


class EmbeddingModel:
    """Thin wrapper around a sentence-transformers model, loaded once and reused
    across country partitions and fields."""

    def __init__(self, model_name: str, batch_size: int = 256):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity == inner product
        )
        return embeddings.astype("float32")


def _build_hnsw_index(embeddings: np.ndarray, m: int, ef_construction: int, ef_search: int):
    import faiss  # lazy import

    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(embeddings)
    return index


def tier2_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    country_value: str,
    config: dict,
    embedder: "EmbeddingModel | None" = None,
) -> pd.DataFrame:
    """Returns columns: source1_entity_id, candidate_entity_id, field, cosine_score,
    source_channel='semantic'.

    Runs cleanly on an unseen country (e.g. France): country is used only as a
    string-equality partition filter, never as a fixed enum/branch.
    """
    cfg = config["tier2_semantic"]
    fields = cfg["fields"]
    top_k = cfg["top_k"]
    hnsw_cfg = cfg["hnsw"]

    if embedder is None:
        embedder = EmbeddingModel(cfg["model_name"], cfg.get("batch_size", 256))

    s1_part = s1_df[s1_df["norm_country"] == country_value]
    parts = {
        "S2": s2_df[s2_df["norm_country"] == country_value] if s2_df is not None else s2_df.iloc[0:0],
        "S3": s3_df[s3_df["norm_country"] == country_value] if s3_df is not None else s3_df.iloc[0:0],
    }

    if len(s1_part) == 0:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "field", "cosine_score", "source_channel"]
        )

    result_frames = []
    for field in fields:
        column = field_column(field)
        s1_texts = s1_part[column].fillna("").tolist()
        if not s1_texts:
            continue
        s1_emb = embedder.encode(s1_texts)

        for _source_name, part_df in parts.items():
            if part_df is None or len(part_df) == 0:
                continue
            texts = part_df[column].fillna("").tolist()
            ids = part_df["entity_id"].tolist()
            emb = embedder.encode(texts)
            k = min(top_k, len(ids))
            if k == 0:
                continue
            index = _build_hnsw_index(
                emb, hnsw_cfg["m"], hnsw_cfg["ef_construction"], hnsw_cfg["ef_search"]
            )
            scores, neighbors = index.search(s1_emb, k)
            for row_i, s1_id in enumerate(s1_part["entity_id"].tolist()):
                for col_j in range(k):
                    nbr_idx = neighbors[row_i, col_j]
                    if nbr_idx < 0:
                        continue
                    result_frames.append(
                        (s1_id, ids[nbr_idx], field, float(scores[row_i, col_j]))
                    )

    if not result_frames:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "field", "cosine_score", "source_channel"]
        )

    out = pd.DataFrame(
        result_frames, columns=["source1_entity_id", "candidate_entity_id", "field", "cosine_score"]
    )
    out["source_channel"] = "semantic"
    return out


def tier2_candidates_all_countries(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    config: dict,
    embedder: "EmbeddingModel | None" = None,
) -> pd.DataFrame:
    """Convenience wrapper: run :func:`tier2_candidates` over every country present
    in ``s1_df`` (open-set), reusing a single loaded embedding model."""
    cfg = config["tier2_semantic"]
    if embedder is None:
        embedder = EmbeddingModel(cfg["model_name"], cfg.get("batch_size", 256))
    countries: Iterable[str] = s1_df["norm_country"].unique()
    frames = [
        tier2_candidates(s1_df, s2_df, s3_df, c, config, embedder=embedder) for c in countries
    ]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "field", "cosine_score", "source_channel"]
        )
    return pd.concat(frames, ignore_index=True)
