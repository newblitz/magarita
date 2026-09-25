"""Phase 4 -- Tier 2 semantic retrieval (multilingual dense embeddings + FAISS,
the LSBlock semantic channel).

1. Load the embedding model named in config.yaml (`sentence-transformers`).
   Model choice: ``paraphrase-multilingual-MiniLM-L12-v2`` -- Apache-2.0, ~118M
   params, comfortably inside the MIT/Apache-2.0, <=8B constraint, and a generic
   pretrained text encoder (no external lookup, fair-play compliant). See
   ``README.md`` for the license verification note.
2. Encode ``name`` and ``name_address`` per country partition, per source,
   batched (never one string at a time) -- and, when more than one GPU is
   visible (e.g. Kaggle's T4 x2), spread the batch across all of them via
   sentence-transformers' multi-process encoding pool (see ``EmbeddingModel``).
3. Build a FAISS index per (country, field, source) combination. When a
   GPU-enabled FAISS build is available, this is an exact brute-force
   ``IndexFlatIP`` replicated/sharded across every visible GPU (FAISS has no
   GPU implementation of HNSW); otherwise it's the CPU ``IndexHNSWFlat`` from
   the architecture report. Controlled by ``tier2_semantic.index_backend``.
4. For every S1 record, query top-k nearest neighbors from the S2 and S3
   indexes for that country partition.
5. Return raw similarity scores, unthresholded -- thresholding is a Phase 5/6
   concern, exactly as for Tier 1.

``sentence-transformers``, ``torch`` and ``faiss`` are imported lazily so the
rest of the pipeline (tiers 0/1, features, classifier, calibration) can run in
environments where these heavier optional dependencies aren't installed yet.
"""

from __future__ import annotations

import gc
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.normalize import field_column

log = logging.getLogger(__name__)


class EmbeddingModel:
    """Wraps a sentence-transformers model, loaded once and reused across every
    country partition and field.

    GPU utilization strategy (config: tier2_semantic.device/multi_gpu/fp16):
    - ``device="auto"`` (default): use every visible CUDA GPU if
      ``multi_gpu`` is true (the default), a single GPU if only one is
      visible, or CPU if none is.
    - When more than one GPU is visible and ``multi_gpu`` is true, encoding
      uses sentence-transformers' persistent multi-process pool (one worker
      process per GPU, e.g. Kaggle's T4 x2) so every ``encode()`` call is
      split across both devices -- the pool is started once in ``__init__``
      and reused for every subsequent call, not respawned per country/field.
    - ``fp16=True`` (default) runs the model in half precision on CUDA
      devices, which roughly doubles T4 throughput; ignored on CPU.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,  # reduced from 256 → 64 for CPU; config.yaml overrides this
        device: str = "auto",
        multi_gpu: bool = False,  # False by default; True only when GPU is available
        fp16: bool = False,       # False by default; True only speeds up CUDA
        require_gpu: bool = False,
    ):
        from sentence_transformers import SentenceTransformer  # lazy import
        import torch  # lazy import

        self.model_name = model_name
        self.batch_size = batch_size
        self._pool = None
        self._devices = self._resolve_devices(device, multi_gpu, torch)

        if require_gpu and not any(d.startswith("cuda") for d in self._devices):
            raise RuntimeError(
                "tier2_semantic.require_gpu is true but torch.cuda.is_available() is "
                "False -- no CUDA device is visible to this process. On Kaggle: (1) "
                "confirm Settings -> Accelerator is set to a GPU (T4 x2) and that you "
                "restarted the session after changing it, (2) run `!nvidia-smi` to "
                "confirm the GPU is attached at the OS level, (3) check that `pip "
                "install ...` didn't silently replace the preinstalled CUDA-enabled "
                "torch with a CPU-only build -- run `import torch; "
                "print(torch.__version__, torch.cuda.is_available())` right after your "
                "pip install cell to catch this immediately. Set require_gpu: false to "
                "allow a silent CPU fallback instead."
            )

        self._model = SentenceTransformer(model_name)
        uses_cuda = any(d.startswith("cuda") for d in self._devices)
        if fp16 and uses_cuda:
            self._model = self._model.half()
        elif not uses_cuda:
            self._model = self._model.to("cpu")

        if len(self._devices) > 1:
            log.info("Tier2 embedding: multi-GPU pool across %s", self._devices)
            self._pool = self._model.start_multi_process_pool(target_devices=self._devices)
        elif uses_cuda:
            self._model = self._model.to(self._devices[0])
            log.info("Tier2 embedding: single device %s", self._devices[0])
        else:
            log.info("Tier2 embedding: CPU (no CUDA device visible)")

    @staticmethod
    def _resolve_devices(device: str, multi_gpu: bool, torch_module) -> list[str]:
        if device not in ("auto", None):
            return [device]
        if not torch_module.cuda.is_available():
            return ["cpu"]
        n = torch_module.cuda.device_count()
        if n == 0:
            return ["cpu"]
        if multi_gpu and n > 1:
            return [f"cuda:{i}" for i in range(n)]
        return ["cuda:0"]

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype="float32")
        if self._pool is not None:
            embeddings = self._model.encode_multi_process(
                texts,
                self._pool,
                batch_size=self.batch_size,
                normalize_embeddings=True,  # cosine similarity == inner product
            )
        else:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        return np.asarray(embeddings, dtype="float32")

    def close(self) -> None:
        """Stop the multi-GPU worker pool (if any). Call once, when done with
        this embedder -- e.g. at the end of a `run_train`/`run_inference` phase."""
        if self._pool is not None:
            from sentence_transformers import SentenceTransformer

            SentenceTransformer.stop_multi_process_pool(self._pool)
            self._pool = None


def _faiss_gpu_available(faiss_module) -> bool:
    return hasattr(faiss_module, "StandardGpuResources") and faiss_module.get_num_gpus() > 0


def _build_index(embeddings: np.ndarray, cfg: dict):
    """Build the FAISS index for one (country, field, source) partition.

    ``tier2_semantic.index_backend``:
    - ``"auto"`` (default): GPU brute-force IndexFlatIP if a GPU-enabled FAISS
      build and at least one CUDA device are available, else CPU HNSW.
    - ``"gpu_flat"``: force GPU (raises if unavailable).
    - ``"cpu_hnsw"``: force the CPU HNSW index from the architecture report.
    """
    import faiss  # lazy import

    dim = embeddings.shape[1]
    hnsw_cfg = cfg["hnsw"]
    backend = cfg.get("index_backend", "auto")

    def _cpu_hnsw():
        index = faiss.IndexHNSWFlat(dim, hnsw_cfg["m"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = hnsw_cfg["ef_construction"]
        index.hnsw.efSearch = hnsw_cfg["ef_search"]
        index.add(embeddings)
        return index

    if backend == "cpu_hnsw":
        return _cpu_hnsw()

    want_gpu = backend in ("auto", "gpu_flat")
    if want_gpu and _faiss_gpu_available(faiss):
        try:
            cpu_index = faiss.IndexFlatIP(dim)
            cpu_index.add(embeddings)
            n_gpu = faiss.get_num_gpus()
            if cfg.get("multi_gpu", True) and n_gpu > 1:
                # Replicates the (small, per-partition) index across every
                # visible GPU so concurrent queries are served in parallel --
                # e.g. across Kaggle's T4 x2.
                return faiss.index_cpu_to_all_gpus(cpu_index)
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception as exc:  # pragma: no cover -- defensive GPU fallback
            if backend == "gpu_flat":
                raise
            log.warning("FAISS GPU index build failed (%s); falling back to CPU HNSW.", exc)

    return _cpu_hnsw()


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

    if embedder is None:
        embedder = _default_embedder(cfg)

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
                del emb
                continue
            index = _build_index(emb, cfg)
            scores, neighbors = index.search(s1_emb, k)
            for row_i, s1_id in enumerate(s1_part["entity_id"].tolist()):
                for col_j in range(k):
                    nbr_idx = neighbors[row_i, col_j]
                    if nbr_idx < 0:
                        continue
                    result_frames.append(
                        (s1_id, ids[nbr_idx], field, float(scores[row_i, col_j]))
                    )
            # Free the FAISS index + embedding matrix + result arrays now —
            # they can be several hundred MB for a large country partition.
            del index, emb, scores, neighbors
            gc.collect()

        # Free S1 embedding before the next field's encode() call.
        del s1_emb
        gc.collect()


    if not result_frames:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "field", "cosine_score", "source_channel"]
        )

    out = pd.DataFrame(
        result_frames, columns=["source1_entity_id", "candidate_entity_id", "field", "cosine_score"]
    )
    out["source_channel"] = "semantic"
    return out


def _default_embedder(cfg: dict) -> "EmbeddingModel":
    return EmbeddingModel(
        cfg["model_name"],
        cfg.get("batch_size", 64),  # default 64 for CPU; config.yaml sets this explicitly
        device=cfg.get("device", "auto"),
        multi_gpu=cfg.get("multi_gpu", True),
        fp16=cfg.get("fp16", True),
        require_gpu=cfg.get("require_gpu", False),
    )


def tier2_candidates_all_countries(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    config: dict,
    embedder: "EmbeddingModel | None" = None,
) -> pd.DataFrame:
    """Convenience wrapper: run :func:`tier2_candidates` over every country present
    in ``s1_df`` (open-set), reusing a single loaded embedding model.

    Owns (and closes) the embedder's GPU worker pool when it creates one
    itself -- pass your own ``embedder`` in if you need it to stay alive for
    further calls after this function returns.

    Calls ``gc.collect()`` between countries so that FAISS index objects and
    embedding arrays from one country are freed before the next is processed --
    critical for 13 GB RAM with many country partitions.
    """
    cfg = config["tier2_semantic"]
    owns_embedder = embedder is None
    if embedder is None:
        embedder = _default_embedder(cfg)
    try:
        countries: Iterable[str] = s1_df["norm_country"].unique()
        frames = []
        for c in countries:
            frame = tier2_candidates(s1_df, s2_df, s3_df, c, config, embedder=embedder)
            if len(frame):
                frames.append(frame)
            gc.collect()
    finally:
        if owns_embedder:
            embedder.close()
    if not frames:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "field", "cosine_score", "source_channel"]
        )
    return pd.concat(frames, ignore_index=True)
